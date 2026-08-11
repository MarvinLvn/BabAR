#!/usr/bin/env python3
"""
BabAR Pipeline: VTC (or LENA from .its files) on all files, then BabAR on all files.

Usage:
    # Standard mode (WAV + VTC)
    uv run src/pipeline.py \
        --wavs audio_folder/ \
        --output results/ \
        --device cpu

    # LENA mode (WAV + ITS, skips VTC)
    uv run src/pipeline.py \
        --wavs audio_folder/ \
        --its lena_its_folder/ \
        --output results/ \
        --device cpu
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

# Add VTC submodule to path so we can import its scripts
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "VTC"))

from scripts.infer import main as vtc_infer
from babar.infer import load_model, run_single as babar_infer
from its_to_rttm import convert_file as its_convert_file

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")

def _check_not_lfs_pointer(path: Path):
    with open(path, "rb") as f:
        if f.read(200).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"{path} is a Git LFS pointer file, not the actual model weights.\n"
                f"Run from the repo root:\n"
                f"    git lfs install\n"
                f"    git submodule update --init --recursive\n"
                f"    git submodule foreach --recursive git lfs pull\n"
                f"then rerun BabAR."
            )

def resolve_device(device: str) -> str:
    """Normalize device string and check availability."""
    if device in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but not available, falling back to CPU.")
        return "cpu"
    if device == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS requested but not available, falling back to CPU.")
        return "cpu"
    return "cpu"


def _free_gpu():
    """Force garbage collection and free GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_audio_duration(wav_path: Path) -> float:
    """Return audio duration in seconds."""
    info = sf.info(wav_path)
    return info.duration


def _save_timing(timing_records: list[dict], output_path: Path):
    """Save timing records to CSV, merging with any existing data."""
    new_df = pd.DataFrame(timing_records)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_df = existing_df.set_index("filename")
        new_df = new_df.set_index("filename")
        combined = existing_df.combine_first(new_df)
        combined = combined.reset_index().sort_values("filename")
    else:
        combined = new_df.sort_values("filename")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)


def _validate_wav_files(wav_files: list[Path]):
    """Warn about wav files that are not 16kHz mono."""
    invalid = []
    for wav in wav_files:
        info = sf.info(wav)
        if info.samplerate != 16000 or info.channels != 1:
            invalid.append(
                f"  {wav.name}: {info.samplerate}Hz, {info.channels} channel(s)"
            )
    if invalid:
        raise ValueError(
            "The following files are not 16kHz mono. "
            "Please convert them first using VTC/scripts/convert.py:\n"
            + "\n".join(invalid)
        )


def run_its_conversion(its_dir: Path, rttm_dir: Path, wav_files: list[Path]) -> int:
    """Convert ITS files to RTTM, skipping files that already have an RTTM.

    Only converts ITS files that have a matching WAV in wav_files.
    Returns the number of newly converted files.
    """
    wav_stems = {w.stem for w in wav_files}
    its_files = sorted(its_dir.glob("*.its"))

    if not its_files:
        raise FileNotFoundError(f"No .its files found in {its_dir}")

    # Check all ITS files have a matching WAV
    missing_wavs = [f for f in its_files if f.stem not in wav_stems]
    if missing_wavs:
        names = ", ".join(f.name for f in missing_wavs)
        raise FileNotFoundError(
            f"The following .its files have no matching .wav in --wavs: {names}"
        )

    to_convert = [
        f for f in its_files
        if f.stem in wav_stems and not (rttm_dir / f"{f.stem}.rttm").exists()
    ]

    if not to_convert:
        logger.info(f"ITS->RTTM: all {len(its_files)} file(s) already converted, skipping.")
        return 0

    logger.info(
        f"ITS->RTTM: converting {len(to_convert)}/{len(its_files)} file(s) "
        f"({len(its_files) - len(to_convert)} already done)..."
    )
    rttm_dir.mkdir(parents=True, exist_ok=True)
    for its_path in to_convert:
        its_convert_file(its_path, rttm_dir)

    return len(to_convert)


def run_pipeline(
    wavs: Path,
    output: Path,
    checkpoint: Path,
    vocab_phoneme_path: Path,
    its: Path | None = None,
    device: str = "cpu",
    context_duration: float = 20.0,
    batch_size: int = 16,
    num_workers: int = 4,
    vtc_batch_size: int = 128,
    max_utt_dur: float = 30.0,
    high_precision: bool = False,
    transcribe_och: bool = False,
):
    """Run VTC on all files, then BabAR on all files.

    If `its` is provided, ITS files are converted to RTTM first and VTC is
    skipped entirely. Otherwise VTC runs as normal.

    Only one model is in memory at a time.
    BabAR skips individual files that already have a phoneme CSV.
    Timing information is written to <output>/timing.csv.
    """
    device = resolve_device(device)

    output.mkdir(parents=True, exist_ok=True)
    rttm_dir = output / "rttm"
    csv_dir = output / "phonemes"
    timing_path = output / "timing.csv"

    wav_files = sorted(wavs.glob("*.wav"))
    if not wav_files:
        logger.error(f"No .wav files found in {wavs}")
        return
    _validate_wav_files(wav_files)
    _check_not_lfs_pointer(checkpoint)
    if its is None:
        _check_not_lfs_pointer(REPO_ROOT / "VTC" / "VTC-2" / "model" / "best.ckpt")
    logger.info(f"Found {len(wav_files)} audio file(s). Device: {device}")

    # -- Step 1: ITS -> RTTM (LENA mode) or VTC (standard mode) -------------
    if its is not None:
        logger.info("Step 1/2: LENA mode — converting .its files to .rttm (skipping VTC).")
        run_its_conversion(its, rttm_dir, wav_files)
    else:
        wavs_needing_vtc = [
            w for w in wav_files
            if not (rttm_dir / f"{w.stem}.rttm").exists()
        ]

        if wavs_needing_vtc:
            logger.info(
                f"Step 1/2: Running VTC on {len(wavs_needing_vtc)}/{len(wav_files)} file(s) "
                f"({len(wav_files) - len(wavs_needing_vtc)} already have RTTM)..."
            )

            vtc_start = time.time()
            logger.info(f"Using {'high precision' if high_precision else 'F1'} thresholds.")
            vtc_infer(
                wavs=str(wavs),
                output=str(output),
                config=str(REPO_ROOT / "VTC" / "VTC-2" / "model" / "config.toml"),
                checkpoint=str(REPO_ROOT / "VTC" / "VTC-2" / "model" / "best.ckpt"),
                batch_size=vtc_batch_size,
                thresholds=REPO_ROOT / "VTC" / "thresholds" / ("hp.toml" if high_precision else "f1.toml"),
                device=device,
            )
            vtc_total_sec = time.time() - vtc_start

            vtc_durations = {
                w.stem: _get_audio_duration(w) for w in wavs_needing_vtc
            }
            total_audio_dur = sum(vtc_durations.values())

            vtc_timing = []
            for w in wavs_needing_vtc:
                audio_dur = vtc_durations[w.stem]
                vtc_file_sec = (
                    vtc_total_sec * audio_dur / total_audio_dur
                    if total_audio_dur > 0
                    else 0.0
                )
                vtc_timing.append({
                    "filename": w.name,
                    "audio_duration_sec": round(audio_dur, 2),
                    "vtc_sec": round(vtc_file_sec, 2),
                })

            _save_timing(vtc_timing, timing_path)
            logger.info(f"VTC total time: {vtc_total_sec:.1f}s (per-file estimates saved to {timing_path})")
        else:
            logger.info(f"Step 1/2: All {len(wav_files)} file(s) already have RTTM, skipping VTC.")

    # Collect non-empty RTTMs
    rttm_files = sorted(
        f for f in rttm_dir.glob("*.rttm")
        if f.stat().st_size > 0
    )

    if not rttm_files:
        logger.warning("No RTTM files with speech found. Nothing to transcribe.")
        return

    logger.info(f"RTTMs ready. {len(rttm_files)} file(s) with speech.")

    _free_gpu()

    # -- Step 2: BabAR on all files with RTTM --------------------------------
    rttm_needing_babar = [
        f for f in rttm_files
        if not (csv_dir / f"{f.stem}.csv").exists()
    ]

    if not rttm_needing_babar:
        logger.info(f"Step 2/2: All {len(rttm_files)} file(s) already have phoneme CSVs, skipping BabAR.")
        return

    logger.info(
        f"Step 2/2: Running BabAR on {len(rttm_needing_babar)}/{len(rttm_files)} file(s) "
        f"({len(rttm_files) - len(rttm_needing_babar)} already done, skipping)..."
    )

    model = load_model(checkpoint, vocab_phoneme_path, device=device)
    model = model.to(device)
    if device != "cpu":
        model = model.half()

    total_utterances = 0
    babar_timing = []

    for i, rttm_file in enumerate(rttm_needing_babar, 1):
        wav_file = wavs / f"{rttm_file.stem}.wav"
        if not wav_file.exists():
            logger.warning(f"No matching WAV for {rttm_file.name}, skipping.")
            continue

        logger.info(f"  BabAR [{i}/{len(rttm_needing_babar)}] {wav_file.name}")

        babar_start = time.time()
        results_df = babar_infer(
            model=model,
            audio_path=wav_file,
            rttm_path=rttm_file,
            device=device,
            context_duration=context_duration,
            batch_size=batch_size,
            num_workers=num_workers,
            max_utt_dur=max_utt_dur,
            speaker_filter=["KCHI", "OCH"] if transcribe_och else ["KCHI"],
        )
        babar_sec = time.time() - babar_start
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{rttm_file.stem}.csv"
        n_utterances = 0
        if results_df is not None and len(results_df) > 0:
            results_df["onset"] = results_df["onset"].round(3)
            results_df["offset"] = results_df["offset"].round(3)
            results_df.to_csv(csv_path, index=False)
            n_utterances = len(results_df)
            total_utterances += n_utterances
            logger.info(f"    {n_utterances} KCHI utterance(s) -> {csv_path} ({babar_sec:.1f}s)")
        else:
            pd.DataFrame(columns=["filename", "onset", "offset", "speaker", "phonemes"]).to_csv(csv_path, index=False)
            logger.info(f"    No KCHI utterances, wrote empty CSV. ({babar_sec:.1f}s)")
        babar_timing.append({
            "filename": wav_file.name,
            "audio_duration_sec": round(_get_audio_duration(wav_file), 2),
            "babar_sec": round(babar_sec, 2),
            "n_utterances": n_utterances,
        })

    _save_timing(babar_timing, timing_path)

    del model
    _free_gpu()

    logger.info(f"Done. {total_utterances} utterances across {len(rttm_needing_babar)} files -> {csv_dir}")
    logger.info(f"Timing saved to {timing_path}")


def main():
    parser = argparse.ArgumentParser(
        description="BabAR: speaker segmentation + phoneme recognition for child-centered recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required parameters
    parser.add_argument("--wavs", type=Path, required=True,
                        help="Folder containing .wav files (16kHz, mono).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory.")
    parser.add_argument("--its", type=Path, default=None,
                        help="Folder containing LENA .its files. "
                             "If provided, skips VTC and uses LENA segmentation instead.")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "gpu", "mps"],
                        help="Compute device.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for BabAR inference.")
    parser.add_argument("--vtc_batch_size", type=int, default=128,
                        help="Batch size for VTC inference (ignored in LENA mode).")

    # Advanced parameters: don't set them if you don't know what you're doing!
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "weights" / "best.ckpt",
                        help="Path to BabAR model checkpoint (.ckpt).")
    parser.add_argument("--vocab_phoneme_path", type=Path,
                        default=REPO_ROOT / "weights" / "vocab-phoneme-tinyvox.json",
                        help="Path to phoneme vocabulary JSON.")
    parser.add_argument("--context_duration", type=float, default=20.0,
                        help="Context window in seconds for BabAR.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers.")
    parser.add_argument('--max_utt_dur', type=float, default=30.0,
                        help='Maximum utterance duration in seconds.')
    parser.add_argument(
        "--high_precision",
        action="store_true",
        help="Use high-precision VTC thresholds (ignored in LENA mode).",
    )
    parser.add_argument(
        "--transcribe_och",
        action="store_true",
        help="If activated, will transcribe OCH utterances too (default: KCHI only).",
    )
    args = parser.parse_args()

    if not args.wavs.is_dir():
        parser.error(f"--wavs must be a directory: {args.wavs}")
    if args.its is not None and not args.its.is_dir():
        parser.error(f"--its must be a directory: {args.its}")
    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.vocab_phoneme_path.exists():
        parser.error(f"Vocabulary file not found: {args.vocab_phoneme_path}")

    run_pipeline(
        wavs=args.wavs,
        output=args.output,
        its=args.its,
        checkpoint=args.checkpoint,
        vocab_phoneme_path=args.vocab_phoneme_path,
        device=args.device,
        context_duration=args.context_duration,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        vtc_batch_size=args.vtc_batch_size,
        max_utt_dur=args.max_utt_dur,
        high_precision=args.high_precision,
        transcribe_och=args.transcribe_och,
    )


if __name__ == "__main__":
    main()
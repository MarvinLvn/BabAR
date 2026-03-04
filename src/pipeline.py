#!/usr/bin/env python3
"""
BabAR Pipeline: VTC on all files, then BabAR on all files.

Each model is loaded once and unloaded before the next, so only one
model occupies memory at any given time.

Usage:
    uv run src/pipeline.py \
        --wavs audio_folder/ \
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


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


def run_pipeline(
    wavs: Path,
    output: Path,
    checkpoint: Path,
    vocab_phoneme_path: Path,
    device: str = "cpu",
    context_duration: float = 20.0,
    batch_size: int = 16,
    num_workers: int = 4,
    vtc_batch_size: int = 128,
    max_utt_dur: float = 30.0,
):
    """Run VTC on all files, then BabAR on all files.

    Only one model is in memory at a time.
    VTC is skipped entirely if all files already have RTTMs, but otherwise
    re-runs on the whole folder (its API does not support per-file skipping).
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

    logger.info(f"Found {len(wav_files)} audio file(s). Device: {device}")

    # -- Step 1: VTC on all files ------------------------------------------
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
        vtc_infer(
            wavs=str(wavs),
            output=str(output),
            config=str(REPO_ROOT / "VTC" / "VTC-2.0" / "model" / "config.yml"),
            checkpoint=str(REPO_ROOT / "VTC" / "VTC-2.0" / "model" / "best.ckpt"),
            batch_size=vtc_batch_size,
            device=device,
        )
        vtc_total_sec = time.time() - vtc_start

        # VTC processes all files as a batch, so we distribute time
        # proportionally to each file's audio duration.
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

    logger.info(f"VTC done. {len(rttm_files)} file(s) with speech.")

    # Free VTC model memory before loading BabAR
    _free_gpu()

    # -- Step 2: BabAR on all files with RTTM ------------------------------
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

    model = load_model(checkpoint, vocab_phoneme_path)
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
        )
        babar_sec = time.time() - babar_start

        n_utterances = 0
        if results_df is not None and len(results_df) > 0:
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f"{rttm_file.stem}.csv"
            results_df.to_csv(csv_path, index=False)
            n_utterances = len(results_df)
            total_utterances += n_utterances
            logger.info(f"    {n_utterances} KCHI utterance(s) -> {csv_path} ({babar_sec:.1f}s)")
        else:
            logger.info(f"    No KCHI utterances. ({babar_sec:.1f}s)")

        babar_timing.append({
            "filename": wav_file.name,
            "audio_duration_sec": round(_get_audio_duration(wav_file), 2),
            "babar_sec": round(babar_sec, 2),
            "n_utterances": n_utterances,
        })

    # Save BabAR timing (merges with existing VTC timing)
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
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "gpu", "mps"],
                        help="Compute device.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for BabAR inference.")
    parser.add_argument("--vtc_batch_size", type=int, default=128,
                        help="Batch size for VTC inference.")

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
                        help='Maximum utterance duration in seconds (filter out longer utterances)')

    args = parser.parse_args()

    if not args.wavs.is_dir():
        parser.error(f"--wavs must be a directory: {args.wavs}")
    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.vocab_phoneme_path.exists():
        parser.error(f"Vocabulary file not found: {args.vocab_phoneme_path}")

    run_pipeline(
        wavs=args.wavs,
        output=args.output,
        checkpoint=args.checkpoint,
        vocab_phoneme_path=args.vocab_phoneme_path,
        device=args.device,
        context_duration=args.context_duration,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        vtc_batch_size=args.vtc_batch_size,
        max_utt_dur=30.0,
    )


if __name__ == "__main__":
    main()

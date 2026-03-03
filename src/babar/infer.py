"""
BabAR inference: phoneme recognition on KCHI segments from RTTM.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from babar.models.BaseModule import BaseModule
from babar.datamodules.contextual_vtc_datamodule import ContextualVTCDataModule

logger = logging.getLogger("babar.infer")


def load_model(
    checkpoint_path: Path,
    vocab_phoneme_path: Optional[Path] = None,
) -> BaseModule:
    """Load BabAR model from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file.
        vocab_phoneme_path: Path to phoneme vocabulary JSON.

    Returns:
        BaseModule in eval mode.
    """
    if vocab_phoneme_path is None:
        vocab_phoneme_path = Path("weights/vocab-phoneme-tinyvox.json")

    if not vocab_phoneme_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_phoneme_path}")

    logger.info(f"Loading BabAR model from {checkpoint_path}")
    model = BaseModule.load_from_checkpoint(
        str(checkpoint_path),
        vocab_phoneme_path=vocab_phoneme_path,
    )
    model.eval()
    return model


def predict_batch(model: BaseModule, batch: dict, device: str) -> list[str]:
    """Run phoneme prediction on a single batch.

    Args:
        model: Loaded BaseModule.
        batch: Batch dict from the dataloader.
        device: Device string.

    Returns:
        List of predicted phoneme sequences (one string per utterance).
    """
    with torch.no_grad():
        batch["array"] = batch["array"].to(device)

        hidden_states, input_lengths, is_valid_mask = model.get_hidden_states(batch)
        logits = model.get_logits(hidden_states)
        logits = model.mask_logits(logits, is_valid_mask)

        predictions = model.decoder.decode(logits)

    return predictions


def run_single(
    model: BaseModule,
    audio_path: Path,
    rttm_path: Path,
    device: str = "cpu",
    context_duration: float = 20.0,
    batch_size: int = 32,
    num_workers: int = 4,
    speaker_filter: str = "KCHI",
) -> Optional[pd.DataFrame]:
    """Run BabAR inference on a single (audio, rttm) pair.

    Args:
        model: Loaded BaseModule (already on device).
        audio_path: Path to .wav file.
        rttm_path: Path to .rttm file.
        device: Device string.
        context_duration: Context window in seconds.
        batch_size: Batch size.
        num_workers: Dataloader workers.
        speaker_filter: Speaker label to extract from RTTM.

    Returns:
        DataFrame with columns [filename, onset, offset, speaker, phonemes],
        or None if no utterances were found.
    """
    datamodule = ContextualVTCDataModule(
        audio_path=audio_path,
        rttm_path=rttm_path,
        context_duration=context_duration,
        batch_size=batch_size,
        num_workers=num_workers,
        speaker_filter=speaker_filter,
    )
    datamodule.set_processor(model.processor)
    datamodule.setup()

    if len(datamodule.dataset) == 0:
        logger.info(f"No {speaker_filter} utterances in {rttm_path.name}")
        return None

    dataloader = datamodule.dataloader()
    results = []

    for batch in tqdm(dataloader, desc=f"  {audio_path.stem}", leave=False):
        predictions = predict_batch(model, batch, device)

        for i, pred in enumerate(predictions):
            results.append(
                {
                    "filename": audio_path.name,
                    "onset": batch["utterance_onset_sec"][i],
                    "offset": batch["utterance_onset_sec"][i]
                    + batch["utterance_duration_sec"][i],
                    "speaker": batch["speaker"][i],
                    "phonemes": pred,
                }
            )

    if not results:
        return None

    return pd.DataFrame(
        results, columns=["filename", "onset", "offset", "speaker", "phonemes"]
    )
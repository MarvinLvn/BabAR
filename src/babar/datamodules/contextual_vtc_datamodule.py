import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import soundfile as sf
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

logger = logging.getLogger("babar.datamodule")


class ContextualVTCDataModule(LightningDataModule):

    def __init__(self, audio_path, rttm_path, context_duration=20.0,
                 batch_size=32, num_workers=4, speaker_filter='KCHI', max_utt_dur=None):
        super().__init__()

        self.audio_path = Path(audio_path)
        self.rttm_path = Path(rttm_path)
        self.context_duration = context_duration
        self.context_duration_ms = self.context_duration * 1000
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.speaker_filter = speaker_filter
        self.max_utt_dur = max_utt_dur

        self.sampling_rate = 16000
        self.processor = None

    def parse_rttm(self) -> List[Dict]:
        """Parse RTTM file and extract utterances for specified speaker type."""
        utterances = []
        filtered_count = 0

        with open(self.rttm_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('SPEAKER'):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    continue

                onset = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]

                if speaker == self.speaker_filter:
                    if self.max_utt_dur is not None and duration > self.max_utt_dur:
                        filtered_count += 1
                        continue

                    utterances.append({
                        'onset': onset * 1000,
                        'offset': (onset + duration) * 1000,
                        'duration': duration * 1000,
                        'speaker': speaker,
                    })

        logger.info(f"Found {len(utterances)} utterances for speaker '{self.speaker_filter}'")
        if self.max_utt_dur is not None and filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} utterances longer than {self.max_utt_dur}s")
        return utterances

    def _create_contextual_metadata(self, utterances: List[Dict]) -> List[Dict]:
        """Create contextual sample metadata from utterance metadata."""
        info = sf.info(self.audio_path)
        total_duration_ms = info.duration * 1000

        contextual_samples = []
        for idx, utt in enumerate(utterances):
            sample = self._create_context_metadata_for_utterance(
                utt, total_duration_ms, idx
            )
            if sample:
                contextual_samples.append(sample)

        return contextual_samples

    def _create_context_metadata_for_utterance(self, target_utt: Dict,
                                               total_duration_ms: float,
                                               utterance_id: int) -> Dict:
        """Create metadata for a contextual sample centered around a target utterance."""
        target_onset = target_utt['onset']
        target_offset = target_utt['offset']

        target_center = (target_onset + target_offset) / 2
        desired_start = target_center - self.context_duration_ms / 2
        desired_end = target_center + self.context_duration_ms / 2

        context_start = max(0, min(desired_start, target_onset))
        context_end = max(desired_end, target_offset)
        context_start = max(0, context_start)
        context_end = min(total_duration_ms, context_end)

        context_duration_ms = context_end - context_start
        target_start_in_context = target_onset - context_start
        target_end_in_context = target_offset - context_start

        estimated_frame_rate = 50.0
        target_start_frame = round(target_start_in_context * estimated_frame_rate / 1000.0)
        target_end_frame = round(target_end_in_context * estimated_frame_rate / 1000.0)
        target_start_frame = max(0, target_start_frame)
        target_end_frame = max(target_start_frame + 1, target_end_frame)

        return {
            'audio_path': str(self.audio_path),
            'utterance_id': utterance_id,
            'speaker': target_utt['speaker'],
            'utterance_onset_sec': target_onset / 1000.0,
            'utterance_duration_sec': target_utt['duration'] / 1000.0,
            'context_start_ms': float(context_start),
            'context_duration_ms': float(context_duration_ms),
            'target_start_ms': float(target_start_in_context),
            'target_end_ms': float(target_end_in_context),
            'target_start_frame': target_start_frame,
            'target_end_frame': target_end_frame,
        }

    def set_processor(self, processor):
        self.processor = processor

    def setup(self, stage=None):
        if self.processor is None:
            raise ValueError("Processor must be set before calling setup().")

        utterances = self.parse_rttm()

        if len(utterances) == 0:
            self.dataset = Dataset.from_list([])
            return

        contextual_samples = self._create_contextual_metadata(utterances)
        self.dataset = Dataset.from_list(contextual_samples)

    def _load_audio_segment(self, audio_path, offset_ms, duration_ms):
        offset_samples = int(offset_ms * self.sampling_rate / 1000.0)
        duration_samples = int(duration_ms * self.sampling_rate / 1000.0)

        audio, sr = sf.read(
            audio_path,
            start=offset_samples,
            stop=offset_samples + duration_samples,
            dtype='float32',
        )

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self.sampling_rate:
            raise ValueError(
                f"Sample rate mismatch in {audio_path}: "
                f"expected {self.sampling_rate}, got {sr}"
            )

        return audio

    def collate_fn(self, batch):
        context_audios = []
        valid_samples = []

        max_duration_ms = max(sample['context_duration_ms'] for sample in batch)
        expected_length = int(self.sampling_rate * max_duration_ms / 1000.0)

        for sample in batch:
            audio = self._load_audio_segment(
                sample['audio_path'],
                sample['context_start_ms'],
                sample['context_duration_ms'],
            )

            if len(audio) < expected_length:
                audio = np.pad(
                    audio,
                    (0, expected_length - len(audio)),
                    mode='constant',
                    constant_values=0.0,
                )

            context_audios.append(audio)
            valid_samples.append(sample)

        if not context_audios:
            raise ValueError("No valid audio samples in batch")

        processed = self.processor(
            context_audios,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        return {
            "array": processed["input_values"],
            "path": [s["audio_path"] for s in valid_samples],
            "target_frame_start": [s["target_start_frame"] for s in valid_samples],
            "target_frame_end": [s["target_end_frame"] for s in valid_samples],
            "target_start_ms": [s["target_start_ms"] for s in valid_samples],
            "target_end_ms": [s["target_end_ms"] for s in valid_samples],
            "utterance_id": [s["utterance_id"] for s in valid_samples],
            "utterance_onset_sec": [s["utterance_onset_sec"] for s in valid_samples],
            "utterance_duration_sec": [s["utterance_duration_sec"] for s in valid_samples],
            "speaker": [s["speaker"] for s in valid_samples],
        }

    def dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
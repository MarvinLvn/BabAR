import torch
from pytorch_lightning import LightningModule
from transformers import (
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)

from babar.decoders.decoders import CTCGreedyDecoder
from babar.models.acoustic_models import get_model


class BaseModule(LightningModule):
    def __init__(self, network_param, optim_param, vocab_phoneme_path=None):
        """
        method used to define our model parameters
        """
        super(BaseModule, self).__init__()
        self.save_hyperparameters()

        # Tokenizer
        if vocab_phoneme_path is not None:
            network_param.vocab_file = vocab_phoneme_path

        self.phonemes_tokenizer = Wav2Vec2PhonemeCTCTokenizer(
            vocab_file=network_param.vocab_file,
            eos_token=network_param.eos_token,
            bos_token=network_param.bos_token,
            unk_token=network_param.unk_token,
            pad_token=network_param.pad_token,
            word_delimiter_token=network_param.word_delimiter_token,
            do_phonemize=False,
        )

        network_param.vocab_size = self.phonemes_tokenizer.vocab_size

        # Blank ID for CTC masking
        self.phoneme_blank_id = self.phonemes_tokenizer.encoder[
            network_param.word_delimiter_token
        ]

        # Feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )

        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=self.phonemes_tokenizer
        )

        # Model
        self.model = get_model(network_param.network_name, network_param)

        # Decoder (greedy only for inference)
        self.decoder = CTCGreedyDecoder(self.phonemes_tokenizer)

    def forward(self, x):
        output = self.model(x)
        return output

    def get_hidden_states(self, batch):
        """Get hidden states from encoder and extract target frames"""
        hidden_states = self.model(batch['array']).last_hidden_state

        target_frame_starts = torch.tensor(batch['target_frame_start'], dtype=torch.long)
        target_frame_ends = torch.tensor(batch['target_frame_end'], dtype=torch.long)

        frame_lengths = target_frame_ends - target_frame_starts
        max_target_frames = frame_lengths.max().item()

        batch_indices = torch.arange(hidden_states.shape[0]).unsqueeze(1)
        frame_indices = torch.arange(max_target_frames).unsqueeze(0)
        absolute_indices = target_frame_starts.unsqueeze(1) + frame_indices
        absolute_indices = torch.clamp(absolute_indices, 0, hidden_states.shape[1] - 1)

        hidden_states = hidden_states[batch_indices, absolute_indices]
        is_valid_mask = frame_indices < frame_lengths.unsqueeze(1)
        input_lengths = frame_lengths

        return hidden_states, input_lengths, is_valid_mask

    def get_logits(self, hidden_states):
        return self.model.phoneme_head(hidden_states)

    def mask_logits(self, logits, is_valid_mask):
        blank_logits = torch.full_like(logits[0, 0], float('-inf'))
        blank_logits[self.phoneme_blank_id] = 10.0
        masked_logits = logits.clone()
        masked_logits[~is_valid_mask] = blank_logits
        return masked_logits
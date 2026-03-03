import torch.nn as nn
from transformers import HubertModel, HubertConfig


def _make_mlp_head(input_size, output_size, hidden_ratio=0.5, dropout=0.1):
    hidden_size = int(input_size * hidden_ratio)
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size),
    )


class AcousticModel(nn.Module):
    """Acoustic model with encoder and phoneme prediction head."""

    def __init__(self, encoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.phoneme_head = _make_mlp_head(
            encoder.config.hidden_size, vocab_size, hidden_ratio=0.5, dropout=0.1
        )

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        encoder_outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        return type(
            "ModelOutput",
            (),
            {
                "logits": None,
                "hidden_states": encoder_outputs.hidden_states
                if output_hidden_states
                else None,
                "last_hidden_state": encoder_outputs[0],
            },
        )()

def BabyHubert(params):
    """Create an empty BabyHubert encoder (weights filled by load_from_checkpoint)."""
    config = HubertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=False,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        final_dropout=0.0,
    )
    return HubertModel(config)


def get_model(model_name, params):
    """Build AcousticModel with the named encoder."""
    encoder = BabyHubert(params)
    return AcousticModel(encoder=encoder, vocab_size=params.vocab_size)
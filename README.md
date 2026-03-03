BabAR is an end-to-end algorithm for phoneme recognition from child-centered long-form audio recordings (although it can be applied to any recordings).
It combines two tools:
1. VTC 2.0 (Voice Type Classifier) to detect when the child speaks
2. BabAR (Babbling Automatic Recognition) to transcribe child speech segments into IPA phonemes

Given a folder of audio files, the pipeline produces .csv files with the onset & offset of each detected utterance.
The key-child's utterances are further enriched with their phonetic transcription. 

# Installation

First, make sure that [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/), and [git-lfs](https://git-lfs.com/) are installed on your system.
You can check that they are by running:

```sh
./check_sys_dependencies.sh
```

You can the clone the repository:

```sh
# Clone repository
git-lfs install
git clone --recurse-submodules https://github.com/MarvinLvn/BabAR.git

# Install python dependencies
cd BabAR
uv sync
```


# Citation

# Acknowledgments

# BabAr

├── pyproject.toml
├── README.md
├── check_sys_dependencies.sh
├── VTC-2.0/                        # git submodule (unchanged)
├── weights/
│   └── babar/                      # BabAR checkpoint + vocab json
├── src/
│   ├── vtc/
│   │   ├── __init__.py
│   │   ├── infer.py                # from VTC scripts/infer.py (minimal edits)
│   │   └── convert.py              # from VTC scripts/convert.py
│   ├── babar/
│   │   ├── __init__.py
│   │   ├── infer.py                # from BabAR infer.py, stripped of OOM retry logic
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── BaseModule.py       # inference-only: keep load_from_checkpoint, forward, get_hidden_states, get_logits, mask_logits, decoder
│   │   │   └── acoustic_models.py  # keep AcousticModel + only the encoders you ship (e.g. BabyHubert)
│   │   ├── datamodules/
│   │   │   ├── __init__.py
│   │   │   └── contextual_vtc_datamodule.py  # unchanged
│   │   ├── decoders/
│   │   │   ├── __init__.py
│   │   │   ├── decoders.py         # CTCGreedyDecoder only (drop beam search → drops torchaudio.models.decoder, kenlm)
│   │   │   └── pipeline.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── logger.py
│   └── pipeline.py                 # the glue
└── scripts/
    └── run.sh
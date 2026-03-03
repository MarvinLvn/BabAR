BabAR is an end-to-end algorithm for phoneme recognition from child-centered long-form audio recordings (although it can be applied to any recordings).
It combines two tools:
1. VTC 2.0 (Voice Type Classifier) to detect when the child speaks
2. BabAR (Babbling Automatic Recognition) to transcribe child speech segments into IPA phonemes

Given a folder of audio files, the pipeline produces .csv files with the onset & offset of each detected utterance.
The key-child's utterances are further enriched with their phonetic transcription. 

## Installation

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

## Usage

```bash
uv run src/pipeline.py \
    --wavs path/to/audio_folder/ \
    --output results/ \
    --device cpu
```

where:
- `--wavs`: folder containing your `.wav` files (must be 16 kHz, mono)
- `--output`: directory where RTTMs and phoneme CSVs will be saved
- `--device`:  use `cpu` to run on your processor (slower but always works), or `gpu` if your machine has an NVIDIA graphics card (much faster)

Note that input audio files must be `.wav`, sampled at **16 kHz, **mono**. If your files are in a different format, you can convert them using:
```shell
uv run VTC/scripts/convert.py --wavs raw_audio/ --output converted_audio/
```

## Output

The pipeline produces one `.csv` per recording in `<output>/phonemes/`:
```
results/
├── rttm/
│   ├── recording1.rttm
│   └── recording2.rttm
└── phonemes/
    ├── recording1.csv
    └── recording2.csv
```

Each `.csv` has the following format:

```csv
filename,onset,offset,speaker,phonemes
recording1.wav,12.34,12.89,KCHI,b a b a
recording1.wav,15.01,15.67,KCHI,m a m a
```

# Citation

# Acknowledgments

# BabAr

BabAR/
├── src/
│   ├── pipeline.py              # Main entry point
│   └── babar/
│       ├── infer.py              # Phoneme recognition
│       ├── models/
│       │   ├── BaseModule.py     # Inference-only Lightning module
│       │   └── acoustic_models.py
│       ├── datamodules/
│       │   └── contextual_vtc_datamodule.py
│       └── decoders/
│           └── decoders.py       # CTC greedy decoder
├── VTC/                          # git submodule (github.com/LAAC-LSCP/VTC)
│   ├── scripts/
│   │   ├── infer.py              # VTC inference (called by pipeline)
│   │   └── convert.py            # Audio format conversion
│   └── VTC-2.0/                  # nested submodule (HuggingFace, model weights)
├── weights/
│   ├── best.ckpt                 # BabAR model checkpoint
│   └── vocab-phoneme-tinyvox.json
├── pyproject.toml
├── .gitmodules
└── README.md
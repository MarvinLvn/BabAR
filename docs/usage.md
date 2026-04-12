## 1. Usage

To run BabAR, you can use the following command:

```bash
uv run src/pipeline.py \
    --wavs path/to/audio_folder/ \
    --output babar_my_dataset/ \
    --device cpu
```

where:
- `--wavs`: path to folder containing your `.wav` files (must be 16 kHz, mono)
- `--output`: path to a folder where RTTMs and phoneme CSVs will be saved
- `--device`:  use `cpu` to run on your processor (slower but always works), or `gpu` if your machine has an NVIDIA graphics card (much faster)

Note that input audio files must be `.wav`, sampled at **16 kHz**, **mono**. If your files are in a different format, you can convert them using:
```shell
uv run VTC/scripts/convert.py --wavs raw_audio/ --output converted_audio/
```

## 2. Output

The pipeline writes all results to the `--output` directory:

```
babar_my_dataset/
├── rttm/
│   ├── recording1.rttm
│   └── recording2.rttm
└── phonemes/
    ├── recording1.csv
    └── recording2.csv
```

### 2.1 RTTM files

VTC 2.0 produces one `.rttm` per recording in `<output>/rttm/`. Each line describes a detected speech/vocalization utterance:

```
SPEAKER recording1 1 12.340 0.550 <NA> <NA> KCHI <NA> <NA>
SPEAKER recording1 1 15.010 0.660 <NA> <NA> FEM <NA> <NA>
```

Speaker labels include `KCHI` (key child), `OCH` (other child), `FEM` (female adult), `MAL` (male adult)

The RTTM format has 10 space-separated columns. Most can be ignored — the relevant ones are:

| Column | Position | Description |
|--------|----------|-------------|
| `filename` | 2nd | Name of the source audio file |
| `onset` | 4th | Start time of the utterance in seconds |
| `duration` | 5th | Duration of the utterance in seconds |
| `speaker` | 8th | Speaker type label |

Speaker labels include `KCHI` (key child), `OCH` (other child), `FEM` (female adult), and `MAL` (male adult).

### 2.2 Phoneme CSV files

BabAR produces one `.csv` per recording in `<output>/phonemes/`. Only key-child utterances are transcribed:

```csv
filename,onset,offset,speaker,phonemes
recording1.wav,12.34,12.89,KCHI,b a b a
recording1.wav,15.01,15.67,KCHI,m a m a
```

| Column | Description |
|--------|-------------|
| `filename` | Name of the source audio file |
| `onset` | Utterance start time in seconds |
| `offset` | Utterance end time in seconds |
| `speaker` | Speaker label (always `KCHI`) |
| `phonemes` | Space-separated IPA phoneme transcription |

Note that some rows may have an empty phonemes field. This happens for very short utterances, faint speech sounds, or when VTC 2.0 misclassifies a non-speech segment as child speech.

## 3. Enriching phoneme CSVs with syllabification

To enrich the phoneme CSVs with syllabification and CV patterns, run:

```bash
uv run src/syllabify.py babar_my_dataset/phonemes/
```

Enriched CSV files are saved to `babar_my_dataset/phonemes_enriched/`. Each file contains the original columns plus:

| Column | Description |
|--------|-------------|
| `syllables` | Syllables separated by `\|`, phonemes within a syllable separated by spaces (e.g. `æ\|b u`) |
| `n_syllables` | Number of syllables |
| `n_phonemes` | Number of phonemes |
| `cv` | CV pattern, with glides treated as vowels (e.g. `V C V`) |

Syllabification uses the Sonority Sequencing Principle (Clements, 1990). No language-specific resources are required.

## 4. GPU memory tip

If you run into out-of-memory errors when running the model on GPU, try reducing `--batch_size` (e.g., 8 or 16).
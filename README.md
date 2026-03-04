<p align="center">
  <img src="docs/BabAR_logo.png" alt="BabAR logo" width="600">
</p>


BabAR is an end-to-end algorithm for phoneme recognition from child-centered long-form audio recordings (although it can be applied to any recordings).
It combines two tools:
1. VTC 2.0 (Voice Type Classifier) to detect when the child speaks
2. BabAR (Babbling Automatic Recognition) to transcribe child speech segments into IPA phonemes

Given a folder of audio files, the pipeline produces .csv files with the onset & offset of each detected utterance.
The key-child's utterances are further enriched with their phonetic transcription.

### How to use?

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Running time](docs/running_time.md)
- [(Advanced) Downloading or recreating TinyVox (BabAR's training data)](https://github.com/MarvinLvn/tinyvox)
- [(Advanced) Retraining BabAR](https://github.com/MarvinLvn/BabAR_training/)

### Citation

### Acknowledgments

We acknowledge funding from the Simons Foundation International (funding from The Simons Foundation International (034070-00033) and the National Institutes of Health (NIH, grant number DP5-OD019812).
We gratefully acknowledge PhonBank, funded by NIH-NICHD grant RO1-HD051698, and thank the data contributors whose corpora made this research possible.
HPC resources from GENCI-IDRIS (Grant 2025-A0181011829).
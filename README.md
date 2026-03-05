<div align="justify">

<p align="center">
  <img src="docs/BabAR_logo.png" alt="BabAR logo" width="500">
</p>


BabAR is an end-to-end algorithm for phoneme recognition from child-centered long-form audio recordings (although it can be applied to any recordings).
It combines two tools:
1. VTC 2.0 (Voice Type Classifier) to detect when the child speaks
2. BabAR (Babbling Automatic Recognition) to transcribe child speech segments into IPA phonemes

Given a folder of audio files, the pipeline produces .csv files with the onset & offset of each detected utterance.
The key-child's utterances are further enriched with their phonetic transcription.
</div>

You can listen to some audio samples on the [project page](https://marvinlvn.github.io/projects/babar/).

### How to use?

- [How to install BabAR?](docs/installation.md)
- [How to run BabAR?](docs/usage.md)
- [What sounds are predicted?](docs/phoneme_inventory.md)
- [How long does it take to run BabAR?](docs/running_time.md)

### For machine learners

- [(Advanced) Downloading or recreating TinyVox (BabAR's training data)](https://github.com/MarvinLvn/tinyvox)
- [(Advanced) Retraining BabAR](https://github.com/MarvinLvn/BabAR_training/)

### References

BabAR preprint:

```bibtex
@misc{lavechin_babar,
    title={BabAR: from phoneme recognition to developmental measures of young children's speech production}, 
    author={Marvin Lavechin and Elika Bergelson and Roger Levy},
    year={2026},
    eprint={incoming},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={incoming}, 
}
```

VTC 2.0 preprint:

```bibtex
@misc{charlot_vtc2,
    title={BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings}, 
    author={Théo Charlot and Tarek Kunze and Maxime Poli and Alejandrina Cristia and Emmanuel Dupoux and Marvin Lavechin},
    year={2025},
    eprint={2509.15001},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2509.15001}, 
}
```

### Acknowledgments

<div align="justify">
We acknowledge funding from the Simons Foundation International (funding from The Simons Foundation International (034070-00033) and the National Institutes of Health (NIH, grant number DP5-OD019812).
We gratefully acknowledge PhonBank, funded by NIH-NICHD grant RO1-HD051698, and thank the data contributors whose corpora made this research possible.
This work was performed using HPC resources from GENCI-IDRIS (Grant 2025-A0181011829).
</div>
# Changelog

## [1.1.0] - 2026-04-10
**Commit:** `a7c67653cba3afed5ee8317598d8825583c6a0d2`

### Added
- **High precision mode**: `--high_precision` flag in `src/pipeline.py` to use stricter VTC thresholds (`thresholds/hp.toml`), reducing false positives at the cost of recall.

### Changed
- Updated VTC submodule from VTC 2.0 to VTC 2.1, improving average F1-score from 64.6 to 66.9. VTC 2.1 also produces fewer spuriously long utterances.
- Updated `segma` dependency to `9c398231011a7b16b76cf7be8e99c8b4dbc7ac35` to support TOML config and threshold files.
- VTC model config path updated from `VTC-2.0/model/config.yml` to `VTC-2/model/config.toml`.
- VTC threshold files switched from YAML to TOML format.

---

## [1.0.0] - 2026-03-31
**Commit:** `dfd07b95f1a9fb08d70aeb6a318261c05795fa35`

### Initial Interspeech release
- End-to-end pipeline combining VTC 2.0 (speaker segmentation) and BabAR (phoneme recognition) for child-centered long-form recordings.
- Outputs per-utterance phoneme transcriptions in IPA for key-child (KCHI) speech segments.

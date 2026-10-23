# Changelog

## [1.4.2] 2026-10-23
**Commit:** `8989a4c4e095f94ccdcbd88174a43928cc1169e5`

### Fixed
- Syllabification (`src/syllabify.py`): consonants of equal sonority between two vowels are now split between the two syllables, e.g. `b a n n a` → `b a n|n a` (was `b a|n n a`). This matches how human annotators syllabify these clusters. Syllable counts are unchanged; canonical syllable counts can increase slightly, since a vowel that gains a coda consonant forms a canonical syllable.

## [1.4.1] 2026-10-21
**Commit:** `7907719d22754972bae06c1fc8982e700b441772`

### Fixed
- Syllabification (`src/syllabify.py`): a syllabic nasal no longer takes an onset; consonants between a vowel and a following syllabic nasal close the vowel's syllable, e.g. `u m m` → `u m|m` (was `u|m m`). This matches how human annotators syllabify these sequences. Syllable counts are unchanged.

## [1.4.0] 2026-10-21
**Commit:** `34bb6427103e87584dbb408bce857f3b8547ed14`

### Changed
- Syllabification (`src/syllabify.py`): nasals with no adjacent vowel are now syllable nuclei (syllabic nasals), e.g. `m m` → `m|m` and `m m a` → `m|m a` instead of a single syllable. This matches how human annotators syllabify children's nasal sequences and increases the number of (non-canonical) syllables in vocalizations containing nasal runs.

## [1.3.0] 2026-06-07
**Commit** 

### Changed

Now use VTC 2.2 (that uses a sliding window at inference) slightly improving speech detection overall

## [1.2.0] - 2026-05-19
**Commit:** `5ffed283994372aa2be1692b6241749e5a1e0c39`

### Added
- BabAR can now be run on LENA-detected segments

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

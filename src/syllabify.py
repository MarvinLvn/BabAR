#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enriches phoneme CSV files with syllabification, CV patterns, and phoneme counts.

Usage:
    uv run src/syllabify.py <input_folder>

The input folder should contain .csv files with columns:
    filename, onset, offset, speaker, phonemes

Results are saved to a 'phonemes_enriched' folder at the same level as the input folder.
The phoneme sonority table is generated automatically from weights/vocab-phoneme-tinyvox.json
and cached as weights/phoneme_sonority.tsv.
"""

import json
import glob
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning, module="panphon")

import pandas as pd
import panphon
from tqdm import tqdm

ft = panphon.FeatureTable()

GLIDES = {"j", "w", "ɥ", "h"}

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = REPO_ROOT / "weights" / "vocab-phoneme-tinyvox.json"
SONORITY_PATH = REPO_ROOT / "weights" / "phoneme_sonority.tsv"


def get_sonority(phoneme: str) -> tuple[int, str]:
    """
    Derive sonority level and category from panphon features.
    Based on Clements (1990) sonority hierarchy.

    Returns (sonority_level, category_name)
    """
    segs = ft.fts(phoneme)
    if segs is None:
        return 0, "unknown"

    f = {feat: val for feat, val in zip(ft.names, segs.numeric())}
    syl  = f['syl']
    son  = f['son']
    cons = f['cons']
    nas  = f['nas']
    cont = f['cont']

    if phoneme in GLIDES:         return 5, "glide"
    if syl == 1:                  return 6, "vowel"
    if son == 1 and nas == 1:     return 3, "nasal"
    if son == 1:                  return 4, "liquid"
    if son == -1 and cont == 1:   return 2, "fricative"
    if son == -1 and cont == -1:  return 1, "stop"
    return 0, "unknown"


def build_sonority_table(vocab_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Build and save the phoneme sonority table from the vocab JSON file.
    Only called once if the file does not exist yet.
    """
    print(f"Building phoneme sonority table from {vocab_path}...")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    phonemes = [p for p in vocab.keys() if p != "<blank>"]

    rows = []
    for phoneme in phonemes:
        sonority, category = get_sonority(phoneme)
        ptype = "vowel" if category == "vowel" else "consonant"
        rows.append({
            "phoneme": phoneme,
            "sonority": sonority,
            "category": category,
            "type": ptype,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Phoneme sonority table saved to {output_path}")
    return df


def load_sonority_table(sonority_path: Path, vocab_path: Path) -> tuple[dict, dict]:
    """Load sonority table, building it first if it doesn't exist."""
    if not sonority_path.exists():
        df = build_sonority_table(vocab_path, sonority_path)
    else:
        df = pd.read_csv(sonority_path, sep="\t")

    sonority_table = dict(zip(df["phoneme"], df["sonority"]))
    type_table = dict(zip(df["phoneme"], df["type"]))
    return sonority_table, type_table


def syllabify_ssp(phonemes: list[str], sonority_table: dict) -> list[list[str]]:
    """
    Syllabify a list of IPA phonemes using the Sonority Sequencing Principle.
    Returns a list of syllables, each syllable being a list of phonemes.
    """
    if not phonemes:
        return []
    if len(phonemes) == 1:
        return [phonemes]

    sonority = [sonority_table.get(p, 0) for p in phonemes]
    nuclei = [i for i, s in enumerate(sonority) if s == 6]

    if not nuclei:
        return [phonemes]
    if len(nuclei) == 1:
        return [phonemes]

    boundaries = [0]
    for n1, n2 in zip(nuclei, nuclei[1:]):
        between = list(range(n1 + 1, n2))
        if not between:
            boundaries.append(n2)
        else:
            min_pos = min(between, key=lambda i: sonority[i])
            boundaries.append(min_pos)
    boundaries.append(len(phonemes))

    return [phonemes[boundaries[i]:boundaries[i+1]]
            for i in range(len(boundaries) - 1)]


def phoneme_to_cv(p: str, type_table: dict) -> str:
    return "V" if type_table.get(p) == "vowel" or p in GLIDES else "C"


def to_cv(syllables: list[list[str]], type_table: dict) -> str:
    """Convert syllabified phonemes to a CV string with | as syllable separator.

    Each syllable's phonemes are mapped to C/V and joined by spaces;
    syllables are separated by |, mirroring the 'syllables' column format.
    Example: [['b', 'a'], ['b', 'a']] -> 'C V|C V'
    """
    return "|".join(
        " ".join(phoneme_to_cv(p, type_table) for p in syl)
        for syl in syllables
    )


def enrich(input_folder: str, anonymize: bool = False):
    # Define output folder
    parent = os.path.dirname(os.path.abspath(input_folder))
    output_folder = os.path.join(parent, "phonemes_enriched")
    os.makedirs(output_folder, exist_ok=True)

    # Load (or build) sonority table
    sonority_table, type_table = load_sonority_table(SONORITY_PATH, VOCAB_PATH)

    # Load all CSV files
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return

    print(f"Found {len(csv_files)} CSV files in {input_folder}")

    for f in tqdm(csv_files, desc="Processing files"):
        df = pd.read_csv(f)

        # Parse phoneme lists, handling empty/NaN phoneme fields
        df["phoneme_list"] = df["phonemes"].astype(str).str.split().apply(
            lambda x: x if isinstance(x, list) and x != ["nan"] else []
        )

        # Syllabify — keep the list for cv computation, then stringify
        df["syllable_list"] = df["phoneme_list"].apply(
            lambda p: syllabify_ssp(p, sonority_table)
        )
        df["syllables"] = df["syllable_list"].apply(
            lambda syls: "|".join(" ".join(syl) for syl in syls)
        )

        # Count syllables and phonemes
        df["n_syllables"] = df["syllable_list"].apply(len).replace(0, 1)
        df["n_phonemes"] = df["phoneme_list"].apply(len)

        # CV pattern — syllable-separated, mirrors 'syllables' column
        df["cv"] = df["syllable_list"].apply(
            lambda syls: to_cv(syls, type_table)
        )

        # Canonical syllable count and utterance-level canonical flag
        df["n_canonical_syllables"] = df["syllable_list"].apply(
            lambda syls: sum(
                1 for syl in syls
                if any(type_table.get(p) != "vowel" and p not in GLIDES for p in syl)
                and any(type_table.get(p) == "vowel" or p in GLIDES for p in syl)
            )
        )
        df["is_canonical"] = df["n_canonical_syllables"] > 0

        # Drop intermediate columns
        df = df.drop(columns=["phoneme_list", "syllable_list"])

        # Anonymize: replace phoneme-level content with <hidden>
        if anonymize:
            df["phonemes"] = "<hidden>"
            df["syllables"] = "<hidden>"

        # Save to output folder
        output_path = os.path.join(output_folder, os.path.basename(f))
        df.to_csv(output_path, index=False)

    print(f"Enriched files saved to {output_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich BabAR phoneme CSVs with syllabification and CV patterns."
    )
    parser.add_argument("input_folder", help="Folder containing per-file phoneme CSVs.")
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Replace 'phonemes' and 'syllables' columns with <hidden> in the output.",
    )
    args = parser.parse_args()

    enrich(args.input_folder, anonymize=args.anonymize)
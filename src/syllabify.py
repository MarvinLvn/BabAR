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

GLIDES = {"j", "w", "ɥ"}

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


def to_cv(phonemes: list[str], type_table: dict) -> str:
    """Convert a list of phonemes to a C/V string, treating glides as vowels."""
    result = []
    for p in phonemes:
        if type_table.get(p) == "vowel" or p in GLIDES:
            result.append("V")
        else:
            result.append("C")
    return " ".join(result)


def enrich(input_folder: str):
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

        # Syllabify
        df["syllables"] = df["phoneme_list"].apply(
            lambda syls: "|".join(
                " ".join(syl) for syl in syllabify_ssp(syls, sonority_table)
            )
        )

        # Count syllables and phonemes
        df["n_syllables"] = df["syllables"].apply(
            lambda s: len(s.split("|")) if s else 0
        )
        df["n_phonemes"] = df["phoneme_list"].apply(len)

        # CV pattern
        df["cv"] = df["phoneme_list"].apply(
            lambda p: to_cv(p, type_table)
        )

        # Drop intermediate column
        df = df.drop(columns=["phoneme_list"])

        # Save to output folder
        output_path = os.path.join(output_folder, os.path.basename(f))
        df.to_csv(output_path, index=False)

    print(f"Enriched files saved to {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run src/syllabify.py <input_folder>")
        sys.exit(1)

    enrich(sys.argv[1])
#!/usr/bin/env python3
"""
Merge emilia_curated.csv + fma_curated.csv into train.csv and val.csv.

The combined manifests are written to the REPO's data/ directory (not inside
the storage tree) because they reference paths from both datasets and belong
to the project, not to either dataset individually.

Validation split rules:
  - Emilia val: speaker-disjoint (held-out speakers never appear in train)
  - FMA val:    random track holdout
  - Total val target: ~50 h (25 h speech + 25 h music)

Usage (login node):
    python build_combined_manifest.py \\
        --emilia  /nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv \\
        --fma     /nfs/turbo/umd-hafiz/issf_server_data/fma/manifests/fma_curated.csv \\
        --out-dir ~/aura_watermark/data \\
        --val-h   50 \\
        --seed    42
"""

import argparse
from pathlib import Path

import pandas as pd

VAL_TARGET_H = 50.0


def speaker_disjoint_val(df: pd.DataFrame, val_h: float, seed: int):
    """Select held-out speakers whose total duration ~ val_h hours."""
    spk_hours    = df.groupby("speaker")["duration_s"].sum().sort_values() / 3600
    spk_shuffled = spk_hours.sample(frac=1, random_state=seed)

    val_speakers, cum = set(), 0.0
    for spk, h in spk_shuffled.items():
        if cum >= val_h:
            break
        val_speakers.add(spk)
        cum += h

    val   = df[df["speaker"].isin(val_speakers)]
    train = df[~df["speaker"].isin(val_speakers)]
    return train, val


def random_val(df: pd.DataFrame, val_h: float, seed: int):
    shuffled = df.sample(frac=1, random_state=seed)
    cum      = shuffled["duration_s"].cumsum() / 3600
    return shuffled[cum > val_h], shuffled[cum <= val_h]


def _fill_missing_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col, default in [
        ("rms_db",    -1.0),
        ("clip_frac", -1.0),
        ("genre_top", "speech"),
        ("track_id",  -1),
    ]:
        if col not in df.columns:
            df[col] = default
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Build combined train.csv + val.csv in repo data/ directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--emilia",  required=True,
                    help="emilia/manifests/emilia_curated.csv")
    ap.add_argument("--fma",     required=True,
                    help="fma/manifests/fma_curated.csv")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory (e.g. ~/aura_watermark/data)")
    ap.add_argument("--val-h",   type=float, default=VAL_TARGET_H)
    ap.add_argument("--seed",    type=int,   default=42)
    args = ap.parse_args()

    emilia = _fill_missing_cols(pd.read_csv(args.emilia))
    fma    = _fill_missing_cols(pd.read_csv(args.fma))

    print(f"Emilia: {len(emilia):,} utts  ({emilia['duration_s'].sum()/3600:.1f} h)")
    print(f"FMA:    {len(fma):,} tracks  ({fma['duration_s'].sum()/3600:.1f} h)")

    # ── Speaker-disjoint split for Emilia ─────────────────────────────────
    emilia_train, emilia_val = speaker_disjoint_val(
        emilia, val_h=args.val_h / 2, seed=args.seed
    )
    print(f"\nEmilia train: {len(emilia_train):,}  "
          f"({emilia_train['duration_s'].sum()/3600:.1f} h)  "
          f"{emilia_train['speaker'].nunique():,} speakers")
    print(f"Emilia val:   {len(emilia_val):,}  "
          f"({emilia_val['duration_s'].sum()/3600:.1f} h)  "
          f"{emilia_val['speaker'].nunique():,} held-out speakers")

    # ── Random holdout for FMA ─────────────────────────────────────────────
    fma_train, fma_val = random_val(fma, val_h=args.val_h / 2, seed=args.seed)
    print(f"\nFMA train: {len(fma_train):,}  ({fma_train['duration_s'].sum()/3600:.1f} h)")
    print(f"FMA val:   {len(fma_val):,}  ({fma_val['duration_s'].sum()/3600:.1f} h)")

    KEEP_COLS = ["path", "duration_s", "speaker", "language",
                 "dnsmos", "rms_db", "clip_frac", "genre_top", "dataset"]

    train = (pd.concat([emilia_train, fma_train], ignore_index=True)
               [KEEP_COLS].sample(frac=1, random_state=args.seed))
    val   = (pd.concat([emilia_val,   fma_val],   ignore_index=True)
               [KEEP_COLS].sample(frac=1, random_state=args.seed))

    # ── Write ─────────────────────────────────────────────────────────────
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out   / "val.csv",   index=False)

    total_train = train["duration_s"].sum() / 3600
    total_val   = val["duration_s"].sum()   / 3600

    print(f"\n{'='*55}")
    print(f"FINAL MANIFESTS  ->  {out}/")
    print(f"{'='*55}")
    print(f"train.csv:  {len(train):,} clips   {total_train:.1f} h")
    print(f"  Speech:   {(train['dataset']=='emilia').sum():,}  "
          f"({train[train['dataset']=='emilia']['duration_s'].sum()/3600:.1f} h)")
    print(f"  Music:    {(train['dataset']=='fma').sum():,}  "
          f"({train[train['dataset']=='fma']['duration_s'].sum()/3600:.1f} h)")
    print(f"val.csv:    {len(val):,} clips   {total_val:.1f} h")


if __name__ == "__main__":
    main()

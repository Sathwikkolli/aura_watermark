#!/usr/bin/env python3
"""
Apply DNSMOS-tier + speaker-cap filtering to Emilia English data.

Tier strategy:
  Tier 1  DNSMOS >= 3.90            -> 1,400 h  (high quality anchor)
  Tier 2  3.50 <= DNSMOS < 3.90    ->   700 h  (good quality)
  Tier 3  3.20 <= DNSMOS < 3.50    ->   300 h  (acceptable; adds NMR robustness)

Additional constraints:
  - utterance duration:  3 s – 30 s
  - per-speaker cap:     1.0 h  (min 2,500 distinct speakers)
  - language:            EN only (no multi-lang cap needed)
  - total target:        2,500 h

Usage (on Great Lakes, login node):
    python curate_emilia.py \\
        --raw  /nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_raw.csv \\
        --out  /nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv \\
        --seed 42
"""

import argparse
from pathlib import Path

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

TIERS = [
    # (dnsmos_min, dnsmos_max, target_hours)
    (3.90, 99.0,  1400),
    (3.50,  3.90,  700),
    (3.20,  3.50,  300),
]

DUR_MIN_S      =  3.0
DUR_MAX_S      = 30.0
SPEAKER_CAP_H  =  1.0
TARGET_TOTAL_H = 2500.0


def _cap_speaker(df: pd.DataFrame, cap_h: float, seed: int) -> pd.DataFrame:
    def _apply(grp):
        grp = grp.sample(frac=1, random_state=seed)
        cum  = grp["duration_s"].cumsum() / 3600
        return grp[cum <= cap_h]
    return df.groupby("speaker", group_keys=False).apply(_apply)


def main():
    ap = argparse.ArgumentParser(
        description="DNSMOS-tiered curation of Emilia EN -> emilia_curated.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw",  required=True,
                    help="emilia_raw.csv from download_emilia.py or scan_emilia.py")
    ap.add_argument("--out",  required=True,
                    help="Output path for emilia_curated.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    print(f"Raw input:  {len(df):,} utterances  ({df['duration_s'].sum()/3600:.1f} h)")

    # ── Step 1: Duration filter ───────────────────────────────────────────────
    df = df[(df["duration_s"] >= DUR_MIN_S) & (df["duration_s"] <= DUR_MAX_S)]
    print(f"After duration [{DUR_MIN_S}s, {DUR_MAX_S}s]:  "
          f"{len(df):,}  ({df['duration_s'].sum()/3600:.1f} h)")

    # ── Step 2: Require known DNSMOS ─────────────────────────────────────────
    df = df[df["dnsmos"] > 0].copy()
    print(f"After DNSMOS known:  {len(df):,}  ({df['duration_s'].sum()/3600:.1f} h)")

    # ── Step 3: DNSMOS tier sampling ──────────────────────────────────────────
    print("\nTier sampling:")
    tier_frames = []
    for dnsmos_min, dnsmos_max, target_h in TIERS:
        tier_df = df[
            (df["dnsmos"] >= dnsmos_min) & (df["dnsmos"] < dnsmos_max)
        ].copy()
        tier_df = tier_df.sample(frac=1, random_state=args.seed)
        cum     = tier_df["duration_s"].cumsum() / 3600
        tier_df = tier_df[cum <= target_h]
        tier_frames.append(tier_df)
        actual_h = tier_df["duration_s"].sum() / 3600
        print(f"  Tier [{dnsmos_min:.2f}, {dnsmos_max:.2f}):  "
              f"target={target_h} h  actual={actual_h:.1f} h  "
              f"({len(tier_df):,} utts)")

    selected = pd.concat(tier_frames, ignore_index=True)

    # ── Step 4: Speaker cap (1 h) ─────────────────────────────────────────────
    h_before = selected["duration_s"].sum() / 3600
    selected = _cap_speaker(selected, SPEAKER_CAP_H, args.seed)
    h_after  = selected["duration_s"].sum() / 3600
    n_spk    = selected["speaker"].nunique()
    print(f"\nSpeaker cap ({SPEAKER_CAP_H} h):  "
          f"{h_before:.1f} h -> {h_after:.1f} h  ({n_spk:,} speakers)")

    # ── Step 5: Trim to total target ──────────────────────────────────────────
    selected = selected.sample(frac=1, random_state=args.seed)
    cum      = selected["duration_s"].cumsum() / 3600
    selected = selected[cum <= TARGET_TOTAL_H]

    total_h  = selected["duration_s"].sum() / 3600
    print(f"\nFinal Emilia EN manifest:")
    print(f"  {len(selected):,} utterances  {total_h:.1f} h")
    print(f"  {selected['speaker'].nunique():,} unique speakers")
    print(f"  Language: EN only")

    selected["dataset"] = "emilia"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()

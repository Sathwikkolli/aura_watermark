#!/usr/bin/env python3
"""
Scan all Emilia JSON metadata files and write emilia_raw.csv.

Handles both JSON schema variants in the wild:
  - {"wav": "...", "duration": 8.4, "speaker": "SPK001", "dnsmos": 3.92}
  - {"wav": "...", "duration": 8.4, "speaker": "SPK001",
     "dnsmos": {"OVRL": 3.92, "SIG": 4.1, "BAK": 3.8}}

Usage (on Great Lakes):
    python scan_emilia.py \\
        --emilia-root /nfs/turbo/umd-hafiz/issf_server_data/emilia \\
        --out         /nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_raw.csv \\
        --workers     32
"""

import argparse
import csv
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

SUPPORTED_LANGS = {"EN"}    # English only


# ─────────────────────────────────────────────────────────────────────────────

def parse_json(json_path: Path):
    """
    Parse a single Emilia JSON metadata file.

    Returns:
        (path_str, duration_s, speaker, language, dnsmos)  or  None on error.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        # ── Resolve audio path ──────────────────────────────────────────────
        wav_key = meta.get("wav") or meta.get("audio") or meta.get("path")
        if wav_key is None:
            # Fall back: infer from JSON filename
            audio = json_path.with_suffix(".mp3")
            if not audio.exists():
                audio = json_path.with_suffix(".wav")
                if not audio.exists():
                    return None
        else:
            wav_p = Path(wav_key)
            audio = wav_p if wav_p.is_absolute() else json_path.parent / wav_key

        if not audio.exists():
            return None

        # ── DNSMOS ─────────────────────────────────────────────────────────
        dnsmos_raw = meta.get("dnsmos") or meta.get("DNSMOS")
        if isinstance(dnsmos_raw, dict):
            # Nested: {"OVRL": ..., "SIG": ..., "BAK": ...}
            dnsmos = float(dnsmos_raw.get("OVRL") or dnsmos_raw.get("ovrl") or 0.0)
        elif dnsmos_raw is not None:
            dnsmos = float(dnsmos_raw)
        else:
            dnsmos = -1.0   # missing — will be excluded in curation

        # ── Other fields ────────────────────────────────────────────────────
        duration = float(meta.get("duration") or meta.get("dur") or 0.0)
        speaker  = str(
            meta.get("speaker") or
            meta.get("spk") or
            meta.get("speaker_id") or
            "UNK"
        )
        lang_raw = str(
            meta.get("language") or
            meta.get("lang") or
            "UNK"
        )
        # Normalise to 2-letter code: "en-US" -> "EN"
        language = lang_raw.split("-")[0].upper()[:2]

        if duration <= 0.0:
            return None

        return (str(audio), duration, speaker, language, dnsmos)

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Scan Emilia JSON metadata → emilia_raw.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--emilia-root", required=True,
                    help="Root of the Emilia dataset (contains EN/ ZH/ DE/ …)")
    ap.add_argument("--out",         required=True,
                    help="Output CSV path")
    ap.add_argument("--workers",     type=int, default=32,
                    help="Parallel worker processes")
    args = ap.parse_args()

    root = Path(args.emilia_root)
    if not root.exists():
        raise FileNotFoundError(f"Emilia root not found: {root}")

    # Collect all JSON files across all supported languages
    json_files = []
    for lang in SUPPORTED_LANGS:
        lang_dir = root / lang
        if lang_dir.exists():
            found = list(lang_dir.rglob("*.json"))
            print(f"  {lang}: {len(found):,} JSON files")
            json_files.extend(found)

    print(f"\nTotal JSON files to scan: {len(json_files):,}")

    # Parallel parse
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(parse_json, p): p for p in json_files}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Scanning", unit="file"):
            result = fut.result()
            if result is not None:
                rows.append(result)

    skipped = len(json_files) - len(rows)
    total_h = sum(r[1] for r in rows) / 3600
    print(f"\nParsed:  {len(rows):,} valid utterances  ({total_h:.1f} h)")
    print(f"Skipped: {skipped:,} (missing audio / parse error / zero duration)")

    # Write output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "duration_s", "speaker", "language", "dnsmos"])
        w.writerows(rows)

    print(f"Wrote: {args.out}")

    # Quick DNSMOS distribution summary
    import statistics
    dnsmos_vals = [r[4] for r in rows if r[4] > 0]
    if dnsmos_vals:
        print(f"\nDNSMOS summary (n={len(dnsmos_vals):,}):")
        print(f"  mean={statistics.mean(dnsmos_vals):.3f}  "
              f"stdev={statistics.stdev(dnsmos_vals):.3f}  "
              f"min={min(dnsmos_vals):.3f}  max={max(dnsmos_vals):.3f}")
        thresholds = [3.2, 3.4, 3.5, 3.8, 3.9]
        for t in thresholds:
            pct = 100 * sum(1 for v in dnsmos_vals if v >= t) / len(dnsmos_vals)
            h = sum(r[1] for r in rows if r[4] >= t) / 3600
            print(f"  DNSMOS >= {t}: {pct:.1f}%  ({h:.0f} h)")


if __name__ == "__main__":
    main()

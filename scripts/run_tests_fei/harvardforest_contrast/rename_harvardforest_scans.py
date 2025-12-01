#!/usr/bin/env python3
import re
import json
import argparse
from pathlib import Path

# Natural-ish sort key for mixed strings (A_10 before A_100, etc.)
def _natkey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def parse_name(stem: str):
    """
    Returns (scene_key, scan_int) or (None, None) if unparseable.
    Supported patterns:
      1) NNN-NNN-NNN   -> scene_key='NNN-NNN', scan='NNN'
      2) X_Y           -> scene_key='X',       scan='Y'
      3) X_Y_Z         -> if Z is digits -> scene_key='X_Y', scan='Z'
                          if Z is 'center' -> scene_key='X_Y', scan=0
      4) X_Center      -> scene_key='X', scan=0
    """
    # 1) 037-001-009
    m = re.fullmatch(r'([^-]+)-([^-]+)-(\d+)', stem)
    if m:
        scene_key = f"{m.group(1)}-{m.group(2)}"
        return scene_key, int(m.group(3))

    # Split by underscores
    parts = stem.split('_')
    # Single token like 'A15' not supported (return None)
    if len(parts) == 1:
        return None, None

    last = parts[-1]
    prev = parts[:-1]

    # X_Center or X_Y_Center
    if last.lower() == 'center':
        scene_key = "_".join(prev) if prev else "Center"
        return scene_key, 0

    # X_Y or X_Y_Z (last numeric)
    if last.isdigit():
        scene_key = "_".join(prev) if prev else ""
        if scene_key == "":
            return None, None
        return scene_key, int(last)

    return None, None

def main():
    ap = argparse.ArgumentParser(description="Standardize HarvardForest scan filenames and write a rename map.")
    ap.add_argument("folder", type=str, help="Path to folder with .las/.laz files")
    ap.add_argument("--apply", action="store_true", help="Actually rename files. (Default: dry-run)")
    ap.add_argument("--map", type=str, default="rename_map.json", help="Output JSON map filename")
    ap.add_argument("--scene-pad", type=int, default=2, help="Digits for scene index (SXX). Default: 2 -> S01")
    ap.add_argument("--scan-pad", type=int, default=3, help="Digits for scan number (YYY). Default: 3 -> 001")
    ap.add_argument("--sep", type=str, default="_", help="Separator between scene and scan in new name. Default: '_'")
    ap.add_argument("--ext", type=str, default=None, help="Force new extension (e.g., las). Default: keep original.")
    args = ap.parse_args()

    folder = Path(args.folder)
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".las", ".laz")])

    # Collect parsed entries
    entries = []
    unparsed = []
    for p in files:
        stem = p.stem
        scene_key, scan = parse_name(stem)
        if scene_key is None or scan is None:
            unparsed.append(p.name)
        else:
            entries.append((p, scene_key, scan))

    if not entries and unparsed:
        print("No files parsed. Unparsed examples:", unparsed[:10])
        return

    # Assign scene indices based on sorted unique scene keys
    scene_keys = sorted({e[1] for e in entries}, key=_natkey)
    scene_to_idx = {sk: i+1 for i, sk in enumerate(scene_keys)}

    # Build rename map
    rename_map = {"files": [], "scenes": {}}
    for sk, idx in scene_to_idx.items():
        rename_map["scenes"][sk] = f"S{idx:0{args.scene_pad}d}"

    for p, sk, scan in entries:
        s_tag = f"S{scene_to_idx[sk]:0{args.scene_pad}d}"
        scan_tag = f"{scan:0{args.scan_pad}d}"
        new_ext = (args.ext.lower() if args.ext else p.suffix.lower().lstrip("."))
        new_name = f"{s_tag}{args.sep}{scan_tag}.{new_ext}"
        rename_map["files"].append({"old": p.name, "new": new_name, "scene_key": sk, "scan": scan})

    # Report
    print(f"Parsed {len(entries)} files; {len(unparsed)} unparsed.")
    if unparsed:
        print("Unparsed filenames (inspect/fix manually):")
        for u in unparsed:
            print("  -", u)

    # Show a small preview
    print("\nPreview (first 10):")
    for item in rename_map["files"][:10]:
        print(f"  {item['old']}  ->  {item['new']}  (scene='{item['scene_key']}', scan={item['scan']})")

    # Write JSON
    out_path = folder / args.map
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rename_map, f, indent=2, ensure_ascii=False)
    print(f"\nWrote map: {out_path}")

    # Apply renames if requested
    if args.apply:
        # Check collisions before touching anything
        new_names = [item["new"] for item in rename_map["files"]]
        if len(new_names) != len(set(new_names)):
            print("ERROR: New name collision detected. Aborting. Inspect the JSON.")
            return

        # Perform renames
        for item in rename_map["files"]:
            src = folder / item["old"]
            dst = folder / item["new"]
            if dst.exists():
                print(f"ERROR: Target exists, skipping: {dst.name}")
                continue
            src.rename(dst)
        print("Renames applied.")

if __name__ == "__main__":
    main()

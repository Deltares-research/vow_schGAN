"""Extract the RD coordinates of every BRO XML file in a folder to a CSV (columns: name, X, Y).

Works for borehole (BHR) and CPT XMLs: `name` is the BRO id (falls back to the
file name), X/Y are the RD New (EPSG:28992) coordinates of the delivered
location. Files without coordinates or that cannot be parsed are skipped and
reported at the end.

Run with:
    .venv/Scripts/python.exe src/extract_bro_xml_coords.py --input-dir <folder with *.xml> --output-csv <file.csv>
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # add src/ to sys.path

from modules.borehole_ic.xml_readers import read_cpt_xml_metadata  # noqa: E402  (reads id + RD coords of any BRO XML)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing BRO XML files.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path (columns: name, X, Y).")
    args = parser.parse_args()

    xml_files = sorted(args.input_dir.glob("*.xml"))
    if not xml_files:
        sys.exit(f"No .xml files found in {args.input_dir}")

    rows, skipped = [], []
    for path in xml_files:
        try:
            meta = read_cpt_xml_metadata(path)
        except Exception as exc:
            skipped.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if meta["x"] is None or meta["y"] is None:
            skipped.append(f"{path.name}: no RD coordinates found")
            continue
        rows.append([meta["bro_id"] or path.stem, meta["x"], meta["y"]])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "X", "Y"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} of {len(xml_files)} files to {args.output_csv}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for line in skipped:
            print("  " + line)


if __name__ == "__main__":
    main()

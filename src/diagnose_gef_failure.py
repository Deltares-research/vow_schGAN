from pathlib import Path
import re
import csv

RE_COLUMNINFO = re.compile(
    r"^#\s*COLUM?NINFO\s*=\s*(\d+)\s*,", re.IGNORECASE
)  # matches "#COLUMNINFO=1, ..."
RE_EOH = re.compile(r"^#\s*EOH\s*$", re.IGNORECASE)


def _read_text_smart(path: Path):
    """Try UTF-8 first; if it fails, fall back to cp1252 and mark encoding."""
    try:
        txt = path.read_text(encoding="utf-8")
        return txt, "utf-8", False
    except UnicodeDecodeError:
        txt = path.read_text(encoding="cp1252", errors="strict")
        return txt, "cp1252", True


def analyze_gef_file(path: Path):
    """
    Return a dict with keys:
      file, ok, reasons (list), notes (list), expected_cols, data_cols_mode, enc, fallback_used
    """
    info = {
        "file": str(path),
        "ok": False,
        "reasons": [],
        "notes": [],
        "expected_cols": None,
        "data_cols_mode": None,
        "enc": None,
        "fallback_used": False,
    }

    try:
        text, enc, fallback = _read_text_smart(path)
        info["enc"] = enc
        info["fallback_used"] = fallback
        if fallback:
            info["notes"].append("encoding_fallback_cp1252")

        lines = text.splitlines()
        if not lines:
            info["reasons"].append("empty_file")
            return info

        # Locate EOH
        eoh_idx = None
        for i, line in enumerate(lines):
            if RE_EOH.match(line.strip()):
                eoh_idx = i
                break
        if eoh_idx is None:
            info["reasons"].append("missing_eoh")
            # we can still attempt limited header checks below

        header_lines = lines[: eoh_idx if eoh_idx is not None else min(len(lines), 200)]
        data_lines = lines[eoh_idx + 1 :] if eoh_idx is not None else []

        # Extract COLUMNINFO indices
        col_indices = []
        for hl in header_lines:
            m = RE_COLUMNINFO.match(hl.strip())
            if m:
                try:
                    col_indices.append(int(m.group(1)))
                except ValueError:
                    pass

        if not col_indices:
            info["reasons"].append("missing_columninfo")
        else:
            # Expected columns: either the count or the max index (GEF usually numbers columns starting at 1)
            expected_cols = max(col_indices)
            info["expected_cols"] = expected_cols

        # Basic required header tags (optional check; just note if absent)
        required_tags = ["#FILEDATE", "#PROJECTID", "#TESTID"]
        missing_tags = [
            t for t in required_tags if not any(t in h for h in header_lines)
        ]
        if missing_tags:
            info["notes"].append(f"missing_meta:{','.join(missing_tags)}")

        # If we have no data lines after EOH
        if eoh_idx is not None and (
            not data_lines or all(not d.strip() for d in data_lines)
        ):
            info["reasons"].append("empty_data_block")
            return info

        # Analyze data block
        row_col_counts = []
        non_numeric_rows = 0
        comma_as_decimal_seen = False

        # Take first 200 non-empty data lines to sample
        sample_count = 0
        for raw in data_lines:
            if not raw.strip():
                continue
            line = raw.strip()

            # detect comma decimals (e.g., "1,23  4,56")
            if re.search(r"\d,\d", line):
                comma_as_decimal_seen = True

            # normalize: split on whitespace (GEF usually space-delimited)
            parts = line.replace(",", " ").split()
            if not parts:
                continue

            # Check numeric
            numeric = True
            for p in parts:
                try:
                    float(p)
                except ValueError:
                    numeric = False
                    break
            if not numeric:
                non_numeric_rows += 1

            row_col_counts.append(len(parts))
            sample_count += 1
            if sample_count >= 200:
                break

        if comma_as_decimal_seen:
            info["notes"].append("comma_decimal_candidates")

        if not row_col_counts:
            info["reasons"].append("no_parsable_data_rows")
            return info

        # Most common column count in sampled rows
        from collections import Counter

        cnt = Counter(row_col_counts)
        data_cols_mode, freq = cnt.most_common(1)[0]
        info["data_cols_mode"] = data_cols_mode

        # Column consistency within data
        if len(cnt) > 1:
            info["reasons"].append("inconsistent_row_lengths")

        # Match against expected column count (if we found any COLUMNINFO)
        if (
            info["expected_cols"] is not None
            and data_cols_mode != info["expected_cols"]
        ):
            info["reasons"].append("data_cols_vs_header_mismatch")

        # Non-numeric rows presence
        if non_numeric_rows > 0:
            info["reasons"].append("non_numeric_values_in_data")

        # If we have no hard failures, call it OK
        if not info["reasons"]:
            info["ok"] = True

        return info

    except UnicodeDecodeError:
        info["reasons"].append("encoding_error_utf8_and_cp1252")
        return info
    except Exception as e:
        info["reasons"].append(f"unexpected_error:{type(e).__name__}")
        info["notes"].append(str(e))
        return info


def analyze_folder(folder: Path, extensions=(".gef", ".GEF", ".gef.txt", ".GEF.TXT")):
    files = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix in {".gef", ".GEF", ".txt", ".TXT"}
        or any(str(p).endswith(ext) for ext in extensions)
    ]
    results = [analyze_gef_file(p) for p in files]

    # Summary counts
    from collections import Counter, defaultdict

    reason_counter = Counter()
    note_counter = Counter()
    ok_count = 0
    fail_count = 0

    # Multi-reason accumulation per file
    for r in results:
        if r["ok"]:
            ok_count += 1
        else:
            fail_count += 1
        for reason in r["reasons"]:
            reason_counter[reason] += 1
        for note in r["notes"]:
            note_counter[note] += 1

    # Write CSV report
    out_csv = folder / "gef_diagnostic_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "ok",
                "reasons",
                "notes",
                "expected_cols",
                "data_cols_mode",
                "encoding",
                "fallback_used",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["file"],
                    "OK" if r["ok"] else "FAIL",
                    "|".join(r["reasons"]) if r["reasons"] else "",
                    "|".join(r["notes"]) if r["notes"] else "",
                    r["expected_cols"] if r["expected_cols"] is not None else "",
                    r["data_cols_mode"] if r["data_cols_mode"] is not None else "",
                    r["enc"] or "",
                    "yes" if r["fallback_used"] else "no",
                ]
            )

    # Print concise summary
    print(f"Scanned {len(results)} files → OK: {ok_count}, FAIL: {fail_count}")
    if reason_counter:
        print("\nTop failure reasons:")
        for reason, n in reason_counter.most_common():
            print(f"  - {reason}: {n}")
    if note_counter:
        print("\nNotes seen:")
        for note, n in note_counter.most_common():
            print(f"  - {note}: {n}")

    print(f"\nCSV report written to: {out_csv}")


if __name__ == "__main__":
    # CHANGE THIS to your 'failure' folder
    folder = Path(r"C:\ark\data\sonderingen\SON\failure")
    analyze_folder(folder)

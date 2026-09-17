#!/usr/bin/env python3
"""ICCIT 2026 venue gate. Exits non-zero on any violation.

Usage: python3 scripts/check_venue_gate.py <paper_dir>
where <paper_dir> holds main.tex and the built main.pdf.

The rules are the live ones, read from the conference submission page on
2026-08-28, not from recollection:

  "Paper submissions should be limited to a maximum of six (6) pages, in the
   IEEE 2-column format, including figures and references."
  "The authors should not include their names, affiliations, postal addresses,
   or email addresses in the initial manuscript. If it is not maintained, the
   manuscript will be immediately rejected."

The glyph audit runs over the rendered PDF as well as the sources, because a
banned glyph can be baked into a generated figure and survive a clean sweep of
every .tex file.
"""
import re
import subprocess
import sys
from pathlib import Path

BANNED = {"§": "section-mark", "¶": "pilcrow",
          "†": "dagger", "‡": "double-dagger"}
MAX_PAGES = 6


def fail(msg):
    print("FAIL:", msg)
    return 1


def main(paper_dir):
    p = Path(paper_dir)
    pdf, log = p / "main.pdf", p / "main.log"
    bad = 0

    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    size = re.search(r"^Page size:\s+(.+)$", info, re.M).group(1)
    print(f"pages={pages} size={size}")
    if pages > MAX_PAGES:
        bad |= fail(f"{pages} pages, limit is {MAX_PAGES} including figures and references")
    if "595" not in size:
        bad |= fail(f"page size {size} is not A4")

    logtext = log.read_text(errors="replace") if log.exists() else ""
    undef = len(re.findall(r"undefined (?:reference|citation)", logtext, re.I))
    over = len(re.findall(r"^Overfull \\hbox", logtext, re.M))
    print(f"undefined={undef} overfull_hbox={over}")
    if undef:
        bad |= fail(f"{undef} undefined references or citations")
    if over:
        bad |= fail(f"{over} overfull hboxes")

    for f in list(p.glob("*.tex")) + list(p.glob("sections/*.tex")) + list(p.glob("bib/*.bib")):
        t = f.read_text(errors="replace")
        for ch, name in BANNED.items():
            if ch in t:
                bad |= fail(f"{name} in {f}")
        if re.search(r"\\d?dagger", t):
            bad |= fail(f"LaTeX dagger macro in {f}")

    txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    for ch, name in BANNED.items():
        if ch in txt:
            bad |= fail(f"{name} in the RENDERED pdf (check generated figures, not only .tex)")

    if "\\blindtrue" not in (p / "main.tex").read_text(errors="replace"):
        bad |= fail("blind toggle is not \\blindtrue; submission would be desk-rejected")

    if not bad:
        print("PASS: all venue gates green")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

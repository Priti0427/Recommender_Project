"""Convert FINAL_REPORT.md to a formatted .docx (single-spaced, 12pt).

Handles: #/##/### headings, **bold**, *italic*, `code`, bullet lists,
blockquotes, horizontal rules, and GitHub-style tables.
"""
from __future__ import annotations

import os
import re

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "FINAL_REPORT.md")
OUT = os.path.join(HERE, "FINAL_REPORT.docx")

HEADING_COLOR = RGBColor(0x1F, 0x4E, 0x79)

# Inline markdown: **bold**, *italic*, `code`
_TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)")


def add_runs(paragraph, text):
    """Add text to a paragraph, parsing inline bold/italic/code."""
    for part in _TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = paragraph.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(10.5)
        elif part.startswith("*") and part.endswith("*"):
            r = paragraph.add_run(part[1:-1])
            r.italic = True
        else:
            paragraph.add_run(part)


def split_table_row(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    doc = Document()
    # Base style: 12pt, single spacing
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(12)
    pf = normal.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.0

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Blank
        if not stripped:
            i += 1
            continue

        # Horizontal rule -> skip (acts as separator)
        if re.fullmatch(r"-{3,}", stripped):
            i += 1
            continue

        # Image: ![alt](path)
        img = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if img:
            path = img.group(2)
            if not os.path.isabs(path):
                path = os.path.join(HERE, path)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if os.path.exists(path):
                p.add_run().add_picture(path, width=Inches(6.2))
            else:
                add_runs(p, f"[missing image: {img.group(2)}]")
            i += 1
            continue

        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            h = doc.add_heading(level=min(level, 4))
            add_runs(h, m.group(2))
            for r in h.runs:
                r.font.color.rgb = HEADING_COLOR
            i += 1
            continue

        # Table: a line with | and the next line is a separator row
        if stripped.startswith("|") and i + 1 < n and re.match(
            r"^\|?[\s:\-|]+\|?$", lines[i + 1].strip()
        ) and "-" in lines[i + 1]:
            header = split_table_row(stripped)
            i += 2  # skip header + separator
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_table_row(lines[i].strip()))
                i += 1
            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Light Grid Accent 1"
            hdr_cells = table.rows[0].cells
            for c, txt in enumerate(header):
                hdr_cells[c].text = ""
                add_runs(hdr_cells[c].paragraphs[0], txt)
                for r in hdr_cells[c].paragraphs[0].runs:
                    r.bold = True
            for row in rows:
                cells = table.add_row().cells
                for c, txt in enumerate(row):
                    if c < len(cells):
                        cells[c].text = ""
                        add_runs(cells[c].paragraphs[0], txt)
            continue

        # Blockquote
        if stripped.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(24)
            add_runs(p, stripped.lstrip("> ").strip())
            for r in p.runs:
                r.italic = True
            i += 1
            continue

        # Bullet list
        if re.match(r"^[-*]\s+", stripped):
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, re.sub(r"^[-*]\s+", "", stripped))
            i += 1
            continue

        # Numbered list
        if re.match(r"^\d+\.\s+", stripped):
            p = doc.add_paragraph(style="List Number")
            add_runs(p, re.sub(r"^\d+\.\s+", "", stripped))
            i += 1
            continue

        # Plain paragraph (center figure captions)
        p = doc.add_paragraph()
        if re.match(r"^\*Figure\b", stripped):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(p, stripped)
        i += 1

    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

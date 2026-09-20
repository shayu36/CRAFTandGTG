#!/usr/bin/env python3
"""Build a small, dependency-free DOCX from the current modeling pipeline Markdown."""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "THREE_LAYER_MODELING_PIPELINE.md"
OUTPUT = ROOT / "docs" / "THREE_LAYER_MODELING_PIPELINE.docx"


def run_text(text: str, *, bold: bool = False, font: str = "等线", size: int = 22) -> str:
    properties = f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}"/><w:sz w:val="{size}"/>'
    if bold:
        properties += "<w:b/>"
    return f"<w:r><w:rPr>{properties}</w:rPr><w:t xml:space=\"preserve\">{html.escape(text)}</w:t></w:r>"


def paragraph(text: str = "", *, style: str = "Normal", code: bool = False) -> str:
    if code:
        return f'<w:p><w:pPr><w:pStyle w:val="Code"/></w:pPr>{run_text(text, font="Consolas", size=18)}</w:p>'
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{run_text(text)}</w:p>'


def convert(markdown: str) -> str:
    output: list[str] = []
    in_code = False
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            output.append(paragraph(line, code=True))
            continue
        if not line:
            output.append(paragraph())
            continue
        match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if match:
            level = len(match.group(1))
            output.append(paragraph(re.sub(r"`([^`]*)`", r"\1", match.group(2)), style=f"Heading{level}"))
            continue
        if line.startswith("- "):
            text = re.sub(r"`([^`]*)`", r"\1", line[2:])
            output.append(f'<w:p><w:pPr><w:pStyle w:val="ListBullet"/></w:pPr>{run_text(text)}</w:p>')
            continue
        text = re.sub(r"`([^`]*)`", r"\1", line)
        output.append(paragraph(text))
    return "".join(output)


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="等线" w:hAnsi="等线" w:eastAsia="等线"/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:rPr><w:b/><w:sz w:val="27"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr></w:style>
</w:styles>"""


def main() -> None:
    body = convert(SOURCE.read_text(encoding="utf-8"))
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body>
</w:document>'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", RELS)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", STYLES)
        archive.writestr("word/_rels/document.xml.rels", DOC_RELS)
    print(OUTPUT)


if __name__ == "__main__":
    main()

"""README.md -> styled print HTML (for headless Chrome to turn into docs/writeup.pdf).

    .venv/bin/python docs/build_writeup.py <out.html>
"""
import sys
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parent.parent

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font: 10.5pt/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1f2328; }
h1 { font-size: 22pt; margin: 4pt 0 6pt; text-align: center; }
h2 { font-size: 14.5pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #d0d7de;
     break-after: avoid; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
blockquote { margin: 6pt 0; padding: 6pt 10pt; background: #fff4e5; border-left: 4px solid #e8590c;
             color: #5c2b00; }
blockquote p { margin: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
tr:nth-child(even) td { background: #fafbfc; }
code { font: 8.8pt "SF Mono", Menlo, monospace; background: #f3f4f6; padding: 0 2pt; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px; padding: 7pt 9pt;
      font-size: 8pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8pt; }
img { max-width: 100%; }
p[align=center] { text-align: center; margin: 6pt 0; break-inside: avoid; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 14pt 0; }
a { color: #0b5cad; text-decoration: none; }
ol li { margin-bottom: 3pt; font-size: 9pt; }
sub { color: #57606a; font-size: 8.5pt; }
"""


def main():
    md = (ROOT / "README.md").read_text()
    html = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)(md)
    out = Path(sys.argv[1])
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<base href="{ROOT.as_uri()}/"><title>Spot Check: skin lesion triage that explains itself</title>
<style>{CSS}</style></head><body>{html}</body></html>""")
    print("wrote", out)


if __name__ == "__main__":
    main()

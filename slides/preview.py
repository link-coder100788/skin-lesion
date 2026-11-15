"""Approximate PNG previews of slides/pitch.pptx without PowerPoint or LibreOffice.

    .venv/bin/python slides/preview.py [out_dir]

Draws every picture, filled shape, table and text box at its real position with Pillow, wraps
text with a system font at the run's point size, and prints a warning for any text that
overflows its box or the slide. Good enough to catch layout mistakes; not a faithful renderer.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu

ROOT = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "preview"
DPI = 60
FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
         "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
BOLD = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"] + FONTS


def px(emu) -> int:
    return int(Emu(emu).inches * DPI)


def font(size_pt: float, bold: bool):
    for f in (BOLD if bold else FONTS):
        if Path(f).exists():
            return ImageFont.truetype(f, max(6, int(size_pt * DPI / 72)))
    return ImageFont.load_default()


def wrap(draw, text, f, width):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word).strip()
            if draw.textlength(trial, font=f) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_text_frame(draw, tf, x, y, w, h, label, warnings):
    cy = y + 4
    for p in tf.paragraphs:
        runs = p.runs
        text = "".join(r.text for r in runs)
        size = next((r.font.size.pt for r in runs if r.font.size), 18)
        bold = any(r.font.bold for r in runs)
        color = next((tuple(r.font.color.rgb) for r in runs if r.font.color and r.font.color.type), (31, 35, 40))
        f = font(size, bold)
        lh = int(size * DPI / 72 * 1.2)
        for line in wrap(draw, text, f, max(10, w - 8)):
            draw.text((x + 4, cy), line, font=f, fill=color)
            cy += lh
        cy += int((p.space_after.pt if p.space_after else 0) * DPI / 72)
    if cy > y + h + 6:
        warnings.append(f"{label}: text overflows its box by {cy - y - h}px")
    return cy


def main():
    prs = Presentation(ROOT / "pitch.pptx")
    W, H = px(prs.slide_width), px(prs.slide_height)
    OUT.mkdir(parents=True, exist_ok=True)
    for n, slide in enumerate(prs.slides, 1):
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        warnings = []
        for sh in slide.shapes:
            x, y, w, h = px(sh.left), px(sh.top), px(sh.width), px(sh.height)
            label = f"slide {n} '{(sh.text_frame.text[:30] if sh.has_text_frame else sh.shape_type)}'"
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pic = Image.open(sh.image._blob if False else __import__("io").BytesIO(sh.image.blob)).convert("RGB")
                img.paste(pic.resize((max(1, w), max(1, h))), (x, y))
                if x + w > W or y + h > H:
                    warnings.append(f"{label}: picture runs off the slide")
                continue
            if sh.has_table if hasattr(sh, "has_table") else False:
                tbl = sh.table
                cy = y
                for r in tbl.rows:
                    cx = x
                    rh = px(r.height)
                    for c, cell in zip(tbl.columns, r.cells):
                        cw = px(c.width)
                        d.rectangle([cx, cy, cx + cw, cy + rh], outline=(220, 220, 220))
                        draw_text_frame(d, cell.text_frame, cx, cy, cw, rh, label + " cell", warnings)
                        cx += cw
                    cy += rh
                if cx > W:
                    warnings.append(f"{label}: table wider than slide")
                continue
            if sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
                try:
                    fill = tuple(sh.fill.fore_color.rgb) if sh.fill.type == 1 else None
                except (AttributeError, TypeError):
                    fill = None
                d.rectangle([x, y, x + w, y + h], fill=fill, outline=(200, 200, 200) if not fill else None)
            if sh.has_text_frame and sh.text_frame.text.strip():
                bottom = draw_text_frame(d, sh.text_frame, x, y, w, h, label, warnings)
                if bottom > H or x + w > W + 2:
                    warnings.append(f"{label}: text runs off the slide")
        img.save(OUT / f"slide{n:02d}.png")
        for wmsg in warnings:
            print("WARN", wmsg)
    print(f"wrote {len(prs.slides)} previews to {OUT}")


if __name__ == "__main__":
    main()

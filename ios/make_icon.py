"""Generate the app icon: a dermoscope lens over a skin spot with a SHAP-style red/blue glow.

    .venv/bin/python ios/make_icon.py

Writes ios/Veritas/Assets.xcassets/AppIcon.appiconset/icon.png (1024 x 1024, opaque; iOS
applies the rounded mask itself).
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

S = 1024
SS = 2                      # supersampling factor for smooth edges
N = S * SS
OUT = Path(__file__).resolve().parent / "Veritas/Assets.xcassets/AppIcon.appiconset/icon.png"
rng = np.random.default_rng(7)

y, x = np.mgrid[0:N, 0:N].astype(np.float32) / N   # 0..1


def smooth(edge, d, width):
    """1 inside (d < edge), 0 outside, with a soft transition `width` wide."""
    return np.clip((edge - d) / width + 0.5, 0, 1)


def blend(base, color, alpha):
    return base * (1 - alpha[..., None]) + np.asarray(color, np.float32) * alpha[..., None]


# background: deep indigo -> teal diagonal gradient
t = np.clip((x + y) / 2, 0, 1)[..., None]
img = (1 - t) * np.array([0.10, 0.11, 0.30]) + t * np.array([0.03, 0.42, 0.50])
img = img.astype(np.float32)

# lens: centre and radius
cx, cy, r = 0.44, 0.44, 0.31
d = np.hypot(x - cx, y - cy)

# skin seen through the lens, slightly lit from the top left
light = np.clip(1.08 - 0.5 * np.hypot(x - (cx - 0.12), y - (cy - 0.14)), 0.75, 1.1)[..., None]
skin = np.array([0.93, 0.74, 0.64], np.float32) * light

# the spot: an irregular blob (radius modulated by a few low-frequency harmonics)
ang = np.arctan2(y - cy, x - cx)
wobble = (1 + 0.10 * np.sin(3 * ang + 0.6) + 0.07 * np.sin(5 * ang + 2.1) + 0.05 * np.sin(8 * ang + 0.3))
spot_r = 0.135 * wobble
ds = np.hypot(x - cx - 0.01, y - cy + 0.005)
spot = smooth(spot_r, ds, 0.012)
core = smooth(spot_r * 0.55, np.hypot(x - cx + 0.025, y - cy - 0.02), 0.03)
skin = blend(skin, (0.45, 0.25, 0.17), spot * 0.95)
skin = blend(skin, (0.24, 0.12, 0.10), core * 0.85)

# SHAP-style glow: red where the spot pushes toward "see a doctor", a little blue on the skin
red = np.exp(-(((x - cx - 0.04) ** 2) / 0.006 + ((y - cy + 0.03) ** 2) / 0.004))
red2 = np.exp(-(((x - cx + 0.06) ** 2) / 0.003 + ((y - cy - 0.05) ** 2) / 0.004))
blue = np.exp(-(((x - cx + 0.16) ** 2) / 0.005 + ((y - cy + 0.12) ** 2) / 0.003))
blue2 = np.exp(-(((x - cx - 0.17) ** 2) / 0.003 + ((y - cy - 0.13) ** 2) / 0.005))
skin = blend(skin, (0.95, 0.17, 0.20), np.clip(0.75 * red + 0.6 * red2, 0, 0.8))
skin = blend(skin, (0.20, 0.40, 0.98), np.clip(0.55 * (blue + blue2), 0, 0.55))

inside = smooth(r, d, 0.003)
img = img * (1 - inside[..., None]) + skin * inside[..., None]

# soft shadow under the lens ring + handle, then the ring and handle themselves
ring_w = 0.045
hx0, hy0 = cx + (r + ring_w * 0.5) * 0.7071, cy + (r + ring_w * 0.5) * 0.7071
hx1, hy1 = 0.90, 0.90
def seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    tt = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(px - ax - tt * vx, py - ay - tt * vy)

ring_d = np.abs(d - r - ring_w / 2)
handle_d = seg_dist(x, y, hx0, hy0, hx1, hy1)
shadow = np.maximum(smooth(ring_w / 2 + 0.02, np.abs(np.hypot(x - cx - 0.012, y - cy - 0.018) - r - ring_w / 2), 0.04),
                    smooth(0.055, seg_dist(x - 0.012, y - 0.018, hx0, hy0, hx1, hy1), 0.05))
img = blend(img, (0.0, 0.02, 0.08), shadow * 0.45)

metal_t = np.clip(0.5 + 0.9 * ((x - cx) * -0.7 + (y - cy) * -0.7), 0, 1)[..., None]
metal = (1 - metal_t) * np.array([0.62, 0.68, 0.76]) + metal_t * np.array([0.97, 0.98, 1.0])
ring = smooth(ring_w / 2, ring_d, 0.004)
handle = smooth(0.042, handle_d, 0.004)
img = img * (1 - ring[..., None]) + metal * ring[..., None]
img = blend(img, (0.96, 0.43, 0.25), handle)                         # warm orange handle
img = blend(img, (1.0, 0.62, 0.45), smooth(0.016, seg_dist(x + 0.012, y - 0.012, hx0 + 0.02, hy0 + 0.02, hx1 - 0.03, hy1 - 0.03), 0.006) * handle * 0.6)

# glass highlight: a soft arc in the top-left of the lens
hl = smooth(0.012, np.abs(np.hypot(x - cx, y - cy) - r * 0.82), 0.01)
hl *= np.clip(-np.sin(ang + 0.785) * 1.4 - 0.25, 0, 1) * inside
img = blend(img, (1, 1, 1), hl * 0.55)

# a whisper of grain so the gradients don't band
img += rng.normal(0, 0.006, img.shape).astype(np.float32)

out = Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).resize((S, S), Image.LANCZOS)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.save(OUT)
print("wrote", OUT)

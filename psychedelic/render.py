"""Turn a festival photo into a beat-synced psychedelic reel.

Layers (all driven by the 140 BPM grid from audio.py):
  * slow pan/zoom with a kick-drum pump and chromatic aberration
  * hue cycling + rippling canopy warp (stronger at the top, crowd stays natural)
  * sweeping searchlights that project a music-note emblem onto the canopy
  * laser fans from the real laser heads in the photo
  * glowing musical notes fired out of the DJ booth
  * rotating sacred-geometry mandala, beat shockwaves, strobes, mirror flashes

usage: python3 render.py source.jpg track.wav out.mp4
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from audio import BEAT, TOTAL_BEATS

W, H = 1080, 1920
HW, HH = W // 2, H // 2
FPS = 30
DUR = TOTAL_BEATS * BEAT + 0.6
NFRAMES = int(DUR * FPS)

# Positions in the photo as fractions of its size.
DJ = (0.487, 0.80)
EMITTERS = [(0.072, 0.730), (0.230, 0.727), (0.487, 0.76), (0.673, 0.722),
            (0.851, 0.727), (0.932, 0.756)]
MANDALA_C = (0.46, 0.33)

SRC = None
SPRITES = None


# ----------------------------------------------------------------- helpers
def init(src_path):
    global SRC, SPRITES
    im = Image.open(src_path).convert("RGB")
    SRC = im.resize((2000, 2000), Image.LANCZOS)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 150)
    SPRITES = []
    for ch in "♪♫♬♩":
        im = Image.new("L", (200, 200), 0)
        ImageDraw.Draw(im).text((100, 100), ch, font=font, fill=255, anchor="mm")
        SPRITES.append(im.crop(im.getbbox()))


def beat_at(t):
    return t / BEAT


def env_since_beat(t, decay=6.0, div=1.0):
    b = beat_at(t) * div
    frac = (b - math.floor(b)) / div * BEAT
    return math.exp(-frac * decay)


def section(b):
    if b < 8:
        return "intro"
    if b < 16:
        return "groove"
    if b < 32:
        return "drop"
    return "outro"


def hsv(h, s=1.0, v=1.0):
    h = (h % 1.0) * 6
    i = int(h)
    f = h - i
    p, q, r = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, r, p), (q, v, p), (p, v, r), (p, q, v), (r, p, v), (v, p, q)][i % 6]


def hue_rotate(arr, deg):
    """Rotate hue of float RGB array (H,W,3) via the standard luminance-preserving matrix."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.array([
        [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
        [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
        [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
    ], dtype=np.float32)
    return arr @ m.T


def camera(t, b):
    """Return scale s and source center (cx, cy) in 2000px source coords."""
    kick = env_since_beat(t, 9) if 8 <= b < 32 else 0.0
    z = 1.0 + 0.06 * (t / DUR) + 0.03 * kick
    s = (H / 2000) * z
    cx = 1000 + 300 * math.sin(2 * math.pi * t / DUR * 1.0 - 0.6)
    half_w = (W / 2) / s
    cx = min(max(cx, half_w), 2000 - half_w)
    half_h = (H / 2) / s
    cy = min(max(1040, half_h), 2000 - half_h)
    # shake on drop kicks
    if 16 <= b < 32:
        cx += 6 * kick * math.sin(t * 91)
        cy += 6 * kick * math.cos(t * 77)
    return s, cx, cy


def to_out(p, cam):
    s, cx, cy = cam
    return ((p[0] * 2000 - cx) * s + W / 2, (p[1] * 2000 - cy) * s + H / 2)


# ----------------------------------------------------------------- layers
def draw_searchlights(t, b, cam):
    """Return (glow layer float HxWx3, disc mask float HxW) at full res."""
    inten = 1.0
    if b < 1:
        inten = b
    if b >= 33:
        inten = max(0, 1 - (b - 33) / 2)
    cone = Image.new("L", (HW, HH), 0)
    disc = Image.new("L", (HW, HH), 0)
    gobo = Image.new("L", (HW, HH), 0)
    dc, dd, dg = ImageDraw.Draw(cone), ImageDraw.Draw(disc), ImageDraw.Draw(gobo)
    speed = 0.35 if b < 16 else 0.6
    for k, base in enumerate([(-40, HH + 60), (HW + 40, HH + 60)]):
        ph = t * speed * 2 * math.pi + k * math.pi
        tx = HW * (0.5 + 0.38 * math.sin(ph))
        ty = HH * (0.20 + 0.10 * math.sin(ph * 0.7 + 1.3))
        r = 72 + 10 * math.sin(ph * 1.3)
        ang = math.atan2(ty - base[1], tx - base[0])
        nx, ny = -math.sin(ang), math.cos(ang)
        dc.polygon([(base[0] - nx * 6, base[1] - ny * 6), (base[0] + nx * 6, base[1] + ny * 6),
                    (tx + nx * r, ty + ny * r * 0.8), (tx - nx * r, ty - ny * r * 0.8)], fill=60)
        dd.ellipse([tx - r, ty - r * 0.8, tx + r, ty + r * 0.8], fill=255)
        # original emblem: a big eighth note silhouette in the beam
        sp = SPRITES[1].resize((int(r * 1.0), int(r * 1.0 * SPRITES[1].height / SPRITES[1].width)))
        sp = sp.rotate(math.degrees(math.sin(ph)) * 12, expand=True)
        gobo.paste(255, (int(tx - sp.width / 2), int(ty - sp.height / 2)), sp)
    cone = cone.filter(ImageFilter.GaussianBlur(10))
    disc = disc.filter(ImageFilter.GaussianBlur(3))
    gobo = gobo.filter(ImageFilter.GaussianBlur(1.5))
    c = np.asarray(cone.resize((W, H), Image.BILINEAR), np.float32) / 255
    d = np.asarray(disc.resize((W, H), Image.BILINEAR), np.float32) / 255
    g = np.asarray(gobo.resize((W, H), Image.BILINEAR), np.float32) / 255
    d = d * (1 - 0.85 * g)
    return c * inten, d * inten


def draw_lasers(t, b, cam):
    sec = section(b)
    if sec == "intro" and b < 4:
        return None
    layer = Image.new("RGB", (HW, HH), 0)
    dr = ImageDraw.Draw(layer)
    if sec == "intro":
        emitters, fan, spread, amp, flick = [2], 1, 0, 25, True
    elif sec == "groove":
        emitters, fan, spread, amp, flick = [0, 2, 5], 5, 70, 20, False
    elif sec == "drop":
        emitters, fan, spread, amp, flick = range(6), 7, 100, 35, False
    else:
        emitters, fan, spread, amp, flick = [1, 2, 4], 5, 90, 15, False
    bar = int(b // 4)
    if sec == "intro" and int(t * 12) % 3 == 0:
        return None
    # 16th-note gating on some drop bars
    if sec == "drop" and bar in (5, 7) and int(b * 4) % 2 == 1:
        return None
    if sec == "outro":
        if b > 34:
            return None
    palette_by_bar = [(0.33,), (0.33,), (0.33, 0.5), (0.5,), (0.33, 0.83), (0.0, 0.33, 0.66),
                      (0.83, 0.5), ("rainbow",)]
    pal = palette_by_bar[bar % len(palette_by_bar)] if sec == "drop" else (0.33,)
    converge = sec == "drop" and bar == 6  # tunnel: beams meet over the stage
    s = cam[0] / 2
    target = to_out(MANDALA_C, cam)
    for ei in emitters:
        ex, ey = to_out(EMITTERS[ei], cam)
        ex, ey = ex / 2, ey / 2
        base = -90 + (HW / 2 - ex) / HW * 50
        for i in range(fan):
            off = spread * ((i / (fan - 1)) - 0.5) if fan > 1 else 0
            sweep = amp * math.sin(t * (2.2 if sec == "drop" else 1.1) + ei * 1.7 + i * 0.15)
            ang = math.radians(base + off + sweep)
            if converge:
                tx, ty = target[0] / 2, target[1] / 2
                ang = math.atan2(ty - ey, tx - ex) + math.radians(off * 0.15)
            hsel = pal[(i + ei) % len(pal)]
            col = hsv(t * 0.5 + i / fan) if hsel == "rainbow" else hsv(hsel)
            col = tuple(int(255 * c) for c in col)
            L = 2500
            dr.line([(ex, ey), (ex + L * math.cos(ang), ey + L * math.sin(ang))], fill=col, width=2)
        dr.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(255, 255, 255))
    glow = layer.filter(ImageFilter.GaussianBlur(5))
    a = np.asarray(layer, np.float32) / 255
    g = np.asarray(glow, np.float32) / 255
    out = a * 0.9 + g * 2.2
    if flick:
        out *= 0.8
    return np.asarray(Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                      np.float32) / 255


def note_particles():
    rng = np.random.default_rng(3)
    parts = []
    for b8 in range(16, 64):  # eighth notes from beat 8 to 32
        b = b8 / 2
        count = 1 if b < 16 else 3
        if b < 16 and b8 % 2:
            continue
        for _ in range(count):
            origin = DJ if rng.random() < 0.6 else EMITTERS[rng.integers(len(EMITTERS))]
            parts.append(dict(
                t0=b * BEAT + rng.uniform(0, 0.05), origin=origin,
                ang=math.radians(rng.uniform(-155, -25)),
                speed=rng.uniform(520, 950), spin=rng.uniform(-260, 260),
                sprite=int(rng.integers(4)), hue=rng.random(), size=rng.uniform(45, 85),
            ))
    # celebratory burst on the drop
    for k in range(28):
        parts.append(dict(t0=16 * BEAT, origin=DJ, ang=2 * math.pi * k / 28, speed=1100,
                          spin=rng.uniform(-300, 300), sprite=k % 4, hue=k / 28, size=70))
    return parts


PARTS = None


def draw_notes(t, cam, core, glowlayer):
    for p in PARTS:
        age = t - p["t0"]
        if age < 0 or age > 2.4:
            continue
        ox, oy = to_out(p["origin"], cam)
        drag = (1 - math.exp(-age * 1.2)) / 1.2
        x = ox + math.cos(p["ang"]) * p["speed"] * drag
        y = oy + math.sin(p["ang"]) * p["speed"] * drag - 60 * age * age
        size = p["size"] * (1 + 0.9 * age)
        alpha = min(1, age * 8) * min(1, (2.4 - age) / 0.6)
        sp = SPRITES[p["sprite"]]
        h = int(size)
        w = max(1, int(size * sp.width / sp.height))
        im = sp.resize((w, h), Image.BILINEAR).rotate(p["spin"] * age, expand=True, resample=Image.BILINEAR)
        col = hsv(p["hue"] + t * 0.4, 0.85, 1.0)
        m = np.asarray(im, np.float32) / 255 * alpha
        x0, y0 = int(x - im.width / 2), int(y - im.height / 2)
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W, x0 + im.width), min(H, y0 + im.height)
        if xa >= xb or ya >= yb:
            continue
        mm = m[ya - y0:yb - y0, xa - x0:xb - x0, None]
        core[ya:yb, xa:xb] += mm * np.array(col, np.float32)
        glowlayer.paste(tuple(int(255 * c * alpha) for c in col), (xa, ya, xb, yb),
                        Image.fromarray((mm[..., 0] * 255).astype(np.uint8)))


def draw_geometry(t, b, cam):
    layer = Image.new("RGB", (HW, HH), 0)
    dr = ImageDraw.Draw(layer)
    any_ = False
    if 16 <= b < 33:
        any_ = True
        cx, cy = to_out(MANDALA_C, cam)
        cx, cy = cx / 2, cy / 2
        pulse = env_since_beat(t, 5)
        for ring, (n, k, r) in enumerate([(12, 5, 260), (8, 3, 170), (6, 2, 100), (16, 7, 360)]):
            rot = t * (0.4 if ring % 2 else -0.3) + ring
            rr = r * (1 + 0.08 * pulse)
            pts = [(cx + rr * math.cos(rot + 2 * math.pi * i / n), cy + rr * 0.75 * math.sin(rot + 2 * math.pi * i / n))
                   for i in range(n)]
            col = tuple(int(255 * c * (0.35 + 0.4 * pulse)) for c in hsv(t * 0.3 + ring * 0.25))
            for i in range(n):
                dr.line([pts[i], pts[(i + k) % n]], fill=col, width=1)
    if 8 <= b < 33:
        any_ = True
        dx, dy = to_out(DJ, cam)
        dx, dy = dx / 2, dy / 2
        for back in range(2):
            bb = math.floor(b) - back
            age = (b - bb) * BEAT
            if bb < 8 or age > 0.7:
                continue
            r = 40 + 900 * age
            col = tuple(int(255 * c * (1 - age / 0.7) * 0.8) for c in hsv(bb * 0.13))
            dr.ellipse([dx - r, dy - r * 0.45, dx + r, dy + r * 0.45], outline=col, width=3)
    if not any_:
        return None
    glow = layer.filter(ImageFilter.GaussianBlur(3))
    out = (np.asarray(layer, np.float32) + np.asarray(glow, np.float32) * 1.5)
    return np.asarray(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                      np.float32) / 255


# ----------------------------------------------------------------- frame
YGRAD = None


def render_frame(fi):
    global YGRAD, PARTS
    if PARTS is None:
        PARTS = note_particles()
    if YGRAD is None:
        y = np.linspace(0, 1, H, dtype=np.float32)
        YGRAD = np.clip(1.2 - y * 1.1, 0.25, 1.0)[:, None, None]
    t = fi / FPS
    b = beat_at(t)
    sec = section(b)
    cam = camera(t, b)
    s, cx, cy = cam
    img = SRC.transform((W, H), Image.AFFINE, (1 / s, 0, cx - W / 2 / s, 0, 1 / s, cy - H / 2 / s), Image.BILINEAR)
    f = np.asarray(img, np.float32) / 255

    # rippling canopy: per-row horizontal shift, strongest at top
    amp = {"intro": 3, "groove": 6, "drop": 12, "outro": 4}[sec]
    rows = np.arange(H)
    shift = (amp * np.sin(rows / 55.0 + t * 4.0) * np.clip(1 - rows / (H * 0.6), 0, 1)).astype(int)
    cols = (np.arange(W)[None, :] - shift[:, None]) % W
    f = np.take_along_axis(f, cols[..., None].repeat(3, 2), axis=1)

    # hue cycling
    if sec == "intro":
        deg = 25 * math.sin(t * 1.5)
    elif sec == "groove":
        deg = 40 * math.sin(t * 2.0) + 20 * (b - 8)
    elif sec == "drop":
        deg = 160 + (b - 16) * 22.5 + 30 * env_since_beat(t, 8)
    else:
        deg = 520 + (b - 32) * 10
    # rotate hue per horizontal band so the canopy trips hardest and colours never grey out
    bands = 32
    edges = np.linspace(0, H, bands + 1).astype(int)
    for i in range(bands):
        y0, y1 = edges[i], edges[i + 1]
        w = float(YGRAD[(y0 + y1) // 2, 0, 0])
        f[y0:y1] = np.clip(hue_rotate(f[y0:y1], deg * w), 0, 1)

    # saturation/contrast boost
    lum = f.mean(axis=2, keepdims=True)
    f = np.clip(lum + (f - lum) * 1.35, 0, 1)

    # kick brightness pump
    if 8 <= b < 32:
        f *= 0.9 + 0.25 * env_since_beat(t, 10)
    elif b < 8:
        f *= 0.75 + 0.15 * (b / 8)

    # searchlights
    cone, disc = draw_searchlights(t, b, cam)
    warm = np.array([1.0, 0.95, 0.85], np.float32)
    f = f + cone[..., None] * 0.35 * warm + f * disc[..., None] * 1.6 + disc[..., None] * 0.18 * warm

    geo = draw_geometry(t, b, cam)
    if geo is not None:
        f = f + geo
    las = draw_lasers(t, b, cam)
    if las is not None:
        f = f + las

    core = np.zeros((H, W, 3), np.float32)
    nglow = Image.new("RGB", (W, H), 0)
    draw_notes(t, cam, core, nglow)
    nglow = nglow.resize((HW, HH), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR)
    f = f + core + np.asarray(nglow, np.float32) / 255 * 1.8

    # mirror flash (kaleidoscope symmetry) on the first beat of some drop bars
    if sec == "drop" and int(b // 4) % 2 == 1 and (b % 4) < 1:
        k = 1 - (b % 4)
        f = np.maximum(f, f[:, ::-1] * k)

    # chromatic aberration on kicks
    if 8 <= b < 32:
        d = int(10 * env_since_beat(t, 10))
        if d:
            f[..., 0] = np.roll(f[..., 0], d, axis=1)
            f[..., 2] = np.roll(f[..., 2], -d, axis=1)

    # strobe: 16th flashes on the drop's first beat, and on the final hit
    if (16 <= b < 17 and int(b * 4) % 2 == 0) or (32 <= b < 32.25) or (8 <= b < 8.15):
        f = f * 0.5 + 0.7

    # vignette + soft clip
    yy, xx = np.ogrid[-1:1:H * 1j, -1:1:W * 1j]
    vig = 1 - 0.35 * np.clip(xx ** 2 + (yy * 0.8) ** 2 - 0.3, 0, 1)
    f = f * vig[..., None]
    f = 1 - np.exp(-f * 1.6)
    f /= 1 - math.exp(-1.6)
    # fade in/out
    if t < 0.4:
        f *= t / 0.4
    if t > DUR - 0.8:
        f *= max(0, (DUR - t) / 0.8)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes()


def main():
    src, wav, out = sys.argv[1:4]
    only = int(sys.argv[4]) if len(sys.argv) > 4 else None
    if only is not None:  # still preview of a single frame
        init(src)
        Image.frombytes("RGB", (W, H), render_frame(only)).save(out)
        return
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out,
    ], stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init, initargs=(src,)) as pool:
        for i, fr in enumerate(pool.imap(render_frame, range(NFRAMES), chunksize=2)):
            ff.stdin.write(fr)
            if i % 60 == 0:
                print(f"frame {i}/{NFRAMES}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()

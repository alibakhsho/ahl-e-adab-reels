"""Synthesise a short 140 BPM psytrance loop that the visuals are synced to.

Pure numpy, no samples, so the track is original and royalty-free.
Structure (in beats): 0-8 intro/build, 8-16 groove, 16-32 drop, 32-36 outro.
"""
import wave

import numpy as np

SR = 44100
BPM = 140
BEAT = 60.0 / BPM
TOTAL_BEATS = 36
DUR = TOTAL_BEATS * BEAT
N = int(DUR * SR) + SR  # one extra second of tail
rng = np.random.default_rng(7)


def t_axis(n):
    return np.arange(n) / SR


def add(buf, sig, at):
    i = int(at * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def onepole_lp(x, a):
    """Cheap one-pole low-pass; a in (0,1], smaller = darker. a may be an array."""
    y = np.empty_like(x)
    acc = 0.0
    a = np.broadcast_to(a, x.shape)
    for k in range(len(x)):
        acc += a[k] * (x[k] - acc)
        y[k] = acc
    return y


def svf_lp(x, cutoff, q):
    """Chamberlin state-variable low-pass with per-sample cutoff (Hz)."""
    f = 2 * np.sin(np.pi * np.clip(cutoff, 20, SR / 6) / SR)
    low = band = 0.0
    out = np.empty_like(x)
    damp = 1.0 / q
    for k in range(len(x)):
        low += f[k] * band
        high = x[k] - low - damp * band
        band += f[k] * high
        out[k] = low
    return out


def saw(freq, n, detune=0.0):
    ph = np.cumsum(np.full(n, freq * (1 + detune)) / SR)
    return 2 * (ph % 1.0) - 1


# ---------------------------------------------------------------- instruments
def kick():
    n = int(0.32 * SR)
    t = t_axis(n)
    f = 45 + 160 * np.exp(-t * 38)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.exp(-t * 9)
    click = np.exp(-t * 600) * 0.6
    return np.tanh(2.2 * np.sin(ph) * env) * 0.95 + click * rng.uniform(-1, 1, n) * 0.15


def bass_note(freq, length):
    n = int(length * SR)
    t = t_axis(n)
    s = saw(freq, n) + 0.5 * saw(freq * 2, n)
    s = onepole_lp(s, 0.06 + 0.25 * np.exp(-t * 40))
    env = np.minimum(1, t * 400) * np.exp(-t * 18)
    return np.tanh(3 * s * env) * 0.55


def hat(open_=False):
    n = int((0.22 if open_ else 0.05) * SR)
    t = t_axis(n)
    x = rng.uniform(-1, 1, n)
    x = np.diff(np.diff(x, prepend=0), prepend=0)  # crude high-pass
    env = np.exp(-t * (14 if open_ else 90))
    return x * env * (0.16 if open_ else 0.12)


def snare():
    n = int(0.18 * SR)
    t = t_axis(n)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.4
    noise = np.diff(rng.uniform(-1, 1, n), prepend=0) * np.exp(-t * 22) * 0.5
    return (tone + noise) * 0.55


def crash():
    n = int(2.5 * SR)
    t = t_axis(n)
    x = np.diff(rng.uniform(-1, 1, n), prepend=0)
    return x * np.exp(-t * 1.6) * 0.25


def pad(freqs, length):
    n = int(length * SR)
    t = t_axis(n)
    s = np.zeros(n)
    for f in freqs:
        for d in (-0.004, 0.0, 0.005):
            s += saw(f, n, d)
    s = onepole_lp(s / (len(freqs) * 3), 0.03)
    env = np.minimum(1, t / 1.5) * np.minimum(1, (length - t) / 1.0)
    return s * env * 0.9


# ---------------------------------------------------------------- arrangement
def build():
    mix = np.zeros(N)
    E1, E2 = 41.2, 82.41
    pad_freqs = [164.8, 196.0, 246.9, 329.6]  # E minor

    # pad through intro, comes back in the outro
    add(mix, pad(pad_freqs, 8 * BEAT + 0.5), 0)
    add(mix, pad(pad_freqs, 4 * BEAT + 1.0), 32 * BEAT)

    # riser 0..8 beats (filtered noise sweeping up)
    n = int(8 * BEAT * SR)
    t = t_axis(n)
    cut = 0.01 + 0.5 * (t / t[-1]) ** 2
    riser = onepole_lp(rng.uniform(-1, 1, n), cut) * (t / t[-1]) ** 1.5 * 0.5
    add(mix, riser, 0)

    # second riser into the drop, beats 12..16
    n = int(4 * BEAT * SR)
    t = t_axis(n)
    cut = 0.02 + 0.6 * (t / t[-1]) ** 2
    add(mix, onepole_lp(rng.uniform(-1, 1, n), cut) * (t / t[-1]) ** 2 * 0.45, 12 * BEAT)

    # snare roll beats 4..8 accelerating
    for b in range(4, 8):
        div = 2 if b < 6 else 4 if b < 7 else 8
        for k in range(div):
            add(mix, snare() * (0.4 + 0.15 * (b - 4)), (b + k / div) * BEAT)

    for b in range(2, 32):
        add(mix, hat(open_=b >= 8), (b + 0.5) * BEAT)
        if b >= 8:
            for k in (0.25, 0.75):
                add(mix, hat() * 0.7, (b + k) * BEAT)

    # kick + rolling bass (kick on 1, bass on the three 16ths after)
    for b in range(8, 32):
        if b == 15:  # half-beat break before the drop
            add(mix, kick(), b * BEAT)
            continue
        add(mix, kick(), b * BEAT)
        for k in (1, 2, 3):
            note = E2 if (b >= 16 and k == 3 and b % 4 == 3) else E1 * 2
            add(mix, bass_note(note, BEAT / 4 * 0.95), (b + k / 4) * BEAT)

    # final hit
    add(mix, kick() * 1.1, 32 * BEAT)
    for at in (8, 16, 24, 32):
        add(mix, crash(), at * BEAT)

    # acid lead in the drop: 16th notes, phrygian, resonant sweep
    scale = [329.6, 349.2, 392.0, 440.0, 493.9, 523.3, 587.3, 659.3]
    patt = [0, 0, 7, 0, 3, 0, 5, 1, 0, 0, 7, 4, 3, 1, 0, 2]
    start, end = 16 * BEAT, 32 * BEAT
    n = int((end - start) * SR)
    t = t_axis(n)
    step = BEAT / 4
    freqs = np.array([scale[patt[int(tt / step) % 16]] / 2 for tt in t])
    ph = np.cumsum(freqs) / SR
    sig = 2 * (ph % 1) - 1
    gate = np.exp(-(t % step) * 25)
    cutoff = 300 + 2800 * (0.5 + 0.5 * np.sin(2 * np.pi * t / (8 * BEAT) - np.pi / 2)) * (0.4 + 0.6 * gate)
    acid = np.tanh(2.5 * svf_lp(sig * gate, cutoff, 6.0)) * 0.22
    add(mix, acid, start)

    # simple stereo widening: delayed copy on the right
    d = int(SR * BEAT * 0.75)
    right = mix.copy()
    right[d:] += 0.25 * mix[:-d]
    left = mix.copy()
    h = d // 2
    left[h:] += 0.18 * mix[:-h]
    st = np.stack([left, right], axis=1)
    st /= np.max(np.abs(st)) + 1e-9
    st = np.tanh(st * 1.4) / np.tanh(1.4)
    fade = int(SR * 1.2)
    st[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return st


def write_wav(path):
    st = build()
    data = (st * 32000).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


if __name__ == "__main__":
    import sys

    write_wav(sys.argv[1] if len(sys.argv) > 1 else "track.wav")

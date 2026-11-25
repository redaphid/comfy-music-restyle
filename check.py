#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "faster-whisper"]
# ///
"""Score restyle takes against the original (needs ffmpeg on PATH):
  uv run check.py in/song.wav out/raw/*.flac [--bpm 132] [--no-whisper]
Prints per take: key (top pitch classes), kick period/BPM + pulse strength (35-120 Hz onset autocorrelation),
chroma similarity to the original, best beat offset (s) to delay the take by when layering the original on top,
and a Whisper small.en transcript. 'Thank you' / 'Thanks for watching' on its own = Whisper hallucinating on no vocals."""
import argparse, subprocess, numpy as np
SR, H = 22050, 256; NAMES = "C C# D D# E F F# G G# A A# B".split()
def load(f, t): return np.frombuffer(subprocess.run(["ffmpeg","-loglevel","error","-i",f,"-ac","1","-ar",str(SR),"-t",str(t),"-f","f32le","-"],capture_output=True).stdout,np.float32)
def stft(y):
    fr = np.lib.stride_tricks.sliding_window_view(y, 2048)[::H] * np.hanning(2048); return np.abs(np.fft.rfft(fr)), np.fft.rfftfreq(2048, 1/SR)
def onset(X, f, lo, hi):
    o = np.maximum(np.diff(np.log1p(X[:, (f > lo) & (f < hi)]), axis=0), 0).sum(1); return o - o.mean()
def chroma(X, f):
    m = (f > 80) & (f < 2000); idx = (np.round(12*np.log2(f[m]/261.63)) % 12).astype(int)
    C = np.stack([X[:, m][:, idx == i].sum(1) for i in range(12)], 1); return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
p = argparse.ArgumentParser(); p.add_argument("orig"); p.add_argument("takes", nargs="+")
p.add_argument("--bpm", type=float, help="source tempo; estimated from the original when omitted"); p.add_argument("--no-whisper", action="store_true"); p.add_argument("--secs", type=float, default=60)
a = p.parse_args()
Xo, f = stft(load(a.orig, a.secs)); Co, Oo = chroma(Xo, f), onset(Xo, f, 30, 11000)
if not a.bpm:
    ac = np.correlate(Oo, Oo, "full")[len(Oo)-1:]; lags = np.arange(len(ac)) * H / SR; m = (lags > 60/180) & (lags < 60/70)
    a.bpm = 60 / lags[m][np.argmax(ac[m])]
key0 = " ".join(NAMES[i] for i in np.argsort(Co.sum(0))[::-1][:4])
print(f"original: ~{a.bpm:.1f} BPM, top pitch classes [{key0}]", flush=True)
beat = 60 / a.bpm
wm = None
if not a.no_whisper:
    from faster_whisper import WhisperModel; wm = WhisperModel("small.en", device="cpu", compute_type="int8")
for t in a.takes:
    y = load(t, a.secs); X, _ = stft(y); C = chroma(X, f); n = min(len(C), len(Co))
    k = onset(X, f, 35, 120); ac = np.correlate(k, k, "full")[len(k)-1:]; ac /= ac[0] + 1e-9
    lags = np.arange(len(ac)) * H / SR; m = (lags > beat*0.8) & (lags < beat*1.25); L = lags[m][np.argmax(ac[m])]
    o = onset(X, f, 30, 11000); n2 = min(len(o), len(Oo)); w = int(beat * SR / H)
    xc = [np.dot(o[:n2-s], Oo[s:n2]) if s >= 0 else np.dot(o[-s:n2], Oo[:n2+s]) for s in range(-w, w+1)]
    key = " ".join(NAMES[i] for i in np.argsort(C.sum(0))[::-1][:4])
    line = f"{t}: key[{key}] bpm {60/L:.1f} kick {ac[m].max():.2f} sim {(C[:n]*Co[:n]).sum(1).mean():.2f} offset {(np.argmax(xc)-w)*H/SR:+.3f}s"
    if wm:
        a16 = np.frombuffer(subprocess.run(["ffmpeg","-loglevel","error","-i",t,"-ac","1","-ar","16000","-f","f32le","-"],capture_output=True).stdout, np.float32)
        line += " | " + " / ".join(s.text.strip() for s in wm.transcribe(a16, beam_size=5, condition_on_previous_text=False)[0])
    print(line, flush=True)

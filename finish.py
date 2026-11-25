#!/usr/bin/env python3
"""Finish a take with ffmpeg: optional layering of the original on top, fade-out, loudness to -14 LUFS / -1 dBTP.

  python finish.py out/raw/v1-d100-s1.flac out/song-chant.wav
  python finish.py out/raw/v1-d100-s1.flac out/song-layered.wav --layer in/song.wav --offset 0.163

--offset comes from check.py: positive = delay the take, negative = delay the original."""
import argparse, subprocess

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("take"); p.add_argument("dest")
p.add_argument("--layer", help="original audio to mix on top of the take")
p.add_argument("--offset", type=float, default=0.0, help="seconds; see check.py")
p.add_argument("--bed-lufs", type=float, default=-17); p.add_argument("--top-lufs", type=float, default=-19)
p.add_argument("--fade", type=float, default=0.8, help="fade-out seconds")
p.add_argument("--lufs", type=float, default=-14)
a = p.parse_args()

dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.take],
                           capture_output=True, text=True).stdout) + max(a.offset, 0)
tail = f"afade=t=out:st={max(dur - a.fade, 0):.2f}:d={a.fade},loudnorm=I={a.lufs}:TP=-1,aresample=44100"
cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", a.take]
if a.layer:
    ms = round(abs(a.offset) * 1000)
    dg, do = (f"adelay={ms}:all=1,", "") if a.offset >= 0 else ("", f"adelay={ms}:all=1,")
    cmd += ["-i", a.layer, "-filter_complex",
            f"[0:a]aresample=44100,{dg}loudnorm=I={a.bed_lufs}:TP=-2[g];[1:a]aresample=44100,{do}loudnorm=I={a.top_lufs}:TP=-2[j];"
            f"[g][j]amix=inputs=2:normalize=0:duration=first,{tail}"]
else:
    cmd += ["-af", tail]
subprocess.run(cmd + ["-c:a", "pcm_s16le", a.dest], check=True)
print("wrote", a.dest)

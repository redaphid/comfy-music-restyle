#!/usr/bin/env python3
"""Restyle a song with ACE-Step 1.5 on a ComfyUI server. Standard library only.

Uploads the input over HTTP, queues one job per (denoise, seed), and downloads every take to out/raw/.
Denoise below 1.0 reworks the source audio; denoise 1.0 with --codes-at-full generates from the
prompt and lyrics alone (the only way to get clearly sung lyrics).

  python restyle.py in/song.wav --tags "gregorian chant, male choir, techno, four on the floor kick" \
      --lyrics-file lyrics.txt --bpm 132 --key "C major" --denoise 0.5,0.85,1.0 --seeds 1-3 --codes-at-full
"""
import argparse, json, os, subprocess, time, urllib.parse, urllib.request, uuid

def call(url, path, body=None, raw=False, headers=None):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data, headers or {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read() if raw else json.load(r)

def upload(url, path):
    """POST the file to /upload/image (ComfyUI's generic input upload; it takes audio too)."""
    b = uuid.uuid4().hex
    name = "restyle-" + os.path.basename(path)
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + open(path, "rb").read() + \
           f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n".encode()
    r = call(url, "/upload/image", body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return (r["subfolder"] + "/" if r.get("subfolder") else "") + r["name"]

def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return round(float(out), 1)

def seeds(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out

def graph(a, audio, seed, denoise, codes, prefix):
    model = ["unet", 0]
    g = {"unet": ("UNETLoader", {"unet_name": a.unet, "weight_dtype": "default"})}
    if a.lora:
        name, _, strength = a.lora.rpartition(":")
        g["lora"] = ("LoraLoaderModelOnly", {"model": model, "lora_name": name, "strength_model": float(strength)})
        model = ["lora", 0]
    g.update({
        "shift": ("ModelSamplingAuraFlow", {"shift": 3.0, "model": model}),
        "clip": ("DualCLIPLoader", {"clip_name1": "qwen_0.6b_ace15.safetensors", "clip_name2": "qwen_1.7b_ace15.safetensors",
                                    "type": "ace", "device": "default"}),
        "vae": ("VAELoader", {"vae_name": "ace_1.5_vae.safetensors"}),
        "load": ("LoadAudio", {"audio": audio}),
        "encode": ("VAEEncodeAudio", {"audio": ["load", 0], "vae": ["vae", 0]}),
        "text": ("TextEncodeAceStepAudio1.5", {"clip": ["clip", 0], "tags": a.tags, "lyrics": a.lyrics, "seed": seed,
                 "bpm": a.bpm, "duration": a.seconds, "timesignature": "4", "language": a.lang, "keyscale": a.key,
                 "generate_audio_codes": codes, "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}),
        "neg": ("ConditioningZeroOut", {"conditioning": ["text", 0]}),
        "sample": ("KSampler", {"seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                   "denoise": denoise, "model": ["shift", 0], "positive": ["text", 0], "negative": ["neg", 0],
                   "latent_image": ["encode", 0]}),
        "decode": ("VAEDecodeAudio", {"samples": ["sample", 0], "vae": ["vae", 0]}),
        "save": ("SaveAudio", {"filename_prefix": prefix, "audio": ["decode", 0]}),
    })
    return {k: {"class_type": c, "inputs": i} for k, (c, i) in g.items()}

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="audio file to restyle (e.g. in/song.wav)")
    p.add_argument("--tags", required=True, help="style prompt; include a LoRA's trigger word if you use one")
    p.add_argument("--lyrics", default="[Instrumental]")
    p.add_argument("--lyrics-file")
    p.add_argument("--lang", default="en", help="lyrics language code (en, la, de, ja, ...)")
    p.add_argument("--bpm", type=int, default=120, help="measure the source; check.py helps")
    p.add_argument("--key", default="C major", help='e.g. "C major", "F# minor"')
    p.add_argument("--seconds", type=float, help="defaults to the input's length")
    p.add_argument("--denoise", default="0.5,0.7,0.85,1.0", help="comma list; 1.0 = ignore the source audio")
    p.add_argument("--seeds", default="1-2", help="e.g. 1-4 or 1,5,9")
    p.add_argument("--codes-at-full", action="store_true", help="LM audio codes on for denoise 1.0 (needed for sung lyrics)")
    p.add_argument("--lora", help="name:strength as ComfyUI lists it, e.g. techno-v1.safetensors:0.8")
    p.add_argument("--unet", default="acestep_v1.5_turbo.safetensors")
    p.add_argument("--tag", default="v1", help="prefix for output names")
    p.add_argument("--out", default="out/raw")
    p.add_argument("--url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    a = p.parse_args()
    if a.lyrics_file:
        a.lyrics = open(a.lyrics_file, encoding="utf-8").read()
    a.seconds = a.seconds or duration(a.input)
    os.makedirs(a.out, exist_ok=True)
    audio = upload(a.url, a.input)
    jobs = []
    for d in [float(x) for x in a.denoise.split(",")]:
        for s in seeds(a.seeds):
            name = f"{a.tag}-d{round(d * 100):03d}-s{s}"
            wf = graph(a, audio, s, d, a.codes_at_full and d >= 0.99, f"restyle/{name}")
            jobs.append((call(a.url, "/prompt", {"prompt": wf})["prompt_id"], name))
            print("queued", name, flush=True)
    for pid, name in jobs:
        while pid not in (h := call(a.url, f"/history/{pid}")):
            time.sleep(2)
        if h[pid]["status"].get("status_str") == "error":
            print("ERROR", name, json.dumps(h[pid]["status"].get("messages"))[:1500])
            continue
        for out in h[pid]["outputs"].values():
            for f in out.get("audio", []):
                q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f["subfolder"], "type": f["type"]})
                dst = os.path.join(a.out, name + os.path.splitext(f["filename"])[1])
                open(dst, "wb").write(call(a.url, f"/view?{q}", raw=True))
                print("done", dst, flush=True)

if __name__ == "__main__":
    main()

# comfy-music-restyle

Restyle a song or jingle with [ACE-Step 1.5](https://huggingface.co/ACE-Step) running in [ComfyUI](https://github.com/comfyanonymous/ComfyUI):
new genre, new sung lyrics, an added beat. Example: a cheerful keyboard jingle turned into gregorian chant over a techno kick.

Three small scripts, driven over ComfyUI's HTTP API (works against a local or remote server):

| script | does | needs |
|---|---|---|
| `restyle.py` | uploads `in/<song>`, queues a sweep of denoise × seeds, downloads takes to `out/raw/` | Python 3.8+, stdlib only; `ffprobe` |
| `check.py` | scores each take: tempo and kick strength, key, similarity to the original, beat offset, Whisper transcript | [uv](https://docs.astral.sh/uv/) (installs numpy + faster-whisper); `ffmpeg` |
| `finish.py` | optional layering of the original on top, fade-out, -14 LUFS / -1 dBTP | `ffmpeg` |

`in/` and `out/` are git-ignored: drop your audio in `in/`, results land in `out/`.

## Setup

ComfyUI 0.4x or newer (native ACE-Step 1.5 nodes), with these models in its `models/` folder
([Comfy-Org/ace_step_1.5_ComfyUI_files](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files)):

- `diffusion_models/acestep_v1.5_turbo.safetensors`
- `text_encoders/qwen_0.6b_ace15.safetensors`, `text_encoders/qwen_1.7b_ace15.safetensors`
- `vae/ace_1.5_vae.safetensors`

Optional style LoRAs (put in `models/loras/`, pass with `--lora name:strength`, and add the trigger word to `--tags`):
[techno](https://huggingface.co/ryanontheinside/techno-acestep1.5-v1) (`roti-t3kn0`),
[edm](https://huggingface.co/ryanontheinside/edm-acestep1.5-v1) (`roti-3dmm`),
[electronic](https://huggingface.co/ryanontheinside/electronic-acestep1.5-v1) (`roti-3l3ktr0n`).

Point the scripts at your server with `--url` or `COMFY_URL` (default `http://127.0.0.1:8188`).

## Use

```sh
# 1. Measure the original (tempo, key)
uv run check.py in/song.wav in/song.wav --no-whisper

# 2. Sweep: partial denoise reworks the song; 1.0 + codes generates fresh with sung lyrics
python restyle.py in/song.wav --bpm 132 --key "C major" \
  --tags "roti-t3kn0, gregorian chant, male monk choir, cathedral reverb, driving techno, four on the floor kick" \
  --lyrics-file examples/lyrics.txt --lora techno-v1.safetensors:0.8 \
  --denoise 0.5,0.7,0.85,1.0 --seeds 1-3 --codes-at-full

# 3. Score every take
uv run check.py in/song.wav out/raw/*.flac

# 4. Finish the picks (layer the original over a generated bed using check.py's offset)
python finish.py out/raw/v1-d100-s1.flac out/song-chant.wav
python finish.py out/raw/v1-d100-s1.flac out/song-layered.wav --layer in/song.wav --offset 0.163
```

About 5 s per 17 s take on an RTX 4090, plus ~20 s to load the models the first time.

## What the denoise knob does

Measured on a 17 s, 132 BPM jingle (techno LoRA 0.8, 8 steps):

| denoise | keeps original (chroma similarity) | vocals heard by Whisper | beat |
|---|---|---|---|
| 0.5 | 0.79 | none | kick and hats over the original's grid |
| 0.65–0.8 | 0.56–0.64 | none; a sub-bass drone appears | weak |
| 0.85–0.95 | 0.50–0.56 | fragments of 1–2 lines | mixed |
| 1.0 + `--codes-at-full` | 0.40–0.56 | every line, near-verbatim | strong four-on-the-floor in ~2 of 5 seeds |

- Partial denoise does not add intelligible vocals: the source audio wins. Sung lyrics need denoise 1.0 with the
  LM audio codes on, which keeps nothing of the original, so layer the original back on with `finish.py --layer`
  (pick a bed whose key agrees with the original; `check.py` prints both).
- Whisper prints "Thank you." on instrumentals. That is a hallucination; read it as "no vocals".
- Made-up words come back spelled by sound (a coined name like "Zibi" may read as "Cb"). That still counts as sung.

## License

MIT

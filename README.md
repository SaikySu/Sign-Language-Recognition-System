
# Sign Reader (YOLO + Temporal Aggregation)

A lightweight, real‑time demo that reads **hand‑sign letters** from a webcam using **Ultralytics YOLO**, then converts the per‑frame predictions into **stable characters and words** with temporal smoothing and robust character‑commit rules.


---

## Features

- **Real‑time detection** with Ultralytics YOLO.
- **Temporal smoothing** (majority voting + confidence threshold) to reduce jitter.
- **Character aggregator** with:
  - K‑stable commit (only append when label is stable for K frames).
  - **De‑duplication** (avoid repeating the same letter without enough BLANK gap).
  - **Auto‑space** insertion after a configurable silent duration.
- **HUD overlay**: live label, smoothed label, FPS, and bounding box.
- **Windows camera backends** fallback (`msmf` → `dshow` → auto) with FOURCC tuning.
- **Keyboard controls**: backspace, space, clear, quit, and quick save to `output.txt`.

---

## Requirements

- Python 3.9+
- `ultralytics`, `opencv-python`, `numpy`

```bash
pip install requirements.txt
```

> Tip: For GPU acceleration, install a CUDA‑enabled PyTorch before `ultralytics` if desired.

---

## Quick Start

1. Export or train a YOLO model for **letter classes** (e.g., A‑Z). If your dataset includes an explicit blank class (e.g., `blank`/`nothing`), note its name.
2. Run the script with your model path:
   ```bash
   python sign_reader_yolo.py --model path/to/best.pt
   ```
3. Optional (Windows): try another camera backend if frames don't arrive:
   ```bash
   python sign_reader_yolo.py --model path/to/best.pt --api dshow
   ```

---

## CLI Arguments

| Flag | Default | Description |
|---|---:|---|
| `--model` | *required* | Path to YOLO `.pt` (Ultralytics). |
| `--camera` | `0` | Camera index. |
| `--api` | `auto` | Camera backend: `auto`, `dshow`, `msmf` (Windows). |
| `--conf` | `0.35` | Min confidence to accept a letter. |
| `--window` | `10` | Smoothing window size (frames). |
| `--kstable` | `4` | Frames of identical label required to commit a character. |
| `--pmajor` | `0.6` | Majority ratio within the window. |
| `--confth` | `0.7` | Avg confidence threshold within the window. |
| `--gap` | `8` | Min BLANK frames between identical letters to allow repetition. |
| `--space_ms` | `800` | Silence duration (ms) to auto‑insert a space. |
| `--class-blank` | `None` | Name of explicit blank class (e.g., `blank`, `nothing`). |
| `--letters` | `None` | Comma‑separated letter list (overrides auto‑infer from `model.names`). |
| `--width` | `960` | Resize width for display (0 = keep original). |
| `--cam-width` | `1280` | Requested camera width. |
| `--cam-height` | `720` | Requested camera height. |
| `--fps` | `30` | Requested camera FPS. |
| `--fourcc` | `MJPG` | Preferred FOURCC (e.g., `MJPG`, `YUY2`, `H264`, `RAW`). |

---

## Keyboard Shortcuts

- **Q / ESC**: Quit
- **Backspace**: Remove last character
- **Space**: Insert a space
- **Enter**: Print current text and append it to `output.txt`
- **C**: Clear all committed text

---

## How It Works

**TemporalSmoother**
- Maintains a sliding window of `(label, confidence)` and uses majority vote plus an average‑confidence threshold to emit a **stable** label (or `BLANK`).

**CharAggregator**
- Commits a character only if the smoothed label is **stable for K frames**.
- Prevents duplicate letters unless separated by **enough BLANK frames**.
- Automatically inserts a **space** if no non‑blank has appeared for `space_ms` milliseconds.

**HUD**
- Draws the current and smoothed labels, the active bounding box, FPS, and the evolving text line at the top of the frame.

---

## Model Notes

- By default the script infers valid letters from `model.names` by picking **single‑character alphabetic** class names—fallback: prefixes like `A_sign` → `A`.
- If your dataset doesn't follow this convention, pass `--letters "A,B,C,..."` explicitly.
- If you trained a `blank`/`nothing` class, set `--class-blank` to its exact name so the script can prefer it when confidence beats letters.

---

## Troubleshooting

- **No frames / camera busy on Windows**: try `--api dshow` or change `--fourcc` to `YUY2`/`H264`. Reduce `--cam-width/--cam-height` and `--fps`.
- **Jittery letters**: increase `--window`, `--kstable`, and/or `--confth`.
- **Missing letters or wrong mapping**: specify `--letters` to match your model classes.
- **Repeated characters**: raise `--gap`.
- **Spaces too frequent/rare**: adjust `--space_ms`.

---

## Output

- The current assembled text is shown on the top banner.
- Press **Enter** to append the line to `output.txt` in the working directory.
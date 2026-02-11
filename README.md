# HighImage

HighImage provides image super-resolution with:
- Traditional pipeline (bicubic + sharpening + IBP)
- SRCNN model-based pipeline (GUI supports x2/x3 scale)

Default GUI model path:
- `SRCNN-pytorch-master/best.pth`

## Option A: Install And Run With Scripts (Recommended For New PC)

From File Explorer, double-click in this order:
1. `install_dependencies.bat`
2. `run_app.bat`

Notes:
- `install_dependencies.bat` installs Python dependencies from `requirements.txt`.
- If Python is missing, it tries to install Python 3.12 via `winget`.
- `run_app.bat` launches `app.py`.

## Option B: Manual Install And Manual Run

### 1) Install Python

Install Python 3.10+ first, and ensure `python` (or `py -3`) is available in terminal.

### 2) Install Dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 3) Run GUI

```powershell
python app.py
```

## Manual CLI Usage

Traditional pipeline:

```powershell
python sr.py --input input.jpg --output output.png --scale 2 --ibp-iters 10 --ibp-lambda 1.0 --sharpen-amount 0.9 --sharpen-radius 1.3 --detail-boost 0.25
```

Batch-generate bicubic baseline images:

```powershell
python tools\generate_bicubic_batch.py --input-dir your_images --output-dir your_bicubic --scale 3 --recursive
```

## Requirements

Dependencies are managed by `requirements.txt`:
- `numpy`
- `pillow`
- `opencv-python`
- `torch`
- `h5py`
- `tqdm`

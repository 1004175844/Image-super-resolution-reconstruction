# Repository Guidelines

## Project Structure & Module Organization
`app.py` contains the Tkinter GUI and user workflow. Core image processing lives in `sr.py`, which exposes `super_resolve()` and `super_resolve_srcnn()`. `SRCNN-pytorch-master/` holds the reference SRCNN code plus pretrained weights (for example `srcnn_x2.pth`). `imgs/` stores sample images or assets used for demos. `requirements.txt` is the minimal dependency list. Keep UI helpers in `app.py`, processing logic in `sr.py`, and avoid hidden state outside these modules.

## Build, Test, and Development Commands
- `python -m venv .venv` creates the virtual environment.
- `.venv\Scripts\activate` activates the environment on Windows.
- `pip install -r requirements.txt` installs `numpy`, `pillow`, and `opencv-python`.
- `pip install torch` installs the optional dependency needed for SRCNN mode.
- `python app.py` launches the GUI.
- `python sr.py --input input.jpg --output output.png --scale 2 --ibp-iters 10 --ibp-lambda 1.0 --sharpen-amount 0.9 --sharpen-radius 1.3 --detail-boost 0.25` runs the traditional CLI pipeline.

## Coding Style & Naming Conventions
Use 4-space indentation and follow PEP 8 naming: `snake_case` for functions/variables and `CapWords` for classes (see `App`). Keep image math in `sr.py` and UI concerns in `app.py`. Prefer explicit parameters over global state; surface new knobs through `super_resolve()` or `super_resolve_srcnn()` and the CLI.

## Testing Guidelines
There is no test suite yet. If you add one, use `pytest`, place files under `tests/`, and name them `test_*.py`. Favor deterministic tests with fixed input images in `imgs/` and small fixtures to keep runtime and repo size low. Run with `python -m pytest`.

## Commit & Pull Request Guidelines
No Git history is available here, so follow a clean, consistent convention such as `feat:`, `fix:`, `docs:` with short, imperative summaries. PRs should describe the change, list new parameters, and include before/after output images for algorithm tweaks. For GUI changes, include a screenshot. For CLI changes, include the exact command used.

## Security & Configuration Tips
Treat input images as untrusted data. Avoid committing large binaries; keep sample images small and relevant. Pretrained weights should live under `SRCNN-pytorch-master/` and be referenced by path. If you add new dependencies, update `requirements.txt` and call them out in the PR description.

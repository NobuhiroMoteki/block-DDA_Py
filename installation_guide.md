# block-DDA_Py — Installation Guide

This document describes how to set up the block-DDA_Py runtime environment from scratch on:

- **Windows machine** running WSL2 with Ubuntu (recommended for Windows users)
- **Native Linux machine** (Ubuntu 22.04 / 24.04 LTS)

block-DDA_Py is developed and tested with **Python 3.13 on Linux (WSL2 Ubuntu)**. Native Windows is **not** supported (the package `pywin32` was removed; FFT performance also benefits from Linux thread scheduling).

---

## Common requirements

| Item | Version | Notes |
|------|---------|-------|
| OS | Ubuntu 22.04 / 24.04 LTS | other Debian derivatives should work |
| Python | 3.13.x | install via `uv` (no system Python pollution) |
| Package manager | [uv](https://docs.astral.sh/uv/) ≥ 0.5 | NEVER use `pip install` directly |
| Git | 2.30+ | for cloning |
| LaTeX (optional) | TeX Live 2023+ | only for rebuilding `docs/theory_note.pdf` |
| RAM | ≥ 16 GB | 32 GB+ recommended for large parameter sweeps |
| CPU | x86_64, AVX2 capable | multi-core strongly recommended |

---

## Part A — Windows (WSL2 + Ubuntu)

### A.1 Enable WSL2 and install Ubuntu

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot if prompted. On first launch, set your UNIX username and password.

Verify WSL2 is the default version:

```powershell
wsl --set-default-version 2
wsl -l -v
```

The Ubuntu distribution must show `VERSION 2`.

### A.2 Update Ubuntu and install build tools

Inside the Ubuntu shell:

```sh
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl ca-certificates
```

### A.3 (Optional) Move WSL data to a faster path

If your `%USERPROFILE%` is on a slow disk, export and re-import the distro to a faster location. Skip this step on a typical SSD setup.

### A.4 Install `uv` (Python package manager)

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc          # or restart the shell
uv --version              # should print >= 0.5
```

`uv` will manage Python 3.13 itself — do **not** install Python via `apt`.

### A.5 Clone block-DDA_Py and create the venv

Place the repository under your **Linux** home directory (not under `/mnt/c/...`, which is much slower):

```sh
mkdir -p ~/Python_in_WSL && cd ~/Python_in_WSL
git clone https://github.com/NobuhiroMoteki/block-DDA_Py.git
cd block-DDA_Py
uv venv --python 3.13
uv pip install -r requirements.txt
```

The first `uv venv` invocation will download Python 3.13 automatically.

### A.6 VS Code integration (recommended)

1. Install **VS Code** on Windows.
2. Install the extensions **Remote – WSL**, **Python**, and **Jupyter**.
3. From the Ubuntu shell:

   ```sh
   cd ~/Python_in_WSL/block-DDA_Py
   code .
   ```

4. In VS Code, select the interpreter
   `~/Python_in_WSL/block-DDA_Py/.venv/bin/python` (Cmd/Ctrl-Shift-P → *Python: Select Interpreter*).

### A.7 Smoke test

```sh
uv run python -c "import numpy, scipy, h5py, sklearn; print('OK')"
uv run jupyter notebook test_dda.ipynb
```

Run the cells in `test_dda.ipynb` once and confirm the DDA ↔ Mie comparison printed at the end is within tolerance.

### A.8 (Optional) LaTeX for rebuilding the theory note

Only required if you want to rebuild `docs/theory_note.pdf`:

```sh
sudo apt install -y texlive-latex-recommended texlive-latex-extra \
                    texlive-science texlive-fonts-recommended latexmk
latexmk -pdf -cd docs/theory_note.tex
```

---

## Part B — Native Linux (Ubuntu)

### B.1 Update Ubuntu and install build tools

```sh
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl ca-certificates
```

### B.2 Install `uv`

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### B.3 Clone block-DDA_Py and create the venv

```sh
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/NobuhiroMoteki/block-DDA_Py.git
cd block-DDA_Py
uv venv --python 3.13
uv pip install -r requirements.txt
```

### B.4 (Optional) LaTeX for rebuilding the theory note

```sh
sudo apt install -y texlive-latex-recommended texlive-latex-extra \
                    texlive-science texlive-fonts-recommended latexmk
latexmk -pdf -cd docs/theory_note.tex
```

### B.5 Smoke test

```sh
uv run python -c "import numpy, scipy, h5py, sklearn; print('OK')"
uv run jupyter notebook test_dda.ipynb
```

---

## Daily workflow

Always invoke commands through `uv run` so the project's `.venv` is used:

```sh
uv run python run_dda.py
uv run python run_dda_spheroid_sweep.py
uv run jupyter notebook
```

To control multi-core FFT parallelism (default = `nproc - 2`):

```sh
DDA_FFT_WORKERS=$(nproc) uv run python run_dda.py
```

---

## Updating dependencies

To pull repository updates and re-sync the venv:

```sh
git pull
uv pip install -r requirements.txt
```

To upgrade an individual package within the pinned set, edit `requirements.txt` and re-run `uv pip install -r requirements.txt`. **Do not** add new dependencies without first confirming with the project owner.

---

## Troubleshooting

### `uv: command not found` after install

Re-source the shell config:

```sh
source ~/.bashrc      # bash
# or
source ~/.zshrc       # zsh
```

### `python: error while loading shared libraries: libpython3.13.so.1.0`

The managed Python from `uv` was removed. Recreate the venv:

```sh
rm -rf .venv
uv venv --python 3.13
uv pip install -r requirements.txt
```

### Jupyter notebook does not open in browser (WSL2)

Forward the port manually. Inside WSL run:

```sh
uv run jupyter notebook --no-browser --port 8888
```

then open `http://localhost:8888/?token=...` (token printed in the terminal) in the Windows browser.

### Slow file I/O when working under `/mnt/c/...`

Move the repository to the Linux home directory (`~/`). WSL2 file I/O across the `/mnt/c` boundary is several times slower than ext4-native I/O.

### Out-of-memory during large sweeps

Peak memory scales as `(1152 + 768 × L) × N_cuboid` bytes. Reduce `L` (orientation count) or `dpl`, or use spheroid mode (`ab_ratio = 1`, `gre_beta = 0`). See the *Performance* section of [README.md](README.md).

### BLAS oversubscription on many-core machines

If you set `DDA_FFT_WORKERS` to all cores **and** BLAS is also using all cores, contention can slow things down. Set:

```sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
DDA_FFT_WORKERS=$(nproc) uv run python run_dda.py
```

---

## Verification checklist

After installation, all of the following should succeed:

- [ ] `uv --version` ≥ 0.5
- [ ] `uv run python --version` returns `Python 3.13.x`
- [ ] `uv run python -c "import numpy, scipy, h5py, sklearn, matplotlib; print('OK')"` prints `OK`
- [ ] `test_dda.ipynb` runs end-to-end and the DDA-vs-Mie comparison is within the tolerance commented in the notebook
- [ ] (Optional) `latexmk -pdf -cd docs/theory_note.tex` rebuilds `docs/theory_note.pdf` without errors

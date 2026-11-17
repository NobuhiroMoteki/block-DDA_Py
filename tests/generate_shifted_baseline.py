"""Generate the Phase-0 reference data for the shifted block-GMRES work.

The reference data are the outputs of the EXISTING code path (independent
block-GMRES solves) for the two cases defined in ``shifted_cases.py``. They are
written to ``.claude/validation/shifted_krylov/baseline_phase0.npz``, which is
outside git.

Usage (from the repository root)::

    .venv/bin/python tests/generate_shifted_baseline.py [--force]

Run it on a commit where the existing solver is known to be valid. The file is
not overwritten unless ``--force`` is given.
"""
import argparse
import datetime
import pathlib
import subprocess
import sys

import numpy as np
import scipy

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import shifted_cases as sc  # noqa: E402  (needs the repository root on sys.path)


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite an existing baseline file")
    args = parser.parse_args()

    if sc.BASELINE_FILE.exists() and not args.force:
        raise SystemExit(f"{sc.BASELINE_FILE} exists; pass --force to overwrite")

    data = sc.collect_baseline()
    data["meta/git_sha"] = np.array(_git("rev-parse", "HEAD"))
    data["meta/git_dirty"] = np.array(bool(_git("status", "--porcelain", "--untracked-files=no")))
    data["meta/created_utc"] = np.array(
        datetime.datetime.now(datetime.timezone.utc).isoformat())
    data["meta/numpy_version"] = np.array(np.__version__)
    data["meta/scipy_version"] = np.array(scipy.__version__)

    sc.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sc.BASELINE_FILE, **data)

    print(f"wrote {sc.BASELINE_FILE}")
    print(f"  git_sha   : {data['meta/git_sha']}  (dirty={bool(data['meta/git_dirty'])})")
    print(f"  spheroid  : lattice_n={data['spheroid/lattice_n']}  N_occ={int(data['spheroid/n_occ'])}")
    for j, (tol, itermax) in enumerate(sc.SPHEROID_SOLVER_SETTINGS):
        print(f"    tol={tol:g} itermax={itermax}: n_iter={data[f'spheroid/tol{j}/n_iter']}")
    print(f"  sphere    : N_occ={int(data['sphere/n_occ'])}  n_iter={int(data['sphere/n_iter'])}")


if __name__ == "__main__":
    main()

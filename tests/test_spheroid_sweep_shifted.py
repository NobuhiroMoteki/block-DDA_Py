"""Acceptance tests of ``run_dda_spheroid_sweep.py --shifted-ri`` (Phase 3).

The sweep script runs at import time and has no command-line control of the
grid. The tests therefore run COPIES of the script as subprocesses, with the
three grid constants replaced by a small grid. The same replacement is applied
to the script of the last commit before the shifted mode was added
(``PRE_CHANGE_COMMIT``), which gives the reference for the default mode.

1. Default mode: every dataset is byte-identical to the pre-change script and
   the config hash is unchanged.
2. ``--shifted-ri``: results equal one existing solve per refractive index on
   the common lattice (relative difference below 1e-8).
3. Resuming a partly finished ``--shifted-ri`` file reproduces it bit for bit.
4. Mixing the two modes in one file, or combining ``--shifted-ri`` with
   ``--delta-n-eff``, is refused.
"""
import os
import pathlib
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from threadpoolctl import threadpool_limits

import shifted_cases as sc
from bl_dda.scatterer import DiscreteDipoles, IncidentField, Target

SCRIPT = sc.REPO_ROOT / "run_dda_spheroid_sweep.py"
PRE_CHANGE_COMMIT = "cb11064"          # main before the shifted mode
PRESET = "liquid512"                   # one wavelength, water host (m_m != 1)
SWEEP_RNG_SEED = 12345                 # RNG_SEED of the sweep script
# The lattices here are tiny, and the geometry builder is several times slower with
# one thread per logical core than with a few. Every run uses the same setting.
N_THREADS = 4

# (original line, reduced-grid line); each must occur exactly once in the script.
GRID_PATCH = (
    ("D_VE_RANGE     = (0.20, 1.20, 50)", "D_VE_RANGE     = (0.20, 0.26, 2)"),
    ("RI_REAL_RANGE  = (1.35, 1.70, 15)", "RI_REAL_RANGE  = (1.35, 1.70, 4)"),
    ("LOG10_AR_RANGE = (0.0, 1.55, 16)", "LOG10_AR_RANGE = (0.0, 0.3, 2)"),
)
DATASETS = ("S_fw_theta_re", "S_fw_theta_im", "S_fw_phi_re", "S_fw_phi_im", "converged")
AXES = ("log_D_ve_grid", "RI_real_grid", "log_AR_grid", "cos_theta_o_half_grid", "phi_o_grid")


def _patched(source):
    for old, new in GRID_PATCH:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


def _run(script, *args):
    """Run a script copy with the repository root on the import path."""
    env = dict(os.environ, PYTHONPATH=str(sc.REPO_ROOT),
               OMP_NUM_THREADS=str(N_THREADS), OPENBLAS_NUM_THREADS=str(N_THREADS),
               DDA_FFT_WORKERS=str(N_THREADS))
    return subprocess.run([sys.executable, str(script), *args], env=env,
                          cwd=script.parent, capture_output=True, text=True)


def _same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("sweep")


@pytest.fixture(scope="module")
def new_script(workdir):
    path = workdir / "sweep_new.py"
    path.write_text(_patched(SCRIPT.read_text()))
    return path


@pytest.fixture(scope="module")
def old_script(workdir):
    try:
        source = subprocess.check_output(
            ["git", "-C", str(sc.REPO_ROOT), "show", f"{PRE_CHANGE_COMMIT}:{SCRIPT.name}"],
            stderr=subprocess.DEVNULL).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"commit {PRE_CHANGE_COMMIT} is not available from git")
    path = workdir / "sweep_old.py"
    path.write_text(_patched(source))
    return path


@pytest.fixture(scope="module")
def default_output(workdir, new_script):
    out = workdir / "default_new.h5"
    proc = _run(new_script, "--preset", PRESET, "--output", str(out))
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


@pytest.fixture(scope="module")
def shifted_output(workdir, new_script):
    out = workdir / "shifted.h5"
    proc = _run(new_script, "--preset", PRESET, "--output", str(out), "--shifted-ri")
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


# ── 1. default mode unchanged ────────────────────────────────────────────────
def test_default_mode_is_byte_identical_to_pre_change_script(workdir, old_script, default_output):
    out_old = workdir / "default_old.h5"
    proc = _run(old_script, "--preset", PRESET, "--output", str(out_old))
    assert proc.returncode == 0, proc.stderr[-2000:]

    with h5py.File(out_old, "r") as old, h5py.File(default_output, "r") as new:
        assert sorted(old.keys()) == sorted(new.keys())
        for axis in AXES:
            assert _same_bytes(old[axis][()], new[axis][()]), axis
        groups = [k for k in old.keys() if k.startswith("wl_")]
        assert groups
        for g in groups:
            assert dict(old[g].attrs) == dict(new[g].attrs)
            assert sorted(old[g].keys()) == sorted(new[g].keys())
            for name in DATASETS:
                assert _same_bytes(old[g][name][()], new[g][name][()]), (g, name)
            assert np.all(new[g]["converged"][()])

        # Root attributes: identical except the version string.
        assert set(old.attrs) == set(new.attrs)
        for key in old.attrs:
            if key != "block_dda_version":
                assert old.attrs[key] == new.attrs[key], key
        assert new.attrs["block_dda_version"] == "0.8.4"
        assert "shifted_ri" not in new.attrs and "lattice_policy" not in new.attrs

        # Provenance: same keys and the SAME config hash.
        assert set(old["provenance"].attrs) == set(new["provenance"].attrs)
        for key in ("producer_repo", "contract_version", "config_hash"):
            assert old["provenance"].attrs[key] == new["provenance"].attrs[key], key
        assert new["provenance"].attrs["producer_version"] == "0.8.4"


# ── 2. shifted mode against per-RI solves on the common lattice ──────────────
def test_shifted_mode_matches_per_ri_solves_on_common_lattice(shifted_output, default_output):
    with h5py.File(shifted_output, "r") as h5:
        assert bool(h5.attrs["shifted_ri"]) is True
        assert h5.attrs["lattice_policy"] == "max_abs_m"
        with h5py.File(default_output, "r") as h5_default:
            assert (h5["provenance"].attrs["config_hash"]
                    != h5_default["provenance"].attrs["config_hash"])
            for axis in AXES:
                assert _same_bytes(h5[axis][()], h5_default[axis][()]), axis

        D_ve_grid = 10.0 ** h5["log_D_ve_grid"][()]
        RI_grid = h5["RI_real_grid"][()]
        AR_grid = 10.0 ** h5["log_AR_grid"][()]
        cos_theta = h5["cos_theta_o_half_grid"][()]
        phi_o = h5["phi_o_grid"][()]
        (group,) = [k for k in h5.keys() if k.startswith("wl_")]
        grp = h5[group]
        wl_0, m_m = float(grp.attrs["wl_0"]), float(grp.attrs["m_m"])
        assert np.all(grp["converged"][()])
        S_theta = grp["S_fw_theta_re"][()] + 1j * grp["S_fw_theta_im"][()]
        S_phi = grp["S_fw_phi_re"][()] + 1j * grp["S_fw_phi_im"][()]

    euler = np.column_stack([np.zeros(cos_theta.size), np.arccos(cos_theta),
                             np.zeros(cos_theta.size)])              # shape: (N_u, 3)
    exp_2p = np.exp(2j * phi_o)                                      # shape: (N_ph,)
    m_max = RI_grid.max() + 0.0j
    worst = 0.0
    slots = [(i, D, j, bc) for i, D in enumerate(D_ve_grid) for j, bc in enumerate(AR_grid)]
    with threadpool_limits(limits=N_THREADS):
        for i_dve, D_ve, i_ar, bc_ratio in slots:
            name, lattice_n, lattice_lf, grid, is_in = sc.quiet(
                sc.build_geometry, D_ve / 2.0, bc_ratio, wl_0,
                np.array([m_max] * 3), SWEEP_RNG_SEED)
            for i_ri, RI in enumerate(RI_grid):
                m_p_xyz = np.array([RI + 0.0j] * 3)
                target = Target(name, lattice_n, lattice_lf, grid, is_in, m_p_xyz, D_ve / 2.0)
                dd = sc.quiet(DiscreteDipoles, target, IncidentField(wl_0, m_m, euler))
                sc.quiet(dd.set_interaction_matrix)
                sc.quiet(dd.solve_matrix_equation)       # sweep defaults: tol=1e-2, itermax=25
                assert dd.converge
                S_s, S_p = dd.compute_PCAS_observable_S_fw()         # shape: (N_u,)
                A, B = (S_s + S_p) / 2, (S_s - S_p) / 2
                ref_theta = A[:, None] + B[:, None] * exp_2p[None, :]  # shape: (N_u, N_ph)
                ref_phi = A[:, None] - B[:, None] * exp_2p[None, :]
                for got, ref in ((S_theta[i_dve, i_ri, i_ar], ref_theta),
                                 (S_phi[i_dve, i_ri, i_ar], ref_phi)):
                    worst = max(worst, np.max(np.abs(got - ref) / np.abs(ref)))
    print(f"max relative difference, shifted vs per-RI solves: {worst:.2e}")
    assert worst < 1e-8


def test_shifted_mode_differs_from_default_only_below_the_top_index(shifted_output,
                                                                    default_output):
    """At the largest RI both modes use the same lattice and agree to rounding.

    Below it the default mode uses a coarser lattice per RI, so the values differ
    at the level of the discretisation error (a few per cent at most here).
    """
    with h5py.File(shifted_output, "r") as a, h5py.File(default_output, "r") as b:
        (group,) = [k for k in a.keys() if k.startswith("wl_")]
        Sa = a[group]["S_fw_theta_re"][()] + 1j * a[group]["S_fw_theta_im"][()]
        Sb = b[group]["S_fw_theta_re"][()] + 1j * b[group]["S_fw_theta_im"][()]
    rel = np.abs(Sa - Sb) / np.abs(Sb)                    # shape: (N_Dve, N_RI, N_AR, N_u, N_ph)
    print(f"shifted vs default mode: top RI {np.max(rel[:, -1]):.2e}, "
          f"lower RI {np.max(rel[:, :-1]):.2e}")
    assert np.max(rel[:, -1]) < 1e-8
    assert 1e-6 < np.max(rel[:, :-1]) < 0.05


# ── 3. resume ────────────────────────────────────────────────────────────────
def test_resume_of_partial_shifted_file_is_bit_identical(workdir, new_script, shifted_output):
    partial = workdir / "shifted_partial.h5"
    shutil.copy(shifted_output, partial)
    with h5py.File(partial, "r+") as h5:
        (group,) = [k for k in h5.keys() if k.startswith("wl_")]
        grp = h5[group]
        # slot (D_ve 0, AR 1): two of four RI missing; slot (D_ve 1, AR 0): all missing
        for idx3 in ((0, 1, 1), (0, 3, 1), (1, 0, 0), (1, 1, 0), (1, 2, 0), (1, 3, 0)):
            grp["converged"][idx3] = False
            for name in DATASETS[:-1]:
                grp[name][idx3] = np.nan

    proc = _run(new_script, "--preset", PRESET, "--output", str(partial), "--shifted-ri")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "(2/4 shifts" in proc.stdout and "(4/4 shifts" in proc.stdout

    with h5py.File(partial, "r") as a, h5py.File(shifted_output, "r") as b:
        for name in DATASETS:
            assert _same_bytes(a[group][name][()], b[group][name][()]), name

    # A finished file is left alone: no slot is solved again.
    proc = _run(new_script, "--preset", PRESET, "--output", str(partial), "--shifted-ri")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "shifts" not in proc.stdout
    with h5py.File(partial, "r") as a, h5py.File(shifted_output, "r") as b:
        for name in DATASETS:
            assert _same_bytes(a[group][name][()], b[group][name][()]), name


# ── 4. refused combinations ──────────────────────────────────────────────────
def test_modes_cannot_be_mixed_in_one_file(workdir, new_script, default_output, shifted_output):
    for source, flags in ((default_output, ["--shifted-ri"]), (shifted_output, [])):
        target = workdir / f"mix_{source.name}"
        shutil.copy(source, target)
        before = target.read_bytes()
        proc = _run(new_script, "--preset", PRESET, "--output", str(target), *flags)
        assert proc.returncode != 0
        assert "--shifted-ri" in proc.stderr and "cannot be resumed" in proc.stderr
        assert target.read_bytes() == before                # file untouched


def test_shifted_mode_rejects_birefringence(workdir, new_script):
    out = workdir / "aniso.h5"
    proc = _run(new_script, "--preset", PRESET, "--output", str(out),
                "--shifted-ri", "--delta-n-eff", "0.05")
    assert proc.returncode != 0
    assert "isotropic" in proc.stderr
    assert not out.exists()

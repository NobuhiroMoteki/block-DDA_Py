"""Acceptance tests of ``run_dda.py --shifted-ri``.

``run_dda.py`` runs at import time on an HDF5 file prepared by
``dda_results/create_h5py.ipynb``. The tests build a small file of the same
layout and run the script as a subprocess. The script of the last release
before the shifted mode (``PRE_CHANGE_REF``) gives the reference for the
default mode.

The sweep has three isotropic non-absorbing refractive indices, one isotropic
absorbing one and one anisotropic one, for four shapes: a spheroid (spheroid
mode), a spheroid with surface roughness, a triaxial ellipsoid and a triaxial
ellipsoid with surface roughness (general mode).

1. Default mode: every dataset is byte-identical to the pre-change script.
2. ``--shifted-ri``: the isotropic entries equal one existing solve per
   refractive index on the common lattice (relative difference below 1e-8).
   The anisotropic entry is byte-identical to the default mode.
3. Resuming a partly finished ``--shifted-ri`` file reproduces it.
4. Mixing the two modes in one file is refused.
"""
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from threadpoolctl import threadpool_limits

import shifted_cases as sc
from bl_dda.scatterer import DiscreteDipoles, IncidentField, Target

SCRIPT = sc.REPO_ROOT / "run_dda.py"
PRE_CHANGE_REF = "v0.8.2"              # release before the shifted mode of run_dda.py
DEFAULT_OUTPUT = "dda_results/pcas_ocbs_simulated_data.hdf5"   # OUTPUT_FILE of the script
SWEEP_RNG_SEED = 12345                 # RNG_SEED of the script
N_THREADS = 4                          # see test_spheroid_sweep_shifted.py

WL_M_M_PAIRS = [[0.638, 1.0]]
M_P_XYZ_LIST = [
    [1.40, 1.40, 1.40],
    [1.55, 1.55, 1.55],
    [1.70, 1.70, 1.70],                 # largest |m| of the isotropic entries
    [1.50 + 0.05j, 1.50 + 0.05j, 1.50 + 0.05j],
    [1.50, 1.50, 1.58],                 # anisotropic
]
ISOTROPIC = [0, 1, 2, 3]
ANISOTROPIC = [4]
I_TOP = 2
R_V_BASE_LIST = [0.12]                 # [um]
BC_RATIO_LIST = [1.5]
AB_RATIO_LIST = [1.0, 1.3]
GRE_BETA_LIST = [0.0, 0.15]
N_ALPHA, N_BETA, N_GAMMA = 2, 3, 2

ORIENTATION_DATASETS = ("C_abs", "C_ext", "S_fw_PCAS_theta", "S_fw_PCAS_phi", "S_bk_OCBS")
MIE_DATASETS = ("C_abs_mie", "C_ext_mie", "S_fw_PCAS_mie", "S_bk_OCBS_mie")
ALL_DATASETS = ("r_ve", "Euler_angles") + ORIENTATION_DATASETS + MIE_DATASETS


def _create_sweep_file(path):
    """HDF5 file with the layout of dda_results/create_h5py.ipynb."""
    num_ori = N_ALPHA * N_BETA * N_GAMMA
    dims = (len(WL_M_M_PAIRS), len(M_P_XYZ_LIST), len(R_V_BASE_LIST), len(BC_RATIO_LIST),
            len(AB_RATIO_LIST), len(GRE_BETA_LIST))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        grp = f.create_group("target")
        grp.attrs["N_alpha_ori"] = N_ALPHA
        grp.attrs["N_beta_ori"] = N_BETA
        grp.attrs["N_gamma_ori"] = N_GAMMA
        grp.attrs["num_orientations"] = num_ori
        grp.create_dataset("wl_m_m_pairs", data=WL_M_M_PAIRS, dtype=np.float64)
        grp.create_dataset("m_p_xyz_list", data=M_P_XYZ_LIST, dtype=np.complex128)
        grp.create_dataset("r_v_base_list", data=R_V_BASE_LIST, dtype=np.float64)
        grp.create_dataset("bc_ratio_list", data=BC_RATIO_LIST, dtype=np.float64)
        grp.create_dataset("ab_ratio_list", data=AB_RATIO_LIST, dtype=np.float64)
        grp.create_dataset("gre_beta_list", data=GRE_BETA_LIST, dtype=np.float64)
        sd = grp.create_group("simulated_data")
        sd.create_dataset("r_ve", dims[2:], dtype=np.float64)
        sd.create_dataset("Euler_angles", dims + (num_ori, 3), dtype=np.float64)
        for name in ("C_abs", "C_ext"):
            sd.create_dataset(name, dims + (num_ori,), dtype=np.float64)
        for name in ("S_fw_PCAS_theta", "S_fw_PCAS_phi", "S_bk_OCBS"):
            sd.create_dataset(name, dims + (num_ori,), dtype=np.complex128)
        for name in ("C_abs_mie", "C_ext_mie"):
            sd.create_dataset(name, dims, dtype=np.float64)
        for name in ("S_fw_PCAS_mie", "S_bk_OCBS_mie"):
            sd.create_dataset(name, dims, dtype=np.complex128)


def _run(script, cwd, *args):
    env = dict(os.environ, PYTHONPATH=str(sc.REPO_ROOT),
               OMP_NUM_THREADS=str(N_THREADS), OPENBLAS_NUM_THREADS=str(N_THREADS),
               DDA_FFT_WORKERS=str(N_THREADS))
    return subprocess.run([sys.executable, str(script), *args], env=env, cwd=cwd,
                          capture_output=True, text=True)


def _same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _data(path):
    with h5py.File(path, "r") as f:
        sd = f["target/simulated_data"]
        return {name: sd[name][()] for name in ALL_DATASETS}


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("run_dda")


@pytest.fixture(scope="module")
def default_output(workdir):
    """Default mode of the current script, run from a directory with the default file name."""
    cwd = workdir / "default_new"
    out = cwd / DEFAULT_OUTPUT
    _create_sweep_file(out)
    proc = _run(SCRIPT, cwd)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


@pytest.fixture(scope="module")
def shifted_output(workdir):
    out = workdir / "shifted.hdf5"
    _create_sweep_file(out)
    proc = _run(SCRIPT, workdir, "--shifted-ri", "--output", str(out))
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


# ── 1. default mode is unchanged ─────────────────────────────────────────────
def test_default_mode_is_byte_identical_to_pre_change_script(workdir, default_output):
    try:
        source = subprocess.check_output(
            ["git", "-C", str(sc.REPO_ROOT), "show", f"{PRE_CHANGE_REF}:{SCRIPT.name}"],
            stderr=subprocess.DEVNULL).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"{PRE_CHANGE_REF} is not available from git")
    cwd = workdir / "default_old"
    out = cwd / DEFAULT_OUTPUT
    _create_sweep_file(out)
    script = cwd / "run_dda_old.py"
    script.write_text(source)
    proc = _run(script, cwd)
    assert proc.returncode == 0, proc.stderr[-2000:]

    old, new = _data(out), _data(default_output)
    for name in ALL_DATASETS:
        assert _same_bytes(old[name], new[name]), name
    assert np.all(np.isfinite(new["C_ext"])) and np.all(new["S_fw_PCAS_mie"].imag != 0)
    with h5py.File(out, "r") as a, h5py.File(default_output, "r") as b:
        assert dict(a["target"].attrs) == dict(b["target"].attrs)
        assert "shifted_ri" not in b["target"].attrs


# ── 2. shifted mode ──────────────────────────────────────────────────────────
def test_shifted_mode_matches_per_index_solves_on_common_lattice(shifted_output):
    with h5py.File(shifted_output, "r") as f:
        assert bool(f["target"].attrs["shifted_ri"]) is True
        assert f["target"].attrs["lattice_policy"] == "max_abs_m_of_isotropic_entries"
    got = _data(shifted_output)
    assert np.all(np.isfinite(got["C_ext"]))

    wl_0, m_m = WL_M_M_PAIRS[0]
    r_v = R_V_BASE_LIST[0]
    bc = BC_RATIO_LIST[0]
    m_top = complex(M_P_XYZ_LIST[I_TOP][0])
    alpha = np.linspace(0, 2 * np.pi, N_ALPHA, endpoint=False)
    beta = np.arccos(np.linspace(1 - 1 / N_BETA, -1 + 1 / N_BETA, N_BETA))
    gamma = np.linspace(0, 2 * np.pi, N_GAMMA, endpoint=False)
    aa, bb, gg = np.meshgrid(alpha, beta, gamma, indexing="ij")
    euler_full = np.column_stack([aa.ravel(), bb.ravel(), gg.ravel()])   # shape: (N_ori, 3)
    euler_beta = np.column_stack([np.zeros(N_BETA), beta, np.zeros(N_BETA)])

    worst = 0.0
    with threadpool_limits(limits=N_THREADS):
        for i_ab, ab in enumerate(AB_RATIO_LIST):
            for i_bt, gre_beta in enumerate(GRE_BETA_LIST):
                spheroid = ab == 1.0 and gre_beta == 0.0
                name, lattice_n, lattice_lf, grid, is_in = sc.quiet(
                    sc.build_geometry, r_v, bc, wl_0, np.array([m_top] * 3), SWEEP_RNG_SEED,
                    ab_ratio=ab, beta=gre_beta)
                for i_mp in ISOTROPIC:
                    m_p_xyz = np.array(M_P_XYZ_LIST[i_mp], dtype=np.complex128)
                    target = Target(name, lattice_n, lattice_lf, grid, is_in, m_p_xyz, r_v)
                    euler = euler_beta if spheroid else euler_full
                    dd = sc.quiet(DiscreteDipoles, target, IncidentField(wl_0, m_m, euler))
                    sc.quiet(dd.set_interaction_matrix)
                    sc.quiet(dd.solve_matrix_equation)   # script defaults: tol=1e-2, itermax=25
                    assert dd.converge
                    S_theta, S_phi = dd.compute_PCAS_observable_S_fw()
                    ref = {"C_abs": dd.compute_C_abs(), "C_ext": dd.compute_C_ext(),
                           "S_fw_PCAS_theta": S_theta, "S_fw_PCAS_phi": S_phi,
                           "S_bk_OCBS": dd.compute_OCBS_observable_S_bk()}
                    for key in ORIENTATION_DATASETS:
                        stored = got[key][0, i_mp, 0, 0, i_ab, i_bt]          # shape: (N_ori,)
                        if spheroid:
                            # alpha = 0, gamma = 0 of the analytically filled grid
                            stored = stored.reshape(N_ALPHA, N_BETA, N_GAMMA)[0, :, 0]
                        if np.all(ref[key] == 0):
                            assert np.all(stored == 0), key
                        else:
                            worst = max(worst, np.max(np.abs(stored - ref[key]) / np.abs(ref[key])))
    print(f"max relative difference, shifted vs per-index solves: {worst:.2e}")
    assert worst < 1e-8


def test_shifted_mode_relation_to_default_mode(shifted_output, default_output):
    a, b = _data(shifted_output), _data(default_output)
    # Shared quantities do not depend on the mode.
    for name in ("r_ve", "Euler_angles") + MIE_DATASETS:
        assert _same_bytes(a[name], b[name]), name
    for name in ORIENTATION_DATASETS:
        # The anisotropic entry is solved in the same way in both modes.
        for i_mp in ANISOTROPIC:
            assert _same_bytes(a[name][:, i_mp], b[name][:, i_mp]), name
        # At the largest isotropic |m| both modes use the same lattice.
        top_a, top_b = a[name][:, I_TOP], b[name][:, I_TOP]
        scale = np.max(np.abs(top_b))
        if scale > 0:
            assert np.max(np.abs(top_a - top_b)) / scale < 1e-8, name
    # Below it the default mode uses a coarser lattice per refractive index.
    lower = [i for i in ISOTROPIC if i != I_TOP]
    rel = np.abs(a["S_fw_PCAS_theta"][:, lower] - b["S_fw_PCAS_theta"][:, lower]) \
        / np.abs(b["S_fw_PCAS_theta"][:, lower])
    print(f"shifted vs default mode, lower refractive indices: {np.max(rel):.2e}")
    assert 1e-8 < np.max(rel) < 0.1


# ── 3. resume ────────────────────────────────────────────────────────────────
def test_resumed_shifted_run_reproduces_the_complete_one(workdir, shifted_output):
    out = workdir / "shifted_resumed.hdf5"
    shutil.copy(shifted_output, out)
    # Mark two isotropic conditions of different shapes as not computed.
    reset = [(0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1)]
    with h5py.File(out, "r+") as f:
        sd = f["target/simulated_data"]
        for idx in reset:
            for name in ORIENTATION_DATASETS:
                sd[name][idx] = 0
            for name in MIE_DATASETS:
                sd[name][idx] = 0
    proc = _run(SCRIPT, workdir, "--shifted-ri", "--output", str(out))
    assert proc.returncode == 0, proc.stderr[-2000:]
    a, b = _data(shifted_output), _data(out)
    for name in MIE_DATASETS + ("r_ve", "Euler_angles"):
        assert _same_bytes(a[name], b[name]), name
    for name in ORIENTATION_DATASETS:
        scale = np.max(np.abs(a[name]))
        if scale > 0:
            assert np.max(np.abs(a[name] - b[name])) / scale < 1e-12, name


# ── 4. the modes cannot be mixed ─────────────────────────────────────────────
def test_modes_cannot_be_mixed(workdir, shifted_output, default_output):
    out = workdir / "mixed_a.hdf5"
    shutil.copy(shifted_output, out)
    before = out.read_bytes()
    proc = _run(SCRIPT, workdir, "--output", str(out))
    assert proc.returncode != 0 and "cannot be resumed" in proc.stderr
    assert out.read_bytes() == before

    out = workdir / "mixed_b.hdf5"
    shutil.copy(default_output, out)
    before = out.read_bytes()
    proc = _run(SCRIPT, workdir, "--shifted-ri", "--output", str(out))
    assert proc.returncode != 0 and "cannot be resumed" in proc.stderr
    assert out.read_bytes() == before

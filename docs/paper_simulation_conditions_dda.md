# Paper Simulation Conditions — block-DDA_Py

Authoritative reference for the DDA-side conditions of the paper-production
sweeps. Mirrors `block-VIEM.jl/docs/paper_simulation_conditions_viem.md` for
direct shape × material × N_DOF comparison between the two solvers.

- **Repository**: `~/Python/block-DDA_Py`
- **Code version**: v0.7.7 (tag pending, pairs with VIEM v0.7.6 `d2bceaf`)
- **Authoring date**: 2026-04-24 (last revised 2026-04-27)
- **Sister project**: [block-VIEM.jl](file:///home/moteki/Julia/block-VIEM.jl) (paper-symmetric)

---

## 1. Software environment

| Item | Value |
|---|---|
| Language | Python 3.13.12 |
| Virtual env | `.venv` (project-local, do not overwrite) |
| Core deps | `numpy 2.4.2`, `scipy 1.17.1`, `h5py 3.16.0` |
| FFT parallelism | `scipy.fft.fftn/ifftn` with multi-threaded batch, default `nproc-2 = 22` workers (env var `DDA_FFT_WORKERS` overrides) |
| BLAS threads | OpenBLAS default (matches `nproc`) |
| Test status | `test_dda.ipynb`, `test_spheroid_symmetry.ipynb` regression clean |

---

## 2. Physical constants

| Quantity | Value | Notes |
|---|---|---|
| Vacuum wavelength λ₀ | **0.638 μm** | hardcoded in [dda_results/paper/_common.py::WL_PAPER](../dda_results/paper/_common.py) |
| Background medium m_m | **1.0** (vacuum) | `M_M_PAPER` |
| Time convention | `exp(+jωt)` (BH83 / block-VIEM.jl compatible) | outgoing wave `exp(-jk₀R)/(4πR)` |

---

## 3. Particle materials (paper "high" 差替済 v0.7.5)

| ID | n_p @ λ=0.638 μm | a_eq list (μm) | Status |
|---|---|---|---|
| `n15` (低屈折率) | 1.5 + 0.01i | 0.05, 0.10, 0.20, 0.40 | active |
| `n20` (paper "high", v0.7.5+) | **2.0 + 0.0i** | 0.05, 0.10, 0.20, 0.40 | active |
| `n317` (legacy high, ≤ v0.7.4) | 3.17 + 0.16i | (reference 用、新規計算には使わない) | retained for legacy comparison |
| `Au` (Johnson & Christy 1972) | **0.17525 + 3.4830i** | 0.05, 0.10, 0.20 | active |

`a_eq` は体積等価球の半径。Au は `|m_p|≈3.49` で波長制約による細かい格子が
必要なため `a_eq ≤ 0.20 μm` に限定（[CLAUDE.md §2](../CLAUDE.md) 参照）。

**v0.7.5 の n317 → n20 差替理由**: DDA `gre × n317 × r_v=0.4 × L=100` で
peak RSS 215 GB / wall 195 min / GMRES iter=100 stagnation を観測。n20
(非吸収) で N_occ が ~4× 縮小し両ソルバとも安定収束見込み。旧 `*_n317.hdf5`
結果は reference として保持（block-VIEM.jl 側と対称）。

---

## 4. Particle shapes

| ID | 形状パラメータ | shape_model entry point |
|---|---|---|
| `sphere` | GRE: `bc_ratio=1, ab_ratio=1, gre_beta=0` | [shape_model/gaussian_ellipsoid.py](../shape_model/gaussian_ellipsoid.py) |
| `oblate` | GRE: `bc_ratio=3, ab_ratio=1, gre_beta=0` | 同上 |
| `gre` | GRE: `bc_ratio=1, ab_ratio=1, gre_beta=0.2` | 同上 |
| `doublet` | 2 球クラスター（後述）| [shape_model/two_sphere_cluster.py](../shape_model/two_sphere_cluster.py) |

### 4.1 形状パラメータの体積等価半径換算

- **球 (sphere)**: 半径 `r = a_eq`
- **扁平回転楕円体 (oblate)**: 3 半軸 (a, b, c) で `a = b = 3c`, `c = a_eq / 9^(1/3)`
- **GRE (Gaussian Random Ellipsoid)**: ベース楕円体 (a=b=c=a_eq) に β_gre=0.2 のガウス変形
  - 形状確定の RNG: `numpy.random.default_rng(12345)`（再現性のため固定）
- **2 球クラスター (doublet)**: 等半径 2 球
  - monomer 半径: `R = a_eq / 2^(1/3) ≈ 0.7937 a_eq`
  - 軸方向 surface-to-surface gap: `g = 0.1 R`
  - **doublet 軸 = 粒子 z 軸**（spheroid mode α-展開公式の前提条件、VIEM と一致）
  - a_eq = 0.05, 0.10, 0.20, 0.40 μm → R ≈ 0.0397, 0.0794, 0.1587, 0.3175 μm

---

## 5. Lattice discretization (auto-dpl, v0.7.6+)

### 5.1 dpl パラメータと体積保存リスケール

DDA の格子間隔は `d = λ₀ / (|m_p|_max · dpl)`、`dpl` = dipoles per wavelength
(粒子内)。格子生成後、[bl_dda/scatterer.py::Target.__init__](../bl_dda/scatterer.py#L20-L26)
で **体積保存リスケール** を適用:

```python
V_target = (4/3)·π·r_v_base³
d_adj = cbrt(V_target / N_occ)
# すべての節点座標を d_adj / lattice_lf 倍にスケール
```

この結果 `N_occ · d_adj³ = V_target` を厳密に満足し、**`r_ve = r_v_base`
(桁落ち精度一致)** が保証される。Green 関数・polarizability・格子間隔
は全て同じ `d_adj` を使うので内部整合性を維持。

### 5.2 auto-dpl (CLAUDE.md §3, v0.7.6+)

**v0.7.6 で固定 dpl=17 を廃止**。各 (shape × material × a_eq) で VIEM の
`n_tet` を target とし、DDA の `n_occ` が **VIEM `n_tet` の 1.5 倍以内
(可能な限り近く)** になるように per-slot で dpl を自動選定する。

実装 [utils/dpl_calibration.py](../utils/dpl_calibration.py):

- `VIEM_N_TET_TABLE`: 12 slot × 3-4 a_eq の VIEM `n_tet` 実測値
  (`julia --project=. viem_results/estimate_cost.jl viem_results/paper/*.hdf5`
  v0.7.6 時点のキャッシュ、2026-04-24 取得)
- 逆算式:

  ```text
  n_occ ≈ V · (|m_p| · dpl / λ₀)³
  ⇒ dpl = (λ₀ / |m_p|) · (n_tet / V)^(1/3)
  ```

- `run_paper_sweep.py` と `estimate_cost.py` が per-slot で自動計算、
  `shape_model.gaussian_ellipsoid.gaussian_ellipsoid_shape_model(dpl=...)`
  または `two_sphere_cluster_shape_model(dpl=...)` に渡す

### 5.3 dpl 実測表 (v0.7.6 calibration)

| shape | material | a_eq 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|
| sphere | n15 | 45.76 | 22.88 | 17.23 | 16.85 |
| sphere | n20 | 34.32 | 17.65 | 17.09 | 16.80 |
| sphere | Au | 19.68 | 17.51 | 16.91 | — |
| oblate | n15 | 92.45 | 46.22 | 23.11 | 16.98 |
| oblate | n20 | 69.34 | 34.67 | 17.32 | 16.79 |
| oblate | Au | 39.76 | 19.88 | 16.85 | — |
| gre | n15 | 142.89 | 71.44 | 35.72 | 17.86 |
| gre | n20 | 107.17 | 53.58 | 26.79 | 16.80 |
| gre | Au | 61.46 | 30.73 | 16.91 | — |
| doublet | n15 | 59.55 | 29.77 | 17.70 | 17.16 |
| doublet | n20 | 44.66 | 22.33 | 17.44 | 17.12 |
| doublet | Au | 25.61 | 17.61 | 17.14 | — |

小 a_eq で dpl が大きく膨らむのは、VIEM 側が `adaptive_lc = min(λ/(|m_p|·N_pw),
c_min/N_per_radius, 0.3·c_min/3)` の**幾何制約** `c_min/3` が支配的な領域で、
DDA もこれに追従する必要があるため。

### 5.4 smoke test 実績 (sphere_n15 auto-dpl, 2026-04-24)

| a_eq | n_tet 目標 | 選択 dpl | DDA N_occ | ratio |
|------|-----------|----------|-----------|-------|
| 0.05 | 652 | 45.76 | 636 | 0.98× |
| 0.10 | 652 | 22.88 | 660 | 1.01× |
| 0.20 | 2229 | 17.23 | 2313 | 1.04× |
| 0.40 | 16671 | 16.85 | 17425 | 1.05× |

全 slot で 1.5× tolerance 余裕クリア。公式の精度は ±5% 以内 (sphere)。
非球形 (oblate, GRE β=0.2, doublet) は表面積効果で ±10-35% 変動するが、
いずれも 1.5× 内で収まることを観測済。

### 5.5 dpl 収束スタディ (§4) での扱い

`scripts/run_dpl_convergence.py` は dpl を明示的に 5 点 [10, 14, 17, 24, 34]
でスイープする設計なので、auto-dpl は**適用しない** — 固定配列を使う。

---

## 6. Orientation grid

### 6.1 多配向 ZYZ Euler グリッド

[scripts/run_paper_sweep.py::_generate_euler_grid](../scripts/run_paper_sweep.py):

| パラメータ | 値 |
|---|---|
| `N_alpha_ori` (α: 0 → 2π) | 4 |
| `N_beta_ori` (β: cos 等分) | 5 |
| `N_gamma_ori` (γ: 0 → 2π) | 5 |
| 公称配向数 | 4 × 5 × 5 = **100** |

α と γ は等間隔開区間 `[0, 2π)`、β は `cos β` を等区間で割って `arccos`
で取得（球面一様サンプル）。VIEM の `generate_euler_grid` とビット単位一致
のループ順 (α 最外 → β 中 → γ 最内)。

### 6.2 Spheroid mode (軸対称粒子の解析展開)

`shape_kind == "doublet"` または `(ab_ratio==1 && gre_beta==0)` のとき有効。
Block-Krylov は `(α=0, γ=0)` の **N_β=5 配向のみ**実 solve、α-expansion で
100 配向に解析展開:

```text
A_fw   = (S_s + S_p) / 2
B_fw   = (S_s - S_p) / 2
S_fw_θ = A_fw + B_fw · exp(2iα)
S_fw_φ = A_fw - B_fw · exp(2iα)
S_bk   = S_bk_0 · exp(2iα)
C_ext, C_abs は α 不変
```

| shape | spheroid_mode | block solver L |
|---|---|---|
| sphere | ✓ | 5 |
| oblate | ✓ | 5 |
| doublet | ✓ (粒子 z 軸対称) | 5 |
| GRE (β=0.2) | ✗ | 100 |

VIEM と同じ式なので、100 配向それぞれで DDA と VIEM が 1:1 比較可能。

---

## 7. Solver settings

### 7.1 本番デフォルト

[scripts/run_paper_sweep.py](../scripts/run_paper_sweep.py):

| 定数 | 値 | 意味 |
|---|---|---|
| ソルバ関数 | `bl_krylov.bl_gmres_mvp_fft` | block-GMRES (v0.7.1+ 既定)、前処理なし |
| 内部実装 | 漸進 block-Givens QR (v0.7.6) | lstsq per-iter (O((kL)³)) → O(kL²+L³)。VIEM の `block_gmres` と同期、L=100 で ~6× 高速化実測 |
| `SOLVER_TOL` | `1.0e-5` | 相対残差 ‖B − A·X‖_F / ‖B‖_F ≤ TOL で converged |
| `MAXITER` | **200** | v0.7.6: 100 → 200 (n20 では通常 ~30-50 iter で収束、Au stagnation や large L に headroom) |
| `RNG_SEED` | 12345 | GRE 形状 RNG (block-VIEM.jl と同一 seed) |
| FFT-MVP | Goodman 1991, `scipy.fft.fftn/ifftn` | doubled grid `(2Nx, 2Ny, 2Nz, 3, 3)`、multi-threaded batch |
| polarizability | Chaumet & Rahmani 2009 (CR2009) | `α_CR = α₀ / (1 − M(1−jka)·exp(jka)/V)` |

### 7.2 Block-BiCGSTAB との比較について

v0.7.6 で paper のスコープから **block-BiCGSTAB を除外**。論文は shape ×
material × N_DOF スケーリングに焦点を絞る。`bl_krylov.bl_bicgstab_mvp_fft`
関数自体は legacy fallback として保持、`scripts/run_rhs_scaling.py::METHODS`
から除外済。復活させたい場合は METHODS に `("bicgstab", bl_bicgstab_mvp_fft)`
を再追加すれば schema 変更なしで動作。

### 7.3 バックワード互換 solver

[bl_dda/scatterer.py::DiscreteDipoles.solve_matrix_equation](../bl_dda/scatterer.py)
は依然として `bl_bicgstab_jacobi_mvp_fft` (Jacobi 前処理付き) を呼ぶ。
これは `test_dda.ipynb` / 既存 `run_dda.py` との後方互換のため。**paper 用
ランナーは `bl_gmres_mvp_fft` を直接呼ぶ** ので legacy path には一切
依存しない。

---

## 8. Output observables

[dda_results/paper/_common.py::create_paper_h5](../dda_results/paper/_common.py)
の HDF5 schema。全データセット `/target/simulated_data/` 直下、shape は
配向数 100 を末尾に持つ。

### 8.1 物理観測量

| dataset | dtype | shape | 単位 | 定義 |
|---|---|---|---|---|
| `C_ext` | float64 | (1,1,N_rv,1,1,1,100) | μm² | 消散断面積 (per orientation) |
| `C_abs` | float64 | 同上 | μm² | 吸収断面積 |
| `S_fw_PCAS_theta` | complex128 | 同上 | μm | 前方散乱振幅 θ 成分 = `S11(0) + i·S12(0)` |
| `S_fw_PCAS_phi` | complex128 | 同上 | μm | 前方散乱振幅 φ 成分 = `S22(0) − i·S21(0)` |
| `S_bk_OCBS` | complex128 | 同上 | μm | 後方散乱振幅 = `(−S11+S22−i·S12−i·S21)(180°)/√2` (theory_note.tex §S-bk; `(−S_bk_θ + S_bk_φ)/√2` with `S_bk_θ = S11+i·S12`, `S_bk_φ = S22−i·S21` at θ=π) |
| `Euler_angles` | float64 | (...,100,3) | rad | (α, β, γ) per orientation |
| `r_ve` | float64 | (N_rv,1,1,1) | μm | 体積保存リスケール後の体積等価半径 (= r_v_base 厳密) |

### 8.2 体積等価球 Mie 参照値

| dataset | dtype | shape |
|---|---|---|
| `C_ext_mie`, `C_abs_mie` | float64 | (1,1,N_rv,1,1,1) |
| `S_fw_PCAS_mie` | complex128 | 同上 |
| `S_bk_OCBS_mie` | complex128 | 同上 |

doublet 形状でも書き込まれるが、Rayleigh 領域以外では参考値（厳密解は
MSTM、VIEM 側から参照、§11 参照）。

### 8.3 規格化光学断面積（解析時に算出）

論文 Figure では `Q_X = C_X / (π · a_eq²)` を使用。`a_eq` は HDF5 の
`r_v_base_list`（目標値）または `r_ve`（実効値）を使う。DDA では
体積保存リスケールにより `r_ve == r_v_base` が厳密に成立するので両者同値。

### 8.4 偏光・入射状態

`/target` group attrs:
- `light_source`: 形状ファイル毎にハードコード（例: "(paper) λ=0.638 μm, n=2.0+0.0i, sphere"）
- `polarization_state`: "left-handed circular: E0_theta=1/sqrt(2), E0_phi=1j/sqrt(2)"
- `S_definition`: BH83 → MI02 → CAS-v2 変換式（Mishchenko 2000 準拠）

block-VIEM.jl の `create_paper_h5` と byte 互換。

---

## 9. /target/cost/ — per-slot cost & solver diagnostics (v0.7.5+)

| dataset | dtype | shape | 意味 |
|---|---|---|---|
| `t_build_s` | float64 | shape_cond | 形状 lattice 生成時間 [s] |
| `t_setup_s` | float64 | shape_cond | polarizability + FFT-init 時間 [s] |
| `t_solve_s` | float64 | shape_cond | block-Krylov solve 時間 [s] |
| `t_total_s` | float64 | shape_cond | end-to-end (build + setup + solve + observables + Mie + HDF5) [s] |
| `peak_rss_bytes` | int64 | shape_cond | per-slot peak RSS [bytes] (`/proc/self/status:VmRSS` 0.2s sampler) |
| `n_cuboid` | int64 | shape_cond | 格子 Nx·Ny·Nz (VIEM `n_tet` に対応) |
| `n_occ` | int64 | shape_cond | 占有 dipole 数 (VIEM `n_dof` に対応、matrix DOF = 3·n_occ) |
| `lattice_lf` | float64 | shape_cond | 格子間隔 d_adj [μm] (VIEM `mean_edge_length` に対応) |
| `iters` | int64 | shape_cond | block-Krylov 反復数 |
| `converged` | int8 | shape_cond | 1 / 0 |
| `solver_err` | float64 | shape_cond | 最終相対残差 |
| `residual_history` | float64 | shape_cond × MAXITER (=200, v0.7.6+) | per-iter 残差、`NaN` パディング |

`shape_cond = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt)`。
VIEM 側 `/target/cost/` と 8 fields (時間 + peak RSS + iters + converged +
solver_err + residual_history) は **bit-for-bit 一致**。残り 3 つは DDA
名称 (`n_cuboid`/`n_occ`/`lattice_lf`)、VIEM 側 (`n_tet`/`n_dof`/
`mean_edge_length`) との物理的 1:1 対応。

### 9.1 RSS sampler

[utils/rss_monitor.py](../utils/rss_monitor.py):
- daemon `threading.Thread` で `/proc/self/status:VmRSS` を 0.2s 間隔ポーリング
- `RSSMonitor.reset()` で per-slot ベースラインリセット
- Linux 専用 (macOS / WSL でも `/proc` がある環境なら動作)

---

## 10. Logging

`run_paper_sweep.py` の `_log(msg)` は HH:MM:SS タイムスタンプ付きで
stdout に出力:

```text
[09:35:01] wl_0=0.6380  m_m=1.000  m_p_xyz=[2.0+0.0j 2.0+0.0j 2.0+0.0j]  r_v=0.100  bc=1.0  ab=1.0  β=0.00  [sphere/n20] dpl=17.65  d=0.01808μm  N_cub=1728  N_occ=721  (target n_tet=710, ratio=1.02×) [spheroid, L_solve=5]
[09:35:02]   solver ✓ iters=8 err=5.3e-06  t_build=0.3s t_setup=0.0s t_solve=0.1s t_total=0.4s  peak_RSS=0.42GB
```

stdout は `run_in_background` で捕捉 or 手動 shell で観察。`h5.flush()` が
per-slot 完了時に呼ばれるので、長時間 slot 中でも外部から cost diagnostics
の一部は読める（HDF5 lock 注意）。

---

## 11. Runners (entry points)

### 11.1 1 スイープファイル = 1 (shape × material) の生成

[dda_results/paper/](../dda_results/paper/) 配下に 12 個 (v0.7.6 active)
+ 1 pilot + 5 legacy n317 reference:

```text
sphere_n15.py   sphere_n20.py   sphere_Au.py
oblate_n15.py   oblate_n20.py   oblate_Au.py
gre_n15.py      gre_n20.py      gre_Au.py
doublet_n15.py  doublet_n20.py  doublet_Au.py
pilot_sphere_n20.py  pilot_doublet.py
```

各ファイルは [_common.py::create_paper_h5](../dda_results/paper/_common.py)
を呼び、`dda_results/paper/{shape}_{material}.hdf5` を空テンプレート
（schema のみ）で生成。

```bash
PYTHONPATH=. .venv/bin/python -m dda_results.paper.sphere_n20
```

### 11.2 本番計算

```bash
PYTHONPATH=. .venv/bin/python scripts/run_paper_sweep.py dda_results/paper/sphere_n20.hdf5
```

Resume: `S_fw_PCAS_mie` の imag が非ゼロのスロットはスキップ
(block-VIEM.jl と同方針)。

### 11.3 事前見積

```bash
PYTHONPATH=. .venv/bin/python scripts/estimate_cost.py dda_results/paper/sphere_n20.hdf5
```

各 shape slot で worst-case (wl_min, |m_p|_max) の auto-dpl から格子を
実構築 → N_occ 取得 → empirical scaling で RSS / setup / solve 時間を
見積。**24h 超 or RSS > 90% MemAvailable** で `⚠️` フラグ。

### 11.4 dpl 収束スタディ (a_eq=0.1 μm 固定、5 dpl)

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dpl_convergence.py <shape> <material>
# shape    ∈ {sphere, oblate, gre}
# material ∈ {n15, n20, Au}  (n317 legacy も accept)
```

出力: `dda_results/paper/convergence_{shape}_{material}.hdf5`、
`/target/dpl_convergence/` 直下に `dpl`, `lattice_lf`, `n_cuboid`, `n_occ`,
`r_ve`, `iters`, `converged`, `t_setup`, `t_solve`, `t_total`,
`peak_rss_bytes`, `Q_*`, `S_*`, **residual_history** (shape
`(n_dpl_points, MAXITER=200)`) などを記録。VIEM の lc convergence factor
`[1.5, 1.0, 0.7, 0.5, 0.35]` に対応する **dpl = [10, 14, 17, 24, 34]**
(adaptive_lc 中央 = dpl=17) の 5 点。

### 11.5 RHS-scaling 診断 (block-Krylov の L 依存性)

```bash
PYTHONPATH=. .venv/bin/python scripts/run_rhs_scaling.py dda_results/paper/sphere_n20.hdf5
```

`L = 1, 2, 4, 8, 16, 32, 64, 128` の 8 点を `bl_gmres_mvp_fft` のみで測定
(v0.7.6、BiCGSTAB は scope 外)。出力先: 入力 HDF5 の
`/target/rhs_scaling/gmres/` サブグループ。配向は
`np.random.default_rng(12345)` の uniform-sphere、L=1 ⊂ ⋯ ⊂ L=128 で nested。

**RHS-scaling 対象は sphere × {n15, n20, Au} のみ** (2026-04-22 決定、
oblate/gre/doublet は scaling 挙動を問わない)。`L_LIST` は VIEM 側の
RSS 実測を見てから最終確定する (L=128 × 重い slot で OOM リスクあり)。

### 11.6 MSTM 厳密解 (doublet のみ)

DDA 側では MSTM を直接計算しない。VIEM 側が生成する
`~/Julia/block-VIEM.jl/viem_results/paper/mstm_doublet_{material}.hdf5`
を paper plot notebook (`plot_paper_results.ipynb`) で読み込む。
`truncation_order = 15` 固定で全材料完全収束
([block-VIEM.jl/viem_results/paper/run_mstm_reference.jl](file:///home/moteki/Julia/block-VIEM.jl/viem_results/paper/run_mstm_reference.jl))。

---

## 12. Resume / Escalation policy

[CLAUDE.md §5](../CLAUDE.md):

1. **Resume**: 既存 HDF5 の `S_fw_PCAS_mie` が non-zero (imag != 0) なら
   該当スロットスキップ。`run_paper_sweep.py` 自動。
2. **Escalation 閾値**: `estimate_cost.py` で 1 slot あたり推定 **wall
   time > 24 h** または推定 **peak RSS > 90% MemAvailable** で `⚠️` フラグ
   → 起動前に必ずユーザー確認。
3. **Solver failure**: `try` / `except KeyboardInterrupt` で中断は許容、
   それ以外は NaN 埋めして次スロットへ続行
   (`converged=0` を /target/cost/ に記録)。

---

## 13. File naming conventions

```text
dda_results/paper/
├── _common.py                               # schema helper, material constants
├── {shape}_{material}.py                    # 11 active sweep generators + 5 legacy n317
├── {shape}_{material}.hdf5                  # production results
├── pilot_sphere_n20.py                      # smoke-test wrapper (a_eq=0.1 only)
├── pilot_doublet.py                         # smoke-test wrapper for doublet
├── convergence_{shape}_{material}.hdf5      # dpl convergence study output
├── cost_estimates.md                        # pre-run estimates + actual results log
└── (mstm_doublet_{material}.hdf5 は VIEM 側から参照)

scripts/
├── run_paper_sweep.py                       # 本番スイープ runner (GMRES, auto-dpl)
├── run_dpl_convergence.py                   # dpl 収束スタディ runner
├── run_rhs_scaling.py                       # RHS-scaling L 依存性 runner
└── estimate_cost.py                         # 事前見積ツール

utils/
├── material.py                              # paper material constants
├── rss_monitor.py                           # /proc/self/status:VmRSS sampler
└── dpl_calibration.py                       # auto-dpl 逆算 (VIEM n_tet → dpl)

shape_model/
├── gaussian_ellipsoid.py                    # GRE (sphere/oblate/gre)
└── two_sphere_cluster.py                    # doublet
```

---

## 14. Cross-reference: block-VIEM.jl side

VIEM 側との対称項目 (要同期):

| 項目 | DDA | VIEM |
|---|---|---|
| 材料定数 | `_common.py::N_LOW/N_20/N_HIGH/N_AU` | `_common.jl::N_LOW/N_20/N_HIGH/N_AU` |
| 形状パラメータ (`R = a_eq/2^(1/3)`, gap=0.1R) | `shape_model/two_sphere_cluster.py` | `run_viem.jl::_doublet_along_z` |
| HDF5 schema (`/target/cost/` etc.) | `_common.py::create_paper_h5` | `_common.jl::create_paper_h5` |
| RNG seed (12345) | `run_paper_sweep.py::RNG_SEED` | 同上 |
| MAXITER (=200, v0.7.6) | `scripts/run_*.py::MAXITER` | `viem_results/run_*.jl::MAXITER` |
| residual_history (length 200, NaN-pad) | `bl_*_mvp_fft` returns + per-slot HDF5 | `BlockSolveResult.residual_history` |
| Incremental block-Givens QR (v0.7.6) | `bl_gmres_mvp_fft` | `block_gmres` |
| RHS-scaling solver scope (GMRES のみ, v0.7.6) | `scripts/run_rhs_scaling.py::METHODS` | `run_rhs_scaling.jl::METHODS` |
| **体積保存リスケール** | `Target.__init__` で `d_adj = cbrt(V_target/N_occ)` 厳密化 | **VIEM 側 v0.7.7 候補** (2026-04-24 ユーザー決定、統合 prompt 送付済) |
| auto-dpl (n_occ ≈ n_tet × 1.5 以内) | `utils/dpl_calibration.py` (VIEM_N_TET_TABLE) | N/A (VIEM は `N_pw=10` 固定 + adaptive_lc) |

---

## 15. Version history (paper-relevant)

| Version | Commit | 主な変更 |
|---|---|---|
| v0.7.0 | (historical) | paper-production scaffolding, block-Krylov solvers |
| v0.7.3 | `38aeaafc` | spheroid sweep grid ranges, `.h5` gitignore |
| v0.7.4 | `dbadc075` | interactive notebook for spheroid sweep |
| v0.7.5 | `0b73abd7` | paper-production infrastructure + DDA↔VIEM cost parity (`/target/cost/` group), `bl_gmres_mvp_fft` / `bl_bicgstab_mvp_fft` 前処理なし変種追加 |
| v0.7.6 | `190ab55f` | material n317 → n20 差替, residual_history (/target/cost/), incremental block-Givens QR (GMRES O((kL)³) → O(kL²+L³), L=100 で 6× 高速化), MAXITER 200, BiCGSTAB scope 外, auto-dpl (VIEM n_tet に 1.5× 以内で per-slot dpl 調整) |
| **v0.7.7** | (pending) | **paper-production sweep 全 run 完了** (12 §11.2 + 9 §11.4 + 3 §11.5、HDF5 を repo 同梱)、**Au material_label の float64 精度修正** (4→5 桁、`0.17524999...` 一致対応)、**run_dpl_convergence の n20 サポート追加**、**run_rhs_scaling の `L > 3·N_occ` ガード**（degenerate QR 回避）、**plot_paper_results.ipynb スケルトン追加** (Q_X / CAS-v2 振幅 / DDA↔VIEM 誤差 / dpl-lc 収束 / RHS-scaling / cost summary) |

---

## 16. References

- [`CLAUDE.md`](../CLAUDE.md) (本リポジトリ): 論文計算フェーズの運用ガイド
- [`README.md`](../README.md): 公開向けプロジェクト解説、benchmark 記録
- [`docs/theory_note.tex`](theory_note.tex): DDA 理論ノート (formal derivations)
- block-VIEM.jl: `~/Julia/block-VIEM.jl` (姉妹プロジェクト、`paper_simulation_conditions_viem.md` authoritative)
- MSTMforCAS.jl: `~/Julia/MSTMforCAS.jl` (doublet 厳密解、VIEM 経由で共有)

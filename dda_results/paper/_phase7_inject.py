"""Canonical source for Section 7 markdown + Phase 7 code cell of
plot_paper_results.ipynb (fig 4 = VIEM ℓ_c convergence on non-sphere
production shapes).

This file is the **source of truth** for the Phase 7 cell. The notebook
itself contains a copy that is injected from here. For non-trivial
Phase 7 edits (logic, observable selection, layout) prefer editing
MD_SOURCE / CODE_SOURCE in this file, then run

    python _phase7_inject.py
    jupyter nbconvert --to notebook --execute --inplace \\
            plot_paper_results.ipynb

to re-inject and re-run.

Trivial visual tweaks (axis labels, marker sizes, legend text) can also
be made directly in the .ipynb via Jupyter / VS Code; this file then
becomes stale until the user copies the notebook's Phase 7 source back
into CODE_SOURCE here. Either workflow is valid — this file is the
recommended starting point for substantive changes.

Idempotent: if cells with id 'section-fig4'/'fig4-conv' already exist,
their source is replaced in-place; otherwise they are appended at the
end of the notebook.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent / 'plot_paper_results.ipynb'

MD_ID    = 'section-fig4'
CODE_ID  = 'fig4-conv'

MD_SOURCE = """## Section 7 — VIEM $\\ell_c$ self-convergence on non-sphere production shapes (oblate + GRE)

VIEM-only extension of Section 4. Same single-orientation convergence sweep
(`viem_results/paper/run_lc_convergence.jl`, ZYZ identity orientation
$\\alpha=\\beta=\\gamma=0$, $r_{\\rm ve}=0.1\\,\\mu$m) but on the **non-sphere**
target shapes — oblate spheroid ($a/b=1$, $b/c=3$) and GRE
($\\beta_{\\rm gre}=0.2$).

Two reference choices are overlaid where data permits:

**TMM reference at $\\beta=0$** (oblate only): the convergence-sweep
orientation $(\\alpha,\\beta,\\gamma)=(0,0,0)$ is now covered by a
dedicated EBCM (T-matrix) computation, `tmm_oblate_conv_<mat>.hdf5`,
produced by
[`run_tmatrix_oblate_conv_reference.jl`](../../../../Julia/block-VIEM.jl/viem_results/paper/run_tmatrix_oblate_conv_reference.jl).
The pre-pulled production `tmm_oblate_*.hdf5` files are kept untouched —
they cover the production $\\beta$ grid (cos $\\beta\\in\\{\\pm 0.8,\\pm 0.4,0\\}$)
needed by Section 3 / fig 3. ε vs TMM uses filled circles ●, solid line.

**Richardson extrapolation** (all panels): 2-point Richardson at the
two finest meshes assuming the SWG h² rate, $X_\\infty^{\\rm Rich} =
(r^p y_{\\rm fine} - y_{\\rm 2nd})/(r^p-1)$ with $r^p =
(n_{\\rm fine}/n_{\\rm 2nd})^{p/3}$, $p=2$. ε vs Richardson uses open
squares □, dashed line. This is the only available reference for GRE
(no exact analytical solution).

The two curves on oblate panels should track each other to within
$\\mathcal O(h^4)$ + solver residual; their visual co-linearity validates
the Richardson approach used as the sole reference on the GRE row.

**Single observable: $|S_{\\rm fw}^\\theta|$**. CAS-v2 polarimetric
forward amplitude — the headline observable that the downstream
Bayesian retrieval consumes directly. Convergence universality across
multiple observables is already covered by fig 1 (sphere, Mie ref);
fig 4 isolates the most paper-relevant scalar to maximise the visual
punch of the dual TMM ⟷ Richardson overlay. Galerkin theory predicts
the same $h^2$ rate for every bounded linear functional of the SWG
solution, so the shape-universality argument needs only one
representative observable.

$|S_{\\rm bk}|$ is **excluded** specifically because at the
convergence-sweep slot (axisymmetric oblate at axial incidence
$\\beta=0$) TMM gives $|S_{\\rm bk}|=0$: $S_1(180°)=-S_2(180°)$ +
off-diagonals vanish ⇒ the CAS-v2 LCP-projected back-amplitude
vanishes by construction. This is **not a general TMM property**:
$|S_{\\rm bk}|\\ne 0$ at $\\beta\\ne 0$ for axisymmetric shapes and at
any orientation for non-axisymmetric shapes (GRE). The
relative-error denominator $|S_{\\rm bk}^{\\rm TMM}|=0$ at our
specific $\\beta=0$ slot makes $\\varepsilon$ undefined; we exclude
rather than special-case.

$Q_{\\rm ext}$ and $Q_{\\rm abs}$ are excluded as redundant with fig 1
(Q_abs additionally noisy on n15 due to small Im $m_p$).

- **Rows**: oblate (3:3:1), GRE ($\\beta_{\\rm gre}=0.2$).
- **Cols**: n15, n20, Au.
- **Reference pair (Richardson)**: $(\\ell_c^{\\rm factor}=0.5, 0.35)$ for oblate (5 points), $(1.0, 0.7)$ for GRE (3 points).
- Stalled $\\ell_c$ points (MAXITER=200) → `×` marker.

**Au residual-threshold gate (paper Option B).** For Au every lc point
exits at MAXITER=200 with non-zero $\\|r\\|/\\|b\\|$. Both the TMM
overlay and the Richardson curve are kept only if $\\|r\\|/\\|b\\|$ at
the two finest meshes is below $10^{-3}$ — a 100× relaxation of the
convergence criterion bounding the observable bias to $\\sim 10^{-2}$
in the plasmonic regime ($\\kappa(A)\\approx 10$ empirically). Series
failing the gate are suppressed and the panel is annotated with the
residual sequence. Per-series result:

| Series | $\\|r\\|/\\|b\\|$ at finest pair | gate $10^{-3}$ | fig 4 action |
| --- | ---: | :---: | --- |
| oblate × Au | $6.4\\times 10^{-4},\\ 7.8\\times 10^{-4}$ | ✅ | TMM + Richardson kept (×) |
| gre × Au    | $5.5\\times 10^{-2},\\ 3.9\\times 10^{-2}$ | ❌ | **suppressed**, panel annotated |

Detailed derivation in [`fig4_description.md` §6.3](fig4_description.md).
"""

CODE_SOURCE = '''"""Phase 7 — VIEM ℓ_c convergence on non-sphere production shapes.

fig4_lc_convergence_nonsphere.{png,pdf} — 2 × 3 panels:
  Rows: oblate (3:3:1), GRE (β_gre=0.2)
  Cols: n15 (m_p=1.5+0.01i), n20 (m_p=2.0), Au (m_p=0.18+3.48i)

Y-axis: relative error |X(ℓ_c) − X_ref| / |X_ref|, plotted against
n_tet on log-log axes. Two reference choices are overlaid where data
permits, with distinct marker / linestyle:

  TMM (oblate only):  X_ref = X_TMM at (a_eq=0.1, β=0)  — exact axisymmetric
    EBCM result from `tmm_oblate_conv_<mat>.hdf5` (run_tmatrix_oblate_conv_
    reference.jl). Filled circle ●, solid line.
  Richardson (all):   X_ref = X_∞^Rich, 2-point Richardson extrapolation
    of the two finest VIEM meshes assuming the SWG h^2 rate:
        X_∞^Rich = (r^p · y_fine − y_2nd) / (r^p − 1),
        r^p = (n_fine / n_2nd)^(p/3),  p = 2.
    Open square □, dashed line.

The TMM and Richardson curves should overlap at the production lc range,
giving a direct cross-check of the Richardson-only treatment used on the
GRE row (where no exact reference exists).

Single observable: |S_fw_θ| — CAS-v2 polarimetric forward amplitude,
the headline scalar of the paper. fig 1 already established convergence
universality across {Q_ext, Q_abs, |S_fw_θ|, |S_bk|} on sphere; the
remaining task for fig 4 is to demonstrate **shape**-universality of
the same h^2 rate, which one observable suffices for (Galerkin theory:
all bounded linear functionals share the same rate).

|S_bk| is excluded for a slot-specific reason: at axisymmetric oblate
+ axial incidence (β=0), TMM gives |S_bk|=0 by construction
(S₁(180°)=−S₂(180°) + off-diagonals zero ⇒ LCP back-amplitude
vanishes). |S_bk| is generally non-zero in TMM for β≠0 or non-axi-
symmetric shapes; the zero is specific to our convergence slot.
Relative-error denominator = 0 → ε undefined → excluded.

Q_ext, Q_abs are excluded as redundant with fig 1 (Q_abs is also
noisy on n15 due to small Im m_p).

Slope vs n_tet recovers the SWG discretization order p = 2 ⇒ slope −2/3.
The slope guide is anchored at the TMM curve's 2nd-finest point on
oblate panels (or the Richardson curve's 2nd-finest on GRE panels).

Residual-threshold gate (paper-policy Option B). Stalled-residual
contamination of X_∞^Rich is acceptable only when ‖r‖/‖b‖ at both
finest meshes is below PHASE7_RESIDUAL_THRESHOLD = 1e-3 (justification
in fig4_description.md §6.3). Series failing the gate are not plotted;
the panel shows the final residuals as a textual annotation. Au series:
oblate Au passes the gate (6.4e-4, 7.8e-4 — both Richardson and TMM
overlays kept); GRE Au fails (5.5e-2, 3.9e-2 — both suppressed).
"""

PHASE7_RESIDUAL_THRESHOLD = 1.0e-3
PHASE7_SHAPES = ['oblate', 'gre']

# Single observable for fig 4 (paper editorial decision): |S_fw_θ| is
# the polarimetric forward amplitude that the CAS-v2 retrieval consumes
# directly — the headline observable of this paper. fig 1 already
# established convergence universality across {Q_ext, Q_abs, |S_fw_θ|,
# |S_bk|} on sphere via the Galerkin theory of linear functionals on
# the SWG basis (rate is shared by all bounded linear functionals); fig
# 4 therefore needs only one observable to demonstrate the **shape**
# universality of the same h^2 rate on non-sphere targets.
#
# |S_bk| is excluded because TMM gives |S_bk|=0 specifically for an
# axisymmetric particle at axial incidence (β=0): S₁(180°)=−S₂(180°)
# and off-diagonals vanish, so the CAS-v2 LCP-projected back-amplitude
# vanishes by construction. (TMM |S_bk| is generally non-zero — for
# axisymmetric particles at β≠0, or any orientation of non-axisymmetric
# particles like GRE.) The relative-error denominator |S_bk^TMM|=0 in
# our convergence-sweep slot makes ε undefined; we exclude rather than
# special-case.
#
# Q_abs is excluded to avoid small-Im(m_p) numerical noise on n15;
# Q_ext is excluded as redundant with fig 1 and to maximise the visual
# punch of the dual TMM ⟷ Richardson overlay on a single observable.
PHASE7_OBS_FIG4 = [
    ('S_fw_theta', r'$|S_{\\rm fw}^{\\theta}|$', 'mag',  'C2'),
]

# TMM β=0 reference for oblate panels — pre-load once.
PHASE7_CONV_REF = {}
for _mat in MATERIALS:
    _p = VIEM_DIR / f'tmm_oblate_conv_{_mat}.hdf5'
    if _p.is_file():
        _ref = _try(load_tmm, _p)
        if _ref is not None:
            PHASE7_CONV_REF[('oblate', _mat)] = _ref
print(f'Phase 7: loaded {len(PHASE7_CONV_REF)} oblate TMM β=0 reference files')


def _phase7_richardson_x_inf(y, x_n, p=2.0):
    """Two-point Richardson X_∞ from the two finest meshes (h ∝ n_tet^{-1/3}).
    Returns float or None when ill-conditioned."""
    if y is None:
        return None
    y   = np.asarray(y,   dtype=float)
    x_n = np.asarray(x_n, dtype=float)
    if len(y) < 2 or len(x_n) != len(y):
        return None
    order  = np.argsort(x_n)
    i_fine, i_2nd = order[-1], order[-2]
    if x_n[i_fine] <= 0 or x_n[i_2nd] <= 0 or x_n[i_fine] == x_n[i_2nd]:
        return None
    r_p = (x_n[i_fine] / x_n[i_2nd]) ** (p / 3.0)
    if abs(r_p - 1.0) < 1e-9:
        return None
    X_inf = (r_p * y[i_fine] - y[i_2nd]) / (r_p - 1.0)
    return X_inf if np.isfinite(X_inf) and X_inf != 0 else None


def _phase7_relerr_against(y, x_inf):
    """ε_i = |y_i − X_inf| / |X_inf| for every i; None if X_inf is invalid."""
    if x_inf is None or not np.isfinite(x_inf) or x_inf == 0:
        return None
    return np.abs(np.asarray(y, dtype=float) - x_inf) / abs(x_inf)


def _phase7_kind_to_scalar(v, kind):
    """Map a (possibly complex) reference value to its plot-axis scalar."""
    if v is None:
        return None
    if kind == 'real':
        return float(np.real(v))
    if kind == 'mag':
        return float(np.abs(v))
    if kind == 'imag':
        return float(np.imag(v))
    return None


def plot_phase7(save=True):
    fig, axes = plt.subplots(len(PHASE7_SHAPES), len(MATERIALS),
                              figsize=(11.5, 6.5),
                              sharex=True, sharey=True)
    fig.suptitle(r'VIEM $\\ell_c$ convergence on non-sphere production shapes '
                  r'at $r_{\\rm ve}=0.1\\,\\mu$m, single orientation '
                  r'($\\alpha=\\beta=\\gamma=0$): '
                  r'$|X(\\ell_c)-X_{\\rm ref}|/|X_{\\rm ref}|$',
                  y=1.00, fontsize=10)

    for i, shape in enumerate(PHASE7_SHAPES):
        for j, mat in enumerate(MATERIALS):
            ax = axes[i, j]
            c = CONV.get((shape, mat), {})
            viem = c.get('viem')
            if (viem is None
                    or viem.get('n_cuboid') is None
                    or len(viem['n_cuboid']) < 2):
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                         ha='center', va='center', fontsize=9, color='gray')
                if i == 0:
                    ax.set_title(MATERIAL_LABEL[mat])
                if j == 0:
                    ax.set_ylabel(SHAPE_LABEL[shape])
                continue

            x_n  = np.asarray(viem['n_cuboid'], dtype=float)   # = n_tet
            n_lc = len(x_n)
            converged = np.asarray(
                viem.get('converged', np.ones(n_lc)), dtype=bool)
            solver_err = viem.get('solver_err')
            if solver_err is not None:
                solver_err = np.asarray(solver_err, dtype=float)

            order  = np.argsort(x_n)
            i_2nd  = order[-2] if n_lc >= 2 else order[-1]
            i_fine = order[-1]

            # Residual-threshold gate (paper Option B). Au cases:
            # oblate × Au passes (residuals ~6e-4); GRE × Au fails
            # (residuals 4e-2 to 6e-2 — plot suppressed).
            if (solver_err is not None
                    and (solver_err[i_fine] > PHASE7_RESIDUAL_THRESHOLD
                         or solver_err[i_2nd] > PHASE7_RESIDUAL_THRESHOLD)):
                seq = ', '.join(f'{s:.1e}' for s in solver_err[order])
                ax.text(
                    0.5, 0.55,
                    'Richardson / TMM\\nplot skipped\\n'
                    + r'($\\|r\\|/\\|b\\|$ at finest pair'
                    + f' > {PHASE7_RESIDUAL_THRESHOLD:.0e})',
                    transform=ax.transAxes,
                    ha='center', va='center', fontsize=8.5, color='C3',
                )
                ax.text(
                    0.5, 0.28,
                    'final residuals (coarse → fine):\\n' + seq,
                    transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, color='gray',
                    family='monospace',
                )
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.tick_params(axis='both', which='both', direction='in',
                                top=True, right=True, bottom=True, left=True)
                if i == 0:
                    ax.set_title(MATERIAL_LABEL[mat])
                if j == 0:
                    ax.set_ylabel(SHAPE_LABEL[shape] + '\\n'
                                  + r'$|X(\\ell_c)-X_{\\rm ref}|/|X_{\\rm ref}|$')
                if i == len(PHASE7_SHAPES) - 1:
                    ax.set_xlabel(r'$n_{\\rm tet}$')
                continue

            anchor_set = False
            tmm_ref = PHASE7_CONV_REF.get((shape, mat))   # None for GRE
            for obs_key, _latex, kind, color in PHASE7_OBS_FIG4:
                y = _conv_value(viem, obs_key, kind)
                if y is None or len(y) != n_lc:
                    continue
                y = np.asarray(y, dtype=float)

                # ----- Richardson curve (always drawn when defined) -------
                X_rich = _phase7_richardson_x_inf(y, x_n, p=2.0)
                eps_r  = _phase7_relerr_against(y, X_rich)
                if eps_r is not None:
                    m_fin = np.isfinite(eps_r) & (eps_r > 0)
                    ax.plot(x_n[m_fin], eps_r[m_fin], ls='--',
                            color=color, lw=0.9, alpha=0.55, zorder=2)
                    m_ok  = m_fin & converged
                    m_stl = m_fin & ~converged
                    ax.plot(x_n[m_ok], eps_r[m_ok], ls='', marker='s',
                            markersize=5, color=color, mfc='none', mew=1.0,
                            zorder=3)
                    if m_stl.any():
                        ax.plot(x_n[m_stl], eps_r[m_stl], ls='', marker='x',
                                markersize=8, mew=1.4, color=color, zorder=4)

                # ----- TMM curve (oblate only, when β=0 ref available) ---
                eps_t = None
                if tmm_ref is not None:
                    ref_v = tmm_ref.get(obs_key)
                    if ref_v is not None:
                        if hasattr(ref_v, 'shape') and ref_v.ndim >= 2:
                            ref_scalar = _phase7_kind_to_scalar(
                                ref_v[0, 0], kind)
                        else:
                            ref_scalar = _phase7_kind_to_scalar(ref_v, kind)
                        eps_t = _phase7_relerr_against(y, ref_scalar)
                        if eps_t is not None:
                            m_fin = np.isfinite(eps_t) & (eps_t > 0)
                            ax.plot(x_n[m_fin], eps_t[m_fin], ls='-',
                                    color=color, lw=0.9, alpha=0.85, zorder=2)
                            m_ok  = m_fin & converged
                            m_stl = m_fin & ~converged
                            ax.plot(x_n[m_ok], eps_t[m_ok], ls='', marker='o',
                                    markersize=5, color=color, mfc=color,
                                    mew=0.5, zorder=3)
                            if m_stl.any():
                                ax.plot(x_n[m_stl], eps_t[m_stl], ls='',
                                        marker='x', markersize=8, mew=1.4,
                                        color=color, zorder=4)

                # ----- Slope guide anchored on TMM > Richardson > coarsest
                if not anchor_set:
                    eps_anchor = None
                    if eps_t is not None and np.isfinite(eps_t[i_2nd]) and eps_t[i_2nd] > 0:
                        eps_anchor = float(eps_t[i_2nd])
                    elif eps_r is not None and np.isfinite(eps_r[i_2nd]) and eps_r[i_2nd] > 0:
                        eps_anchor = float(eps_r[i_2nd])
                    if eps_anchor is not None:
                        xs = np.array([x_n.min(), x_n.max()])
                        ys = eps_anchor * (xs / x_n[i_2nd]) ** (-2.0 / 3.0)
                        ax.plot(xs, ys, ls=':', color='k', lw=0.9, alpha=0.6,
                                zorder=0)
                        anchor_set = True

            if not anchor_set:
                xs = np.array([x_n.min(), x_n.max()])
                ys = 1e-2 * (xs / x_n.max()) ** (-2.0 / 3.0)
                ax.plot(xs, ys, ls=':', color='k', lw=0.9, alpha=0.6, zorder=0)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.axhline(1e-2, color='gray', lw=0.4, ls=':')
            ax.axhline(1e-3, color='gray', lw=0.4, ls=':')
            ax.tick_params(axis='both', which='both', direction='in',
                            top=True, right=True, bottom=True, left=True)
            if i == 0:
                ax.set_title(MATERIAL_LABEL[mat])
            if j == 0:
                ax.set_ylabel(SHAPE_LABEL[shape] + '\\n'
                              + r'$|X(\\ell_c)-X_{\\rm ref}|/|X_{\\rm ref}|$')
            if i == len(PHASE7_SHAPES) - 1:
                ax.set_xlabel(r'$n_{\\rm tet}$')

    handles_obs = [
        Line2D([], [], color=col, marker='', linestyle='-', lw=1.4, label=latex)
        for _, latex, _, col in PHASE7_OBS_FIG4
    ]
    handles_ref = [
        Line2D([], [], color='gray', marker='o', markersize=6,
                linestyle='-', lw=0.9, label=r'vs TMM ($\\beta\\!=\\!0$, oblate)'),
        Line2D([], [], color='gray', marker='s', markersize=6, mfc='none',
                mew=1.0, linestyle='--', lw=0.9,
                label=r'vs Richardson $X_\\infty^{\\rm Rich}$'),
    ]
    handles_extra = [
        Line2D([], [], color='k', linestyle=':', lw=0.9, alpha=0.6,
                label=r'slope $-2/3$ ($\\varepsilon\\propto h^2$)'),
        Line2D([], [], color='gray', marker='x', linestyle='', mew=1.4,
                markersize=7, label='solver stalled (MAXITER=200)'),
    ]
    fig.legend(handles=handles_obs + handles_ref + handles_extra,
                ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.07),
                fontsize=8)
    plt.tight_layout(rect=(0, 0.06, 1, 0.97))
    if save:
        save_fig(fig, 'fig4_lc_convergence_nonsphere')
    plt.show()


plot_phase7()
'''


def _split_lines_keep_eol(text: str):
    """Split into list of lines retaining trailing '\n' (jupyter convention)."""
    lines = text.splitlines(keepends=True)
    return lines if lines else ['']


def make_md_cell(cid, source):
    return {
        'cell_type': 'markdown',
        'id': cid,
        'metadata': {},
        'source': _split_lines_keep_eol(source),
    }


def make_code_cell(cid, source):
    return {
        'cell_type': 'code',
        'id': cid,
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': _split_lines_keep_eol(source),
    }


def main():
    nb = json.load(NB.open())
    cells = nb['cells']
    ids = [c.get('id', '') for c in cells]

    new_md   = make_md_cell(MD_ID, MD_SOURCE)
    new_code = make_code_cell(CODE_ID, CODE_SOURCE)

    if MD_ID in ids and CODE_ID in ids:
        # Replace existing cells in-place
        i_md   = ids.index(MD_ID)
        i_code = ids.index(CODE_ID)
        cells[i_md]   = new_md
        cells[i_code] = new_code
        action = 'replaced'
    elif MD_ID not in ids and CODE_ID not in ids:
        # Append at the end (after the cost-summary cell)
        cells.append(new_md)
        cells.append(new_code)
        action = 'appended'
    else:
        raise RuntimeError(
            f'partial existence: MD_ID in cells={MD_ID in ids}, CODE_ID in cells={CODE_ID in ids}'
            ' — clean up manually before re-running.'
        )

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    print(f'{action} cells {MD_ID!r} + {CODE_ID!r} in {NB}')
    print(f'total cells now: {len(cells)}')


if __name__ == '__main__':
    main()

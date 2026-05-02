# VIE Formulation in `block-VIEM.jl`

block-VIEM.jl ([github.com/NobuhiroMoteki/block-VIEM.jl](https://github.com/NobuhiroMoteki/block-VIEM.jl))
で採用する volume integral equation (VIE) の定式化と、先行研究 (本リポジトリ
[`.claude/refs/viem/`](../../../.claude/refs/viem/) 以下) に対する位置づけ・
引用系譜をまとめる。

§1 Intro / §2 Theory / §3 Methods 執筆時の根拠資料。並列文書として
[`dda_polarizability_prescription.md`](dda_polarizability_prescription.md)
が block-DDA_Py 側の polarizability formulation を扱う。

---

## 1. 全体像 — block-VIEM.jl の構成要素

| 要素 | 採用方式 | 直接の出典 |
| --- | --- | --- |
| 整理されている定式化 | **EFVIE-D formulation** ($\boldsymbol{D} = \varepsilon \boldsymbol{E}$ を unknown) | Schaubert et al. 1984 |
| Mesh & Basis | tetrahedral + **SWG (half-SWG)** divergence-conforming | Schaubert et al. 1984 (boundary 扱い含む) |
| 特異積分 | **Vertex Duffy transform** ($\beta=1$ 標準形) | Mousavi & Sukumar 2010 (Duffy 1982 の現代的再述) |
| MVP 加速 | **AIM (Adaptive Integral Method) + FFT** | Bleszynski et al. 1996 |
| Solver | **Block GMRES / Block BiCGSTAB** (default Block GMRES since v0.7.1) | Saad & Schultz 1986; Simoncini & Gallopoulos 1996; El Guennouni et al. 2003; etc. |
| 異方性媒質 | diagonal $\varepsilon_p = \mathrm{diag}(\varepsilon_x, \varepsilon_y, \varepsilon_z)$ | Kobidze & Shanker 2004 (一般 anisotropic VIE) |
| Validation | Mie / MSTM / block-DDA_Py / α-symmetry | (本研究 fig 3 / fig 4) |

ソースコード [src/impedance.jl](../../../../../Julia/block-VIEM.jl/src/impedance.jl) 冒頭に
形式名が明示:

```julia
# Impedance matrix element Z_mn for the SWG/EFVIE-D formulation.
```

## 2. 体積積分方程式の D-formulation

非磁性 ($\mu = \mu_0$)、線型媒質に対する electric-field VIE:
$$
\boldsymbol{E}(\boldsymbol{r}) = \boldsymbol{E}_{\rm inc}(\boldsymbol{r})
+ \int_V \boldsymbol{G}(\boldsymbol{r},\boldsymbol{r}')
   \,\chi(\boldsymbol{r}')\,\boldsymbol{E}(\boldsymbol{r}')\,\mathrm{d}^3\boldsymbol{r}'
$$

ここで $\chi = \varepsilon_r - 1$、$\boldsymbol{G}$ は自由空間 dyadic Green 関数。

unknown を $\boldsymbol{D} = \varepsilon \boldsymbol{E}$ に変換 (Schaubert et al. 1984
Eq. (7)-(8)) し、contrast ratio $\kappa = (\varepsilon - \varepsilon_0)/\varepsilon$ を
導入すると、polarization current は $\boldsymbol{J} = j\omega\kappa\boldsymbol{D}$ と
なり、媒質界面で **normal component が連続な物理量 $\boldsymbol{D}$** が
unknown に。

block-VIEM.jl の impedance matrix element ([impedance.jl](../../../../../Julia/block-VIEM.jl/src/impedance.jl)):
$$
Z_{mn} = \int \frac{\boldsymbol{f}_m \cdot \boldsymbol{f}_n}{\varepsilon(\boldsymbol{r})}\,\mathrm{d}V
- \frac{1}{\varepsilon_{\rm bg}}\!\!\iint\!\! \kappa(\boldsymbol{r}')
   \bigl[k_0^2\,\boldsymbol{f}_m\!\cdot\!\boldsymbol{f}_n'
   - (\nabla\!\cdot\!\boldsymbol{f}_m)(\nabla'\!\cdot\!\boldsymbol{f}_n')\bigr]
   G(R)\,\mathrm{d}V'\,\mathrm{d}V
$$

これは **weakened pair integral form** (微分作用素を testing function に
移して特異性を緩和した形) で、Schaubert et al. 1984 Section II の枠組みを
そのまま実装したもの。

## 3. SWG basis on tetrahedral mesh

### Internal face (隣接 2 tet で共有)

[swg.jl](../../../../../Julia/block-VIEM.jl/src/swg.jl) の Eq. (10), (12) of Schaubert
et al. 1984 に対応:
$$
\boldsymbol{f}_n(\boldsymbol{r}) = \pm\frac{a_n}{3 V_n^\pm}(\boldsymbol{r} - \boldsymbol{p}_n^\pm),
\qquad
\nabla\cdot\boldsymbol{f}_n = \pm\frac{a_n}{V_n^\pm}
$$

性質 (Schaubert et al. 1984 §II の項目 1)–5)):

1. tet 内では 4 face 分の basis 線型結合で任意定方向 vector 表現可能
2. 共有面 $a_n$ 以外では normal 成分ゼロ
3. $a_n$ 上では normal 成分が一定かつ連続 → **$\boldsymbol{D}\cdot\hat{n}$ の連続条件**
4. $\boldsymbol{f}_n$ の係数 $D_n$ がそのまま $a_n$ 上の $D\cdot\hat{n}$ を表す

### Boundary face (片側 tet のみ) — "half-SWG"

Schaubert et al. 1984 p.79 **項目 6** が陽に規定:

> *"If face $n$ is on the boundary of $V$, then only one of the tetrahedrons,
> $T_n^+$ or $T_n^-$, is interior to $V$. In this case it is assumed that
> $\boldsymbol{f}_n$ is defined only over the interior tetrahedron and the
> exterior tetrahedron is not defined."*

block-VIEM.jl の "half-SWG" はこの **片側 tet support の標準形そのもの**で、
独自概念ではない。コード [swg.jl](../../../../../Julia/block-VIEM.jl/src/swg.jl) の
docstring も "Following Schaubert-Wilton-Glisson (1984)" と明記。

境界面では $\boldsymbol{D}\cdot\hat{n} \ne 0$ が許され、これが **表面分極電荷**
を表現する。Schaubert et al. 1984 Eq. (17):
$$
\rho_{sn}(\boldsymbol{r}) = D_n(\kappa_+ - \kappa_-),\qquad \boldsymbol{r} \in a_n
$$

外部真空 ($\kappa_- = 0$) では $\rho_{sn} = D_n\kappa_+$ で表面分極電荷を陽に
扱う仕組みが、basis function の構築自体に組み込まれている。

## 4. 異方性媒質 (diagonal $\varepsilon_p$) への拡張

[impedance.jl](../../../../../Julia/block-VIEM.jl/src/impedance.jl) の `_aniso_params`
関数で diagonal anisotropy を扱う。重み付き形:
$$
Z_{mn} = \sum_\alpha \frac{1}{\varepsilon_\alpha}\!\!\int\!\!(\boldsymbol{f}_m)_\alpha(\boldsymbol{f}_n)_\alpha\,\mathrm{d}V
- \frac{1}{\varepsilon_{\rm bg}}\!\!\iint\!\!\bigl[k_0^2 \sum_\alpha \kappa_\alpha (\boldsymbol{f}_m)_\alpha(\boldsymbol{f}_n)_\alpha
  - \kappa_{\rm avg}(\nabla\cdot\boldsymbol{f}_m)(\nabla'\cdot\boldsymbol{f}_n')\bigr]G\,\mathrm{d}V'\mathrm{d}V
$$
($\kappa_\alpha = (\varepsilon_\alpha - \varepsilon_{\rm bg})/\varepsilon_\alpha$,
 $\kappa_{\rm avg} = (\kappa_x+\kappa_y+\kappa_z)/3$)

SWG 構造 ($\partial(\boldsymbol{f}_n)_\alpha/\partial r_\alpha = a_n/(3V)$ for all $\alpha$) を
利用して $\nabla\nabla\cdot$ 項を簡略化 (isotropic では厳密、mildly anisotropic では
$O(|\Delta\kappa|/|\kappa|)$ 誤差)。

一般 ε, μ tensor の **VIE 定式化の foundational reference は Kobidze & Shanker 2004**
(volume equivalence theorem + 3D RWG basis)。block-VIEM.jl は diagonal
anisotropy に絞った実用的特殊化。

## 5. 特異積分 — Vertex Duffy transform

self-element $T_m \cap T_n \ne \emptyset$ における $1/R$ 特異性を、Duffy 変換
で解析的にキャンセル。block-VIEM.jl [duffy.jl](../../../../../Julia/block-VIEM.jl/src/duffy.jl) は
**Mousavi & Sukumar 2010 Eq. (1) with $\beta = 1$** を直接実装:

$$
(u_1, u_2, u_3) \to (\lambda_s, \lambda_a, \lambda_b, \lambda_c):\quad
\lambda_s = 1 - u_1,\;\lambda_a = u_1(1-u_2),\;\lambda_b = u_1 u_2(1-u_3),\;\lambda_c = u_1 u_2 u_3
$$

Jacobian:
$$
J_D(u) = 6 V_T\, u_1^2\, u_2
$$

singular vertex で $R \sim u_1$ なので、Jacobian の $u_1^2$ 因子が $1/R$ 特異性を
**解析的に相殺**。ソースコードのコメント:

```julia
# The factor `u1^2` cancels the 1/R singularity at `v_s` because R ~ u1 there.
```

Mousavi-Sukumar 2010 の novel 部分 ($\alpha \ne 1$ への一般化) は本研究では不要。
$\alpha = 1$ ($1/R$) の場合 $\beta = 1$ が optimal と論文自身が認めており、本研究は
この標準形を採用。Duffy 1982 の original paper の現代的再述。

## 6. AIM-FFT による行列ベクトル積加速

block-VIEM.jl は **AIM (Adaptive Integral Method)** で MVP を $O(N \log N)$ に圧縮
([aim_operator.jl](../../../../../Julia/block-VIEM.jl/src/aim_operator.jl) etc.)。AIM の
原典は **Bleszynski, Bleszynski & Jaroszewicz 1996**。

AIM の概略 (Bleszynski et al. 1996):
$$
A = A_{\rm near} + A_{\rm far}
$$

- $A_{\rm near}$: SWG 基底の near-zone 寄与、sparse MoM 行列として直接保持
- $A_{\rm far}$: Cartesian grid 上の auxiliary point sources へ projection、
   Toeplitz 構造を FFT で circulant 拡張、MVP は $O(N \log N)$

block-VIEM.jl は AIM grid pitch を mesh 統計から auto-detect する機構を持つ。
DDA-Goodman の **lattice-restricted FFT** と異なり、AIM は **任意の tetrahedral
mesh** に適用できるのが本質的利点。

## 7. Block-Krylov solver

multi-orientation 入射で生じる multiple right-hand sides $\boldsymbol{B} \in
\mathbb{C}^{N \times L}$ を block-GMRES で同時に解く ([block_krylov.jl](../../../../../Julia/block-VIEM.jl/src/block_krylov.jl), [solver.jl](../../../../../Julia/block-VIEM.jl/src/solver.jl))。

理論的根拠は block-DDA_Py と共通の citation chain:

- Saad & Schultz 1986 — GMRES 原典
- Simoncini & Gallopoulos 1996 — block-GMRES 収束理論
- O'Leary 1980 — block-Krylov 起源 (symmetric)
- Gutknecht-Schmelzer 2009 — block grade 理論
- El Guennouni et al. 2003 — EM scattering 文脈での block 法、Bl-BiCGSTAB

## 8. 先行研究の中での位置づけ — 引用系譜

```text
[D-formulation 起源]
   Schaubert et al. 1984       D = εE, divergence-conforming SWG basis,
                                Eq. (17) で表面分極電荷、項目 6 で
                                境界面 (half-SWG) の標準形を確立
              │
              │ (high-permittivity への拡張、tetrahedral 維持)
              ▼
[VIE for plasmonic / high-perm]
   Kottmann & Martin 2000       tetrahedral VIE + Green tensor 正則化、
                                plasmon-polariton 共鳴を再現

              │ (Galerkin formulation の包括的比較)
              ▼
[VIE Galerkin formulation 比較]
   Botha 2006                   div-conforming vs curl-conforming basis、
                                solenoidal vs divergence-conforming の議論
                                → D-formulation 採用の正当化
              │
              │ (D vs E vs H の安定性比較、high contrast)
              ▼
[D vs E vs H 比較]
   Markkanen et al. 2012       VEFIE-D / VEFIE-E / VMFIE を high-contrast
                                で比較、D-formulation が high-contrast に
                                最適と結論 (lowest mixed-order basis 使用時)
                                → block-VIEM.jl の D 選択を直接 justify

              │ (alternative: equivalent currents formulation)
              ▼
[modern FFT-VIE alternative]
   Polimeridis et al. 2014     JVIE (equivalent currents) + FFT、
                                stable for high contrast (uniqueness preservation)、
                                ─ block-VIEM.jl と直接的競合


[一般 anisotropic VIE foundational]
   Kobidze & Shanker 2004      ε, μ tensor を扱う 3D RWG VIE、
                                volume equivalence theorem、
                                ← block-VIEM.jl の anisotropic 対応の基準


[特異積分]
   Duffy 1982 (入手困難)         vertex singularity の Duffy 変換 (origin)
        │
        │ (現代的再述、$1/r^\alpha$ への一般化)
        ▼
   Mousavi & Sukumar 2010       generalized Duffy transform、
                                $\beta = 1$ の標準形は本研究用 →
                                ← block-VIEM.jl duffy.jl が直接実装


[FFT-accelerated MoM]
   Bleszynski et al. 1996       AIM 原典、A = A_near + A_far、
                                volumetric で O(N log N) MVP →
                                ← block-VIEM.jl AIM 実装の基礎


[VIE 演算子の数学的性質]
   Peltoniemi 1996 (JQSRT)       variational VIEM、$G_1$, $G_2$ の self-Green
                                解析評価
   Rahola 2000                  (I − Gχ) の固有値分布 (球散乱体)
   Budko & Samokhin 2006        本質スペクトル λ_ess = ε_r、plasmon 分類
                                → §5 Discussion で Au carve-out の理論基盤


[block-Krylov]
   Saad & Schultz 1986         GMRES base
   O'Leary 1980                block-Krylov (symmetric)
   Gutknecht 2007              Krylov methods survey
   Gutknecht & Schmelzer 2009  block grade 理論
   Simoncini & Gallopoulos 1996 block-GMRES 収束
   El Guennouni et al. 2003    Bl-BiCGSTAB + EM 動機
                                → block-VIEM.jl block_krylov.jl, solver.jl
```

## 9. block-VIEM.jl の novel 寄与 (先行研究にない要素)

整理すると以下の 2 点に絞られる:

1. **D-formulation × AIM × Block-Krylov の三位一体**
   先行研究 (Bleszynski 1996 = AIM だが MoM 一般、Polimeridis 2014 = FFT-JVIE
   だが単一 RHS) は MVP 加速か block 化のどちらか一方のみ。block-VIEM.jl は
   両者を D-formulation の枠組みで統合し、plasmonic / anisotropic / 高コントラスト
   いずれにも適用可能。

2. **block-DDA_Py との完全互換 schema (HDF5 + CAS-v2 observable + 形状モデル)**
   同一物理を二系統の独立コードで検証可能。本研究 §4.3 (fig 3) の DDA vs 厳密解
   と §3.X (block-DDA_Py vs block-VIEM.jl) の cross-validation がこれに依存。

これら以外の要素 (D-formulation そのもの / SWG basis / half-SWG 境界扱い /
Duffy quadrature / AIM / 各 block-Krylov アルゴリズム / KobidzeShanker 系
anisotropic 枠組み) は **既に確立された先行研究の組合せ**であり、本研究の
contribution として誇張すべきではない。

## 10. 関連 references (refs.bib key)

VIE methodology lineage (上流 → 下流):

- `Schaubert1984` — *IEEE TAP* 32(1):77-85 (DOI: 10.1109/TAP.1984.1143193)
   D-formulation と SWG basis の原典。**項目 6 で境界面 (half-SWG) を陽に規定**、
   Eq. (17) で表面分極電荷を導出。
- `KottmannMartin2000` — *IEEE TAP* 48(11):1719-1726
   tetrahedral VIE + 正則化、plasmon-polariton。
- `Botha2006` — *J. Comput. Phys.* 218:141-158
   Galerkin VIE 比較、div-conforming basis の正当化。
- `KobidzeShanker2004` — *IEEE TAP* 52(10):2650-2659
   一般 anisotropic VIE (ε, μ tensor)、3D RWG。
- `Markkanen2012` — *IEEE TAP* 60(5):2367-2374
   D vs E vs H 比較、high-contrast での D-formulation 優位を結論。
- `Polimeridis2014` — *J. Comput. Phys.* 269:280-296
   JVIE + FFT、modern alternative。

singular integration:

- `MousaviSukumar2010` — *Comput. Mech.* 45:127-140 (open access)
   generalized Duffy transform、本研究は $\beta = 1$ 標準形を採用。
   Duffy 1982 の現代的再述。

FFT-based MoM acceleration:

- `Bleszynski1996` — *Radio Sci.* 31(5):1225-1251
   AIM 原典。volumetric で $O(N \log N)$。

VIE 演算子 spectrum & resonance:

- `Peltoniemi1996` — *JQSRT* 55(5):637-647
- `Rahola2000` — *SIAM J. Sci. Comput.* 21(5):1740-1754
- `BudkoSamokhin2006` — *Phys. Rev. Lett.* 96, 023904

block-Krylov: 並行文書 [`dda_polarizability_prescription.md`](dda_polarizability_prescription.md)
末尾と共通。

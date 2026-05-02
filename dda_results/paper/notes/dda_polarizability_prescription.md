# Dipole Polarizability Prescription in `block-DDA_Py`

本コード [bl_dda/scatterer.py](../../../bl_dda/scatterer.py) (`set_interaction_matrix`,
L143-156) で採用している双極子分極率 (dipole polarizability) の数式と、その
出典である Chaumet & Rahmani 2009 (JQSRT) Eq. (16)-(17) と Peltoniemi 1996
(JQSRT) の解析的 self-Green 評価との対応関係をまとめる。

§3 Methods で polarizability の式を述べる際の根拠資料。

---

## 1. 体積積分方程式の出発点

非磁性散乱体 ($\mu = 1$) に対する volume integral equation:
$$
\boldsymbol{E}(\boldsymbol{r}) = \boldsymbol{E}_{\rm inc}(\boldsymbol{r}) + \int_V \boldsymbol{G}^{ee}(\boldsymbol{r},\boldsymbol{r}')\,\chi(\boldsymbol{r}')\,\boldsymbol{E}(\boldsymbol{r}')\,\mathrm{d}^3\boldsymbol{r}'
$$

ここで $\chi = \varepsilon_r - 1$、$\boldsymbol{G}^{ee}$ は自由空間 dyadic
Green 関数。

体積 $V$ を $N$ 個の cubic セル $V_j$ (体積 $d^3$) に離散化し、セル内で
$\boldsymbol{E}$ と $\chi$ が一定とすれば、site $i$ における式は:
$$
\boldsymbol{E}(\boldsymbol{r}_i) = \boldsymbol{E}_{\rm inc}(\boldsymbol{r}_i) + \sum_{j=1}^{N} \underbrace{\left[\int_{V_j} \boldsymbol{G}^{ee}(\boldsymbol{r}_i,\boldsymbol{r}')\,\mathrm{d}^3\boldsymbol{r}'\right]}_{\equiv\;\boldsymbol{G}_{\rm int}^{ee}(\boldsymbol{r}_i,\boldsymbol{r}_j)} \chi(\boldsymbol{r}_j)\,\boldsymbol{E}(\boldsymbol{r}_j)
$$

これが CR2009 Eq. (9) の本質。**$i=j$ の self-term**
$\boldsymbol{G}_{\rm int}^{ee}(\boldsymbol{r}_i,\boldsymbol{r}_i)$ は特異積分で、
ここを正確に評価するのが議論の核心。

## 2. Peltoniemi 1996 の self-term 解析評価

Peltoniemi は、点 $\boldsymbol{r}_i$ を中心とする半径 $a$ の小球 $V_a$ で
特異点を切り出し、$\boldsymbol{E}$ を Taylor 展開して $V_a$ 内の積分を解析的
に評価。結果が Eq. (4) の self-action:
$$
\int_{V_a} \boldsymbol{G}^{ee}(\boldsymbol{r}_i,\boldsymbol{r}')\,\chi\,\boldsymbol{E}(\boldsymbol{r}_i)\,\mathrm{d}^3\boldsymbol{r}' \;=\; \Bigl[-\tfrac{1}{3} + G_1(ka) + m^2 G_2(ka)\Bigr]\,\chi\,\boldsymbol{E}(\boldsymbol{r}_i)\,\boldsymbol{I}
$$

with
$$
G_1(ka) = \tfrac{2}{3}\bigl[(1-ika)\,e^{ika} - 1\bigr], \quad
G_2(ka) = \bigl(1-ika - \tfrac{7}{15}(ka)^2 + \tfrac{2}{15}(ka)^3\bigr)e^{ika} - 1
$$

各項の物理的意味:

- $-\tfrac{1}{3}$: 球の **静的 depolarization factor** (Lorentz limit)
- $G_1$: **leading-order radiation correction** (dipole 放射の reaction)
- $G_2$: **higher-order correction** (Peltoniemi 自身の novel 貢献)

## 3. CR2009 の polarizability 行列表現

CR2009 は self-term を「polarizability の中に吸収する」定式化を採用。
Eq. (12) で local field $\boldsymbol{E}$ を $i \ne j$ の和で書き直し:
$$
\boldsymbol{E}(\boldsymbol{r}_i) = \boldsymbol{E}_{\rm inc}(\boldsymbol{r}_i) + \sum_{j \ne i} \boldsymbol{G}_{\rm int}^{ee}(\boldsymbol{r}_i,\boldsymbol{r}_j)\,\boldsymbol{\alpha}^e(\boldsymbol{r}_j)\,\boldsymbol{E}(\boldsymbol{r}_j)
$$

ここで polarizability tensor $\boldsymbol{\alpha}^e$ が **self-term を内包**
するように定義される。具体的には Eq. (17):
$$
\boldsymbol{\alpha}^e(\boldsymbol{r}_j) = \boldsymbol{\alpha}_0^e(\boldsymbol{r}_j)\,\Bigl[\boldsymbol{I} - \bigl(\boldsymbol{G}_{\rm int}^{ee}(\boldsymbol{r}_j,\boldsymbol{r}_j) + \tfrac{4\pi}{3}\boldsymbol{I}\bigr)\,\frac{\boldsymbol{\alpha}_0^e(\boldsymbol{r}_j)}{d^3}\Bigr]^{-1}
$$

ここで Eq. (16):
$$
\boldsymbol{\alpha}_0^e(\boldsymbol{r}_i) = \frac{3d^3}{4\pi}\,(\boldsymbol{\varepsilon}(\boldsymbol{r}_i)-\boldsymbol{I})(\boldsymbol{\varepsilon}(\boldsymbol{r}_i)+2\boldsymbol{I})^{-1}
$$

は **静的 Clausius-Mossotti polarizability** (Lorentz-Lorenz form)。

## 4. 二つの定式化を結ぶ等式

CR2009 自身は self-Green 評価の詳細を **Chaumet, Sentenac & Rahmani 2004
(Phys. Rev. E 70, 036606)** (CR2009 ref [3]) に委ねている。CSR2004 Eq. (11)
は cubic cell 上の $\boldsymbol{G}_{\rm int}^{ee}(\boldsymbol{r}_i,\boldsymbol{r}_i)$
を Weyl 展開で**近似なし**に解析的評価し、$\Delta \to 0$ 極限で
$\lim \boldsymbol{G}_{\rm int}^{ee}(\boldsymbol{r}_i,\boldsymbol{r}_i) = -(4\pi/3)\boldsymbol{I}$
を導出している (これが CR2009 Eq. (17) の $+(4\pi/3)\boldsymbol{I}$ の出所)。

実用上は CSR2004 の cubic-cell 評価を **等価体積球** ($V_a = d^3$、半径
$a = (3d^3/4\pi)^{1/3}$) で近似した Peltoniemi 1996 の解析評価が等価で扱い
やすい。これを使って計算すると:
$$
\boldsymbol{G}_{\rm int}^{ee}(\boldsymbol{r}_i,\boldsymbol{r}_i) = 4\pi\Bigl[-\tfrac{1}{3} + G_1(ka)\Bigr]\boldsymbol{I} \quad (\text{$G_2$ 項を省略})
$$

すると CR2009 Eq. (17) の括弧内は:
$$
\boldsymbol{G}_{\rm int}^{ee}(\boldsymbol{r}_i,\boldsymbol{r}_i) + \frac{4\pi}{3}\boldsymbol{I}
\;=\; 4\pi\Bigl[-\tfrac{1}{3} + G_1(ka)\Bigr]\boldsymbol{I} + \tfrac{4\pi}{3}\boldsymbol{I}
\;=\; 4\pi\,G_1(ka)\,\boldsymbol{I}
$$

つまり **CR2009 の自己項補正 ≡ Peltoniemi の $G_1$ × $4\pi$**。$-1/3$ の
static depolarization は CR2009 が陽に書いた $+(4\pi/3)\boldsymbol{I}$ で
打ち消される (両者は同じ静的部分を別の場所で扱っている)。

具体的に書き下すと:
$$
\boxed{\;4\pi\,G_1(ka) = \frac{8\pi}{3}\Bigl[(1-ika)\,e^{ika} - 1\Bigr]\;}
$$

これが [scatterer.py:152](../../../bl_dda/scatterer.py#L152) の `M_term`:

```python
M_term = (8 * np.pi / 3) * ((1 - 1j * self.k * a) * np.exp(1j * self.k * a) - 1)
```

## 5. 等価体積球の半径の対応

Peltoniemi の $a$ は exclusion sphere の半径だが、DDA で cubic cell
($V_{\rm cell} = d^3$) を**同体積の球** (Doyle 1989 の prescription) に置き
換えると:
$$
\frac{4\pi}{3}a^3 = d^3 \;\Longleftrightarrow\; a = \left(\frac{3d^3}{4\pi}\right)^{1/3} = \left(\frac{3}{4\pi}\right)^{1/3} d \approx 0.620\,d
$$

これが [scatterer.py:151](../../../bl_dda/scatterer.py#L151) の:

```python
a = (3 * self.element_vol / (4 * np.pi)) ** (1 / 3)
```

## 6. 最終形 (コードと CR2009 の完全対応)

CR2009 Eq. (17) を非磁性 ($\mu = 1$)・等方 ($\varepsilon = \varepsilon\boldsymbol{I}$)
に特殊化、scalar 形で書くと:
$$
\alpha^e = \frac{\alpha_0^e}{\,1 - \bigl(G_{\rm int}^{ee} + \tfrac{4\pi}{3}\bigr)\alpha_0^e/d^3\,}
\;=\;\frac{\alpha_0^e}{\,1 - 4\pi\,G_1(ka)\,\alpha_0^e/d^3\,}
\;=\;\frac{\alpha_0^e}{\,1 - M_{\rm term}\,\alpha_0^e/V_{\rm cell}\,}
$$

これが [scatterer.py:148, 153](../../../bl_dda/scatterer.py#L148):

```python
alpha0_E = (3 / (4 * np.pi)) * ((self.eper_r - 1) / (self.eper_r + 2)) * self.element_vol
self.alpha_E = alpha0_E / (1 - M_term * alpha0_E / self.element_vol)
```

## 7. 関係性のまとめ

| 役割 | Peltoniemi 1996 | CSR2004 | CR2009 | block-DDA_Py |
| --- | --- | --- | --- | --- |
| 静的部分 | $-\tfrac{1}{3}\chi$ in Eq. (4) | Eq. (8): $\alpha_0 = (3V/4\pi)(\varepsilon-1)/(\varepsilon+2)$ | $\boldsymbol{\alpha}_0^e$ in Eq. (16) | `alpha0_E` |
| 放射補正 ($G_1$) | $\tfrac{2}{3}[(1-ika)e^{ika}-1]$ | Eq. (11): cubic cell 上の $\boldsymbol{G}_{\rm int}^{ee}$ exact | $\boldsymbol{G}_{\rm int}^{ee} + \tfrac{4\pi}{3}\boldsymbol{I} = 4\pi G_1$ | `M_term` |
| 高次補正 ($G_2$) | $m^2 G_2(ka)$ | (Weyl 評価により暗に含まれる) | 省略 | **省略** |
| 全体形 | Eq. (4) self-action | Eq. (10) integrated form | Eq. (17) inverse form | `alpha_E = alpha0_E / (1 - M_term * alpha0_E / V)` |
| 磁性媒質 | 非対象 | 非対象 | Eq. (15) で $\boldsymbol{\alpha}^m$ も導出 | 非対象 ($\mu=1$) |

### 一文で書くなら

> *CR2009 Eq. (17) の self-term 補正 $\bigl(\boldsymbol{G}_{\rm int}^{ee} + \tfrac{4\pi}{3}\boldsymbol{I}\bigr)$ は、**等価体積球** ($a^3 = 3d^3/4\pi$) を用いた Peltoniemi 1996 の self-Green 解析評価において **$G_1$ 部分のみ**を保持したものに $4\pi$ を掛けた量に等しい (静的 $-1/3$ は CR2009 の $+4\pi/3$ で相殺)。$G_2$ 高次項は両者とも省略している。*

これが両論文と本コードを貫く数学的同値関係。

---

## 関連 references (refs.bib key)

Citation chain (上流 → 下流):

- `Peltoniemi1996` — *J. Quant. Spectrosc. Radiat. Transfer* 55(5):637-647
  (DOI: 10.1016/0022-4073(96)00007-6)。**等価体積球**で self-Green を解析評価
  した $G_1$, $G_2$ の歴史的出典。
- `ChaumetSentenacRahmani2004` — *Phys. Rev. E* 70:036606
  (DOI: 10.1103/PhysRevE.70.036606)。**cubic cell** 上で
  $\boldsymbol{G}_{\rm int}^{ee}$ を Weyl 展開で**近似なし**に評価 (Eq. 11)。
  CR2009 ref [3]。
- `ChaumetRahmani2009` — *J. Quant. Spectrosc. Radiat. Transfer* 110(1-2):22-29
  (DOI: 10.1016/j.jqsrt.2008.09.004)。CSR2004 を磁性媒質に拡張、Eq. (17) の
  逆行列形式を確立。**コード直接出典**。

参考:

- (補足) Doyle 1989 — 等価体積球 ($a^3 = 3d^3/4\pi$) の prescription の歴史的
  出典。必要なら refs.bib に追加する。

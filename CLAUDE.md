# block-DDA_Pyとそのリファクタリング方針

## block-DDA_Py概要
- block-DDA_Pyの機能と特徴についてはREADME.mdに記載のとおり
- 単発のDDAシミュレーション実行用のコード: test_dda.ipynb
- 入力パラメータスイープのための繰り返しDDAシミュレーション用のコード: run_dda.py
- run_dda.py実行時の出力データ格納ファイル(.hdf5)生成コード: dda_results/create_h5py.ipynb

## モジュール構成と役割
- `bl_dda/`       : DDA本体（双極子分極・散乱振幅の計算）
- `mvp_fft/`      : 行列ベクトル積のFFT高速化（今回の主な変更対象）
- `bl_krylov/`    : block-Krylov反復ソルバー
- `shape_model/`  : Gaussian Random Ellipsoid(GRE)形状モデルの生成
- `analytical_scattering_theories/` : Mie解（数値検証用）

## リファクタリングの目的
- mvp_fftにおけるFFTを用いた高速な行列ベクトル積の方法を Barrowesのアルゴリズムから → Goodmanのアルゴリズムへ置き換え
- bl_krylovにおける並列化の方法を ray → NumPyまたはScipyのbroadcasting (自動vectorization) に変更
- Python 3.13 対応
- 物理モデル、入力する物理パラメータ、出力される物理パラメータはいずれも変更なし

## リファクタリングの参考資料
-設計方針の詳細は .claude/design_notes.md を参照。
ただし記載情報とコードは参考例として用いるにとどめ、claude自身で独立に検討し正当性を自己点検すること。
- Goodmanアルゴリズム原論文: .claude/Goodman1991.pdf
ただし原論文は参考として用いるにとどめ、claude自身で独立に検討し正当性を自己点検すること。


## リファクタリング後のDDAコードの検証方法
- 単発計算用 test_dda.ipynb を検証に使用する。
- test_dda.ipynb に記載の入力パラメータ条件でのDDA解を、球形粒子についてのMie解(関数 mie_compute_q_and_sの出力）と比較する。
- DDA解のMie解に対する許容誤差は、test_dda.ipynbのセル内のコメントに記述してある。

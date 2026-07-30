# 入力画像の出所

本レポートの実験で用いた入力画像は、著作権上の問題が生じない2系統のみである。

## 1. scikit-image 同梱データ (`skimage.data`)

scikit-image (BSD-3-Clause) に同梱され、再配布が許諾されている画像。
本リポジトリには画像そのものを含めず、`dataset.py` が実行時に `skimage.data` から読み込む。
このうち `holdout_images()` として取り分けた 11 点は、改善とその適用条件を作るのには
使わず、第 6 節の検証だけに用いている（`skimage.data.cat` は `chelsea` のエイリアスで
同一画像なので held-out に入れていない）。

**本レポートの図1に写っているのは camera（CC0）とその生成結果である。**
バージョンは実験に用いた scikit-image 0.22.0 に準拠する。

| 識別子 | 出所 |
|---|---|
| camera | **CC0**（撮影者 Lav Varshney）。scikit-image 0.18 で、著作権上の懸念があった従来の cameraman から差し替えられたもの（[issue #3927](https://github.com/scikit-image/scikit-image/issues/3927)）。**本レポートの図1はこの画像を使っている** |
| coffee, chelsea | No copyright restrictions（scikit-image の記載） |
| rocket | public domain |
| astronaut, hubble_deep_field | NASA 提供のパブリックドメイン |
| cell, retina, grass | CC0 |
| brick, gravel | CC0Textures |
| clock | public domain（撮影者 Stefan van der Walt） |
| coins | Brooklyn Museum Collection、既知の著作権制限なし |
| immunohistochemistry | scikit-image 同梱の顕微鏡画像 |
| shepp_logan_phantom, binary_blobs, checkerboard, logo | scikit-image が生成する合成画像 |
| moon, page, horse | scikit-image 同梱のテスト画像 |

各画像の詳細な帰属は scikit-image の公式ドキュメント
<https://scikit-image.org/docs/stable/api/skimage.data.html> に従う。

## 2. 合成画像

`dataset.py: synthetic_images()` が OpenCV で生成する。
円と矩形の反復数だけを変え、エッジ密度を独立変数として制御するために本レポート用に作成した。
第三者の著作物を一切含まない。

## 生成画像について

Stable Diffusion v1.5 および sd-controlnet-canny の重みは CreativeML OpenRAIL-M ライセンスで公開されており、
研究・教育目的での利用が認められている。生成物は本レポートの実験結果としてのみ用いる。

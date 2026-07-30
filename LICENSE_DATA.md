# 入力画像の出所

本レポートの実験で用いた入力画像は、著作権上の問題が生じない2系統のみである。

## 1. scikit-image 同梱データ (`skimage.data`)

scikit-image (BSD-3-Clause) に同梱され、再配布が許諾されている画像。
本リポジトリには画像そのものを含めず、`dataset.py` が実行時に `skimage.data` から読み込む。
下表の後半 8 点は `holdout_images()` として取り分けたもので、改善とその適用条件を
作るのには一切使わず、第 6 節の検証だけに用いている。

| 識別子 | 出所 |
|---|---|
| astronaut | NASA 提供のパブリックドメイン画像（Eileen Collins） |
| camera | 古典的なテスト画像（cameraman） |
| coffee, chelsea, rocket | scikit-image が CC0 相当で同梱 |
| coins, page, horse | scikit-image 同梱のテスト画像 |
| immunohistochemistry, cell | scikit-image 同梱の顕微鏡画像 |
| brick, grass, gravel | scikit-image 同梱のテクスチャ画像 |
| cat, clock, moon | scikit-image 同梱のテスト画像 |
| retina | scikit-image 同梱の眼底画像 |
| hubble_deep_field | NASA/STScI 提供のパブリックドメイン画像 |

各画像の詳細な帰属は scikit-image の公式ドキュメント
<https://scikit-image.org/docs/stable/api/skimage.data.html> に従う。

## 2. 合成画像

`dataset.py: synthetic_images()` が OpenCV で生成する。
円と矩形の反復数だけを変え、エッジ密度を独立変数として制御するために本レポート用に作成した。
第三者の著作物を一切含まない。

## 生成画像について

Stable Diffusion v1.5 および sd-controlnet-canny の重みは CreativeML OpenRAIL-M ライセンスで公開されており、
研究・教育目的での利用が認められている。生成物は本レポートの実験結果としてのみ用いる。

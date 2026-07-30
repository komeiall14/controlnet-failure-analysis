# ControlNet の制御が効かなくなる条件と、その機構

東京大学「映像メディア学」レポート課題の実験コード。

対象論文: Lvmin Zhang, Anyi Rao, Maneesh Agrawala. "Adding Conditional Control to
Text-to-Image Diffusion Models." **ICCV 2023 (main track)**, pp.3813-3824.
DOI: [10.1109/ICCV51070.2023.00355](https://doi.org/10.1109/ICCV51070.2023.00355)

## 何を示したか

ControlNet が期待どおりに働かなくなる条件を実験的に切り出し、
**UNet へ注入される残差を直接測る**ことでその理由を説明した。

| # | 失敗条件 | 要点 |
|---|---|---|
| 1 | 推奨されている省メモリ設定が沈黙したまま出力を壊す | `enable_attention_slicing()` を有効にすると UNet 順伝播で NaN。例外も警告も出ず黒画像が返る。fp16 限定で、分割の粒度には依存しない |
| 2 | 条件地図の密度が下がると制御が消える | 60条件中15条件でエッジ画素率が 0。本来指定したかった輪郭との一致が 0.7329 → 0.2391 へ落ちる。**壊れているのは制御であって生成ではない**（画像自体は普通に出る） |
| 3 | 推論時の既定値が構造整合に対して弱い | 既定 `controlnet_conditioning_scale=1.0` が最適な入力は 7件中 0件。ただしプロンプト追従と併せると妥当な妥協点 |

**機構の直接証拠**: ControlNet 単体を 1 回順伝播させ、UNet へ渡される残差の RMS を測定。
条件地図の密度と Spearman ρ=+0.875（密度>0 に限れば +0.669）。
残差を空間方向に分解すると、条件が消えた入力でも空間成分は 67.5% 残る一方、
大きさ自体が 3.3 分の 1 に縮む。**制御が消えるのは残差が消えるからではなく、
条件地図に由来する成分が縮むからである。**

**改善**: 目標密度への二分探索 + CLAHE、密度連動 scale。生成 1 枚あたりの計算コストは増えない。
低コントラスト側で構造整合が改善し（Δ中央値 +0.0665、共通参照）、プロンプト追従は下がらない。
密度が目標を下回るときだけ適用する形にすると p=0.0019 まで下がる。

## GPU なしで 5 分で検証する

`results/` に全実験の CSV を含めているので、**GPU も生成画像も無い環境で
レポート本文の全数値を再現できる**。

```bash
pip install numpy pandas scipy opencv-python scikit-image
python3 stats_report.py    # 本文の統計量をすべて再現する
python3 audit_numbers.py   # 本文の数値が手元のデータから出るか機械的に照合する
```

`stats_report.py` は本文の数値の出どころである。節ごとに、その節が主張する統計量を
出力する。生成画像（158MB、リポジトリに含めない）が必要な箇所は
`results/image_metrics.csv` に測定値をキャッシュしてあり、画像の有無で
同じ値になることを確認済み。実行時にどちらから計算したかを表示する。

`audit_numbers.py` は REPORT.md 中の数値を抜き出し、`stats_report.py` の出力か
`results/*.csv` に一致するかを照合する。どちらにも当たらない数値は「出所不明」として
一覧に出す。現在、出所不明はすべて非データ数値（学籍番号・DOI・ライブラリのバージョン等）
のみで、データ由来の数値はゼロである。

## 実験を最初から回す

```bash
pip install torch diffusers transformers accelerate safetensors
bash run_all.sh
```

各段は `results/<名前>.done` を作る。再実行すると完了済みの段はスキップされるので、
中断しても続きから再開できる。CSV は 1 行ごとに追記するため、途中で落ちても
そこまでの結果は残る。1 段が異常終了しても後段は続行し、ログに `★` 付きで記録する。

全段で約 12 時間（Apple M1）。生成は 1 枚あたり 76〜184 秒
（20ステップ・512×512、n=157、中央値 106.5 秒、平均 110.8 秒）。

## 実行環境

Apple M1（8コアGPU）/ RAM 16GB / macOS 14.4.1。CUDA なし、PyTorch 2.2.2 の MPS。
diffusers 0.31.0 / transformers 4.44.2 / OpenCV 4.11.0。
Stable Diffusion v1.5 + sd-controlnet-canny（fp16）。

**`enable_attention_slicing()` は使っていない。** 公式文書は RAM 64GB 未満で有効化を
推奨しているが、本環境では UNet の順伝播で NaN が発生し、例外を出さないまま黒画像が返る
（第4.1節）。`diag2.py` で潜在変数の段階の NaN を確認済み。無効にすると同一シードで正常に生成される。

サンプラは DPMSolver++ 2M。UniPC は補正段で `torch.linalg.solve` を呼び、MPS 未実装のため使えない。

## ファイル

### 実験

| | |
|---|---|
| `pipeline.py` | パイプライン構築、条件地図の生成、指標（edge F1 / Chamfer）。**生成直後に NaN と単色を検出して例外を投げる** |
| `dataset.py` | 入力画像。`skimage.data` 同梱 + 自作の合成画像のみ（出所は `LICENSE_DATA.md`） |
| `exp.py` | 各実験。`python3 exp.py {smoke,exp1,expvar,exp2,exp2b,exp3,exp4,exp5,exp6}` |
| `probe.py` | 機構の測定。ControlNet 単体を 1 回順伝播させ、UNet へ渡される残差の RMS を取る |
| `probe_spatial.py` | 残差を「チャネルごとに一様な成分」と「空間的に変化する成分」に分解する |
| `bench_dtype.py` | fp16 と fp32 の生成時間。ウォームアップ 1 回を捨てて 3 回測る |
| `exp7_policy.py` | 改善策にプロンプト追従（CLIP）を測り、推奨形（条件付き適用）を評価する |
| `clip_score.py` | 生成済み画像に CLIP スコアを後付けする（拡散を回さない） |
| `run_all.sh` | 全実験を無人で順に完走させる |

### 集計・検証

| | |
|---|---|
| `stats_report.py` | **本文に載せた統計量を再現する**。本文の数値の出どころ |
| `audit_numbers.py` | 本文の数値が手元のデータから出るかを機械的に照合する |
| `cache_image_metrics.py` | 生成画像から導く測定値を CSV に書き出す（画像を配布せずに検証できるようにするため） |
| `analysis.py` | 作図。`python3 analysis.py {exp1,exp2,exp3,probe,all}` |
| `check_health.py` | 全生成画像の健全性を一括検査（黙って壊れていないかの確認） |
| `diag.py` `diag2.py` | 黒画像の原因切り分け。潜在変数の NaN と attention slicing の関与を特定 |
| `make_pdf.py` | 提出用 PDF の生成と検証（欠字ゼロ・マークアップ漏れ・ページ数） |

### 主な出力

| | |
|---|---|
| `results/exp1_density.csv` | 密度と構造整合（60条件） |
| `results/exp2_scale.csv` `exp2b_scale_ext.csv` | scale 掃引（7入力 × 10段階） |
| `results/exp3_improvement.csv` | 改善前後（15画像 × 2水準 × 3シード = 90対） |
| `results/exp4_conflict.csv` | 条件とプロンプトの衝突（5画像 × 4プロンプト） |
| `results/exp5_slicing.csv` | attention slicing の NaN 格子（2精度 × 5設定） |
| `results/exp6_depth_generality.csv` | 深度条件での残差（10画像 × 7水準） |
| `results/probe_residual.csv` | 残差の測定（順伝播136回） |
| `results/probe_spatial.csv` | 残差の空間分解 |
| `results/exp7_policy.csv` | 改善策の CLIP と共通参照での F1 |
| `results/image_metrics.csv` | 生成画像から導いた測定値のキャッシュ |

## 注意している点

- **静かに壊れる失敗を検出する**: `pipeline.generate()` は生成直後に NaN / Inf と単色を
  検出して例外を投げる。第4.1節の失敗は例外も警告も出ないため、これが無いと
  下流のすべての指標が無意味になる
- **指標が定義から返す値を測定結果として扱わない**: 条件地図が空だと edge F1 は
  生成画像が何であっても 0 を返す。この 0 を破綻の証拠にはしていない（第4.2節）
- **独立でない観測を独立として数えない**: 順伝播の回数・1画像から出る複数行・
  総当たりの組み合わせは、いずれも独立標本ではない。検定の単位を本文に明記している

## 入力画像の出所

`LICENSE_DATA.md` に記載。第三者の著作物は含まない。

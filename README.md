# ControlNet の制御が効かなくなる条件と、その機構

東京大学「映像メディア学」レポート課題の実験コード。
対象論文: Zhang, Rao, Agrawala. "Adding Conditional Control to Text-to-Image
Diffusion Models." ICCV 2023, pp.3813-3824. DOI: 10.1109/ICCV51070.2023.00355

## 何を調べたか

ControlNet の制御が期待どおりに働かなくなる条件を実験的に切り出し、
UNet へ注入される残差を直接測ることでその理由を説明した。

1. 推奨されている省メモリ設定 `enable_attention_slicing()` が、例外を出さないまま NaN を発生させ黒画像を返す
2. 条件地図のエッジ密度が下がると制御が消える（指定した輪郭との一致が偶然の水準まで落ちる）
3. 推論時の既定 `controlnet_conditioning_scale=1.0` は構造整合に対して弱すぎる

## 実行環境

Apple M1（8コアGPU）/ RAM 16GB / macOS 14.4.1。CUDA なし、PyTorch 2.2.2 の MPS。
diffusers 0.31.0 / transformers 4.44.2 / OpenCV 4.11.0。
Stable Diffusion v1.5 ＋ sd-controlnet-canny（fp16）。

**`enable_attention_slicing()` は使っていない。** 公式文書は 64GB 未満の RAM で有効化を
推奨しているが、本環境では UNet の順伝播で NaN が発生し、例外を出さないまま黒画像が返る。
`diag2.py` で潜在変数の段階の NaN を確認済み。無効にすると同一シードで正常に生成される。

サンプラは DPMSolver++ 2M。UniPC は補正段で `torch.linalg.solve` を呼び、MPS 未実装のため使えない。

## ファイル

| | |
|---|---|
| `pipeline.py` | パイプライン構築、条件地図の生成、指標（edge F1 / Chamfer）。生成直後に NaN と単色を検出して例外を投げる |
| `dataset.py` | 入力画像。`skimage.data` 同梱＋自作の合成画像のみ（出所は `LICENSE_DATA.md`） |
| `exp.py` | 各実験。`python3 exp.py {smoke,exp1,expvar,exp2,exp2b,exp3,exp4,exp5,exp6}` |
| `probe.py` | 機構の測定。ControlNet 単体を1回順伝播させ UNet へ渡される残差の RMS を取る |
| `analysis.py` | 統計処理と作図。`python3 analysis.py {exp1,exp2,exp3,probe,all}` |
| `clip_score.py` | 生成済み画像に CLIP スコア（プロンプト追従）を後付けする |
| `check_health.py` | 全生成画像の健全性を一括検査（黙って壊れていないかの確認） |
| `diag.py` `diag2.py` | 黒画像の原因切り分け。潜在変数の NaN と attention slicing の関与を特定 |
| `make_pdf.py` | 提出用 PDF の生成と検証（欠字ゼロ・マークアップ漏れ・ページ数） |
| `run_all.sh` | 全実験を無人で順に完走させる |

## 実験の再現

```bash
bash run_all.sh
```

各段は `results/<名前>.done` を作る。再実行すると完了済みの段はスキップされるので、
中断しても続きから再開できる。CSV は1行ごとに追記するため、途中で落ちてもそこまでの結果は残る。
1段が異常終了しても後段は続行し、ログに `★` 付きで記録する。

生成は 1 枚あたり 101〜184 秒（20ステップ・512×512、n=129、中央値 106.8 秒）。
全段で約 12 時間かかる。

## 入力画像の出所

`LICENSE_DATA.md` に記載。第三者の著作物は含まない。

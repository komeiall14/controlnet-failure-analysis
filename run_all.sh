#!/bin/bash
# 全実験を無人で順に完走させる。
# 各段は CSV を1行ごとに追記するため、途中で落ちてもそこまでの結果は残る。
cd "$(dirname "$0")" || exit 1
L=results/run_all.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

say "=== run_all 開始 ==="

run() {  # run <名前> <引数>
  local name=$1 arg=$2
  if [ -f "results/${name}.done" ]; then say "$name は完了済みのためスキップ"; return; fi
  say "--- $name 開始 ---"
  # $arg は "exp.py expvar" のように引数を含むので、あえてクォートしない。
  # クォートすると全体が1つのファイル名として解釈され、起動に失敗する。
  # shellcheck disable=SC2086
  if python3 -u $arg >> "results/${name}.log" 2>&1; then
    touch "results/${name}.done"
    say "--- $name 完了 ---"
  else
    say "★ $name が異常終了 (exit=$?)。ログ: results/${name}.log"
    tail -5 "results/${name}.log" | tee -a "$L"
    # 1段が落ちても後段は続ける。全滅を避けるため
  fi
}

# 疎通確認。2枚。第3節の速度統計（n=157）はこの2枚も数える
run smoke "exp.py smoke"
# 機構の直接証拠。数分で終わり最も価値が高いので先に取る
run probe "probe.py"
# 第4.2節の中核。15画像×4コントラストの60条件（約2時間）
run exp1 "exp.py exp1"
# 条件内のシード分散。主張が分散に埋もれていないことの担保
run expvar "exp.py expvar"
# conditioning_scale の感度
run exp2 "exp.py exp2"
# 改善前後比較（最も重い）
run exp3 "exp.py exp3"
# scale掃引の延長。exp2 で全件が上限1.3に張り付き、範囲が狭すぎたと判明したため
run exp2b "exp.py exp2b"
# 追加の失敗条件。既存の重みだけで回るのでディスクを増やさない
run exp5 "exp.py exp5"    # slicing NaN の発生条件を地図化（軽い。1条件20秒）
run exp4 "exp.py exp4"    # 条件とプロンプトの意味的衝突（20枚）
# 予備測定。depth条件でも同じ残差の縮小が起きるか（拡散を回さないので数分）
run exp6 "exp.py exp6"

# --- 第4.1節：NaN の原因の切り分けと修正の検証 ---
# NaN がどの条件で出るか（精度・粒度・加算元の初期化を振る）
run slicingcause "exp8_slicing_cause.py"
# 加算元バッファの中身を直接見る（拡散4ステップ×2精度）
run buffer "exp10_buffer_content.py"
# baddbmm が beta=0 で加算元を無視するか。モデルを読まないので数秒
run beta0 "exp11_beta_zero.py"
# 修正が本番条件（20ステップ）で実用になるか
run fixusable "exp12_fix_usable.py"
# fp16 と fp32 の生成時間。ウォームアップ1回を捨てて3回
run bench "bench_dtype.py"
# 修正版が省メモリ効果を保つか（生成中の確保量のピーク）
run memory "exp13_memory.py"

# --- 第5節：残差の空間分解 ---
run spatial "probe_spatial.py"

# --- 第6節：改善の評価 ---
# 生成済み画像に CLIP を後付けし、推奨形（条件付き適用）を評価する
run policy "exp7_policy.py"
# 開発に使っていない画像での検証（最も重い。約6時間）
run holdout "exp9_holdout.py"

# --- 集計の下ごしらえ ---
# 生成画像から導く測定値をキャッシュする（画像を配布せずに検証できるように）
run cache "cache_image_metrics.py"

say "--- 健全性検査 ---"
python3 check_health.py 2>&1 | tee -a "$L"

say "--- CLIP 付与 ---"
python3 clip_score.py exp1_density exp2_scale exp2b_scale_ext expvar_seed exp4_conflict >> "$L" 2>&1 || say "CLIP付与でエラー"

say "--- 作図と統計 ---"
python3 analysis.py all 2>&1 | tee -a "$L"

say "=== run_all 完了 ==="
ls -1 results/*.csv 2>/dev/null | tee -a "$L"

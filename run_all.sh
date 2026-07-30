#!/bin/bash
# 全実験を無人で順に完走させる。
# API の利用上限とは無関係に動くので、離席中でも GPU 作業は止まらない。
# 各段は CSV を1行ごとに追記するため、途中で落ちてもそこまでの結果は残る。
cd "$(dirname "$0")" || exit 1
L=results/run_all.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }

say "=== run_all 開始 ==="

# 既に走っている exp1 の完了を待つ
if pgrep -f "exp.py exp1" > /dev/null; then
  say "exp1 実行中。完了を待機"
  while pgrep -f "exp.py exp1" > /dev/null; do sleep 30; done
  say "exp1 終了 ($(grep -c 'f1=' results/exp1.log 2>/dev/null)枚)"
fi

run() {  # run <名前> <引数>
  local name=$1 arg=$2
  if [ -f "results/${name}.done" ]; then say "$name は完了済みのためスキップ"; return; fi
  say "--- $name 開始 ---"
  # $arg は "exp.py expvar" のように引数を含むので、あえてクォートしない。
  # クォートすると全体が1つのファイル名として解釈され即死する（実際にやった）。
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

# 機構の直接証拠。数分で終わり最も価値が高いので先に取る
run probe "probe.py"
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
# 一般化の検証。depth条件でも同じ残差の縮小が起きるか（拡散を回さないので数分）
run exp6 "exp.py exp6"

say "--- 健全性検査 ---"
python3 check_health.py 2>&1 | tee -a "$L"

say "--- CLIP 付与 ---"
python3 clip_score.py exp1_density exp2_scale exp2b_scale_ext expvar_seed exp4_conflict >> "$L" 2>&1 || say "CLIP付与でエラー"

say "--- 作図と統計 ---"
python3 analysis.py all 2>&1 | tee -a "$L"

say "=== run_all 完了 ==="
ls -1 results/*.csv 2>/dev/null | tee -a "$L"

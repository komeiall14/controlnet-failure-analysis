#!/bin/bash
# 現在走っている run_all.sh は起動時のスクリプトを読んでいるため exp2b を知らない。
# その完走を待ってから、更新版の run_all.sh を再実行する。
# .done マーカーがあるので既に終わった段はスキップされ、exp2b だけが走る。
cd "$(dirname "$0")" || exit 1
while pgrep -f "bash run_all" > /dev/null; do sleep 60; done
echo "[$(date '+%m-%d %H:%M:%S')] 先行チェーン完了。更新版を再実行（exp2b と後処理）" >> results/run_all.log
exec bash run_all.sh

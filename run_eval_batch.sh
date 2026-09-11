#!/bin/bash
# 内存受限环境的解析续跑：每批 N 份一个独立进程（进程退出即释放内存），
# 断点续跑（docs 里已有 json 的自动跳过）；硬失败文档记账防死循环。
# 背景：run_eval_prep.sh 单进程连跑 94 份会累积内存峰值被系统击杀
# （2026-09-11 实测 25/106 处被杀）。
# 解析全部完成后，序列化/切分/索引等阶段不吃内存，
# 直接再跑 bash run_eval_prep.sh（其解析阶段会整段跳过）。
# 用法：bash run_eval_batch.sh [批大小，默认 5]
set -u
cd /d/workspace/enterprise-rag
export PYTHONIOENCODING=utf-8 MINERU_MODEL_SOURCE=modelscope
PY=.venv/Scripts/python.exe
BATCH=${1:-5}
FAILED=data/parsed/parse_failed.txt
touch "$FAILED"

stall=0
while true; do
    batch=()
    for pdf in data/raw/eval/*.pdf; do
        name=$(basename "$pdf" .pdf)
        [ -f "data/parsed/docs/$name.json" ] && continue
        grep -qxF "$pdf" "$FAILED" && continue
        batch+=("$pdf")
        [ ${#batch[@]} -ge $BATCH ] && break
    done
    if [ ${#batch[@]} -eq 0 ]; then
        echo "=== 解析全部完成 $(date) ==="
        break
    fi
    before=$(ls data/parsed/docs/*.json 2>/dev/null | wc -l)
    echo "=== 批（${#batch[@]} 份）$(date)：${batch[*]} ==="
    $PY -u -m enterprise_rag.parsing.pdf_parser "${batch[@]}" \
        || echo "[WARN] 批退出码非零，继续下一批"
    after=$(ls data/parsed/docs/*.json 2>/dev/null | wc -l)
    if [ "$after" -le "$before" ]; then
        stall=$((stall + 1))
        echo "[WARN] 本批无进展（$stall/3）"
        if [ $stall -ge 3 ]; then
            echo "[ABORT] 连续 3 批无进展，退出（人工检查后删 $FAILED 可重试）"
            exit 1
        fi
        sleep 20
    else
        stall=0
        # 本批没产出 json 的记为失败（下轮跳过；想重试就删 FAILED 里的行）
        for pdf in "${batch[@]}"; do
            name=$(basename "$pdf" .pdf)
            [ -f "data/parsed/docs/$name.json" ] || grep -qxF "$pdf" "$FAILED" \
                || echo "$pdf" >> "$FAILED"
        done
    fi
    sleep 5
done

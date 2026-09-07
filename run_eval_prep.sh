#!/bin/bash
# eval 100 份全量数据准备：解析 -> 表格序列化 -> 图片序列化 -> 切分 -> 建索引
# 全部可断点续跑；任何一步失败链条停止
set -e
cd /d/workspace/enterprise-rag
export PYTHONIOENCODING=utf-8 MINERU_MODEL_SOURCE=modelscope
PY=.venv/Scripts/python.exe

echo "=== [1/5] PDF 解析 $(date) ==="
$PY -u -m enterprise_rag.parsing.pdf_parser data/raw/eval
echo "=== [2/5] 表格序列化 $(date) ==="
$PY -u -m enterprise_rag.processing.table_serializer
echo "=== [3/5] 图片序列化 $(date) ==="
$PY -u -m enterprise_rag.processing.image_serializer
echo "=== [4/5] 文本切分 $(date) ==="
$PY -u -m enterprise_rag.processing.text_splitter
echo "=== [5/5] 建索引 $(date) ==="
$PY -u -m enterprise_rag.indexing.ingestor
echo "=== 全部完成 $(date) ==="

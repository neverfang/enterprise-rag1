"""评测：answers_run.json 对 answers_eval.json 标准答案算准确率，与获奖基线对比。

自用评分规则（官方评测另有一套，这里是开发迭代用的快速反馈）：
- number：双方数值化后相对误差 <=1% 视为对（千分位逗号/百分号/括号负数均
  规范化）；'N/A' == 'N/A' 也算对（不可答题考的就是这个）
- boolean：真值等价
- name：规范化（小写/去标点/压缩空白）后相等或互为子串
- names：名单集合相等（宽松口径：标准名单 ⊆ 预测名单，单独统计）
- 标准答案 answers 是字符串形式的 list（可能多个可接受答案），命中任一即对
- N/A 题单独统计——45/100 是故意不可答，乱答会双倍扣分（对不上了还丢了 N/A 分）

用法：
    python -m enterprise_rag.evaluate <answers_run.json> \
        [--gt data/answers_eval.json] [--baseline data/answers_baseline_o3mini.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _norm_text(s) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s)


def _num(v):
    """数值化；括号包裹视为负数；失败返回 None。"""
    s = str(v).strip().replace(",", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").rstrip("%")
    try:
        x = float(s)
        return -x if neg else x
    except ValueError:
        return None


def _to_bool(v) -> bool | None:
    s = str(v).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return None


def _is_na(v) -> bool:
    return _norm_text(v) in ("n a", "na", "none", "not available", "")


def match_one(pred, gt, kind: str) -> bool:
    """单个预测对单个可接受答案的匹配。"""
    if kind == "number":
        if _is_na(pred) or _is_na(gt):
            return _is_na(pred) and _is_na(gt)
        a, b = _num(pred), _num(gt)
        return a is not None and b is not None and (
            abs(a - b) <= 0.01 * max(abs(a), abs(b), 1e-9) or a == b)
    if kind == "boolean":
        pa, gb = _to_bool(pred), _to_bool(gt)
        if pa is None or gb is None:
            return _norm_text(pred) == _norm_text(gt)
        return pa == gb
    if kind == "name":
        p, g = _norm_text(pred), _norm_text(gt)
        return bool(p) and bool(g) and (p == g or p in g or g in p)
    if kind == "names":
        pl = pred if isinstance(pred, list) else [pred]
        gl = gt if isinstance(gt, list) else [gt]
        return {_norm_text(x) for x in pl} == {_norm_text(x) for x in gl}
    return _norm_text(pred) == _norm_text(gt)


def score(pred, gt_answers: list, kind: str) -> bool:
    return any(match_one(pred, g, kind) for g in gt_answers)


def parse_answers(s) -> list:
    """gt 的 answers 字段是字符串形式的 list，引号风格不一（直/弯/混合），
    逐级降级解析，最后兜底手工抽取引号段。"""
    if isinstance(s, list):
        return s
    for fn in (ast.literal_eval,
               lambda t: json.loads(t.replace("'", '"')),
               lambda t: json.loads(
                   t.replace("‘", "'").replace("’", "'").replace("'", '"'))):
        try:
            v = fn(s)
            if isinstance(v, list):
                return v
        except Exception:  # noqa: BLE001 —— 降级链就是用来试错的
            pass
    parts = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
    if parts:
        return [a or b for a, b in parts]
    return [s.strip("[]'\" ")]


def evaluate(name: str, preds: dict[str, tuple], gt: dict) -> dict:
    """preds: 题目文本 -> (final_answer, kind)；返回分组统计并打印。"""
    rows = []
    for q, meta in gt.items():
        if q not in preds:
            continue
        pred, _ = preds[q]
        gts = parse_answers(meta["answers"])
        ok = score(pred, gts, meta["kind"])
        rows.append({"q": q, "kind": meta["kind"], "ok": ok, "pred": pred,
                     "gt": gts, "na": all(_is_na(g) for g in gts)})
    if not rows:
        print(f"[{name}] 没有可对齐的题目（run 与 gt 无交集）")
        return {}

    def report(rs: list[rows], tag: str) -> float:
        acc = sum(r["ok"] for r in rs) / len(rs) if rs else 0.0
        print(f"  {tag:<24} {sum(r['ok'] for r in rs):3d}/{len(rs):<3d} {acc:6.1%}")
        return acc

    print(f"\n== {name}（可对齐 {len(rows)} 题） ==")
    report(rows, "总体")
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for k in sorted(by):
        report(by[k], f"  {k}")
    report([r for r in rows if r["na"]], "  N/A 题（不编造）")
    report([r for r in rows if not r["na"]], "  非 N/A 题")
    wrong = [r for r in rows if not r["ok"]]
    if wrong:
        print("  -- 错题（前 10）--")
        for r in wrong[:10]:
            print(f"    [{r['kind']}] {r['q'][:70]}")
            print(f"      pred={str(r['pred'])[:80]!r} gt={str(r['gt'])[:80]}")
    return {"rows": rows}


def load_run(path: Path) -> dict[str, tuple]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {a["question"]: (a.get("final_answer"), a.get("kind")) for a in data}


def main() -> None:
    ap = argparse.ArgumentParser(description="第 9 步：答案评测（对标准答案 + 基线）")
    ap.add_argument("answers", type=Path)
    ap.add_argument("--gt", type=Path, default=DATA_DIR / "answers_eval.json")
    ap.add_argument("--baseline", type=Path,
                    default=DATA_DIR / "answers_baseline_o3mini.json")
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    evaluate("本项目", load_run(args.answers), gt)

    if args.baseline and args.baseline.exists():
        raw = json.loads(args.baseline.read_text(encoding="utf-8"))
        preds = {a["question_text"]: (a.get("value"), a.get("kind"))
                 for a in raw["answers"]}
        evaluate("基线 o3-mini（获奖提交）", preds, gt)


if __name__ == "__main__":
    main()

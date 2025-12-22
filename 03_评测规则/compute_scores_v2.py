#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用途：
- 汇总 cross_model_evaluation_results 下的 JSON 评测结果，输出 CSV（主方法 + 补充方法）
- 主方法(main_method)：judge_{judge}_bundle_{generator}_v2.json（跨模型互评）
- 补充方法(supporting_method)：ChatGPT.json / Claude.json / Gemini.json（自评）
- 同时支持把 invalid_results 一起纳入（bundle 不合规也会出 validity 表）

计分口径：
- 每个 PDF 一条 per_file_scores
- 每条有五维 A~E，每维 0/1/2，总分 total = A+B+C+D+E（0~10）
- 本脚本不“重新打分”，只做：读取→校验→汇总→导出
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

DIM_KEYS = ["A_structure", "B_snapshot_constraint", "C_actionability", "D_completeness", "E_drift_failure"]

# per_file_scores[i]["file"] 形如：q3 baseline explicit.pdf
FILE_RE = re.compile(r"^(q[34])\s+(baseline|long|weak|conflict)\s+(implicit|explicit)\.pdf$", re.IGNORECASE)

# 主方法 bundle 文件名：judge_chatgpt_bundle_claude_v2.json
JUDGE_BUNDLE_RE = re.compile(r"^judge_(.+)_bundle_(.+)_v2\.json$", re.IGNORECASE)


def norm_name(x: str) -> str:
    s = (x or "").strip().lower()
    if any(k in s for k in ["openai", "gpt", "chatgpt"]):
        return "chatgpt"
    if any(k in s for k in ["google", "gemini"]):
        return "gemini"
    if any(k in s for k in ["anthropic", "claude"]):
        return "claude"
    s = re.sub(r"\s+", "_", s)
    return s or "unknown"


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_files(base_dir: str, recursive: bool) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(base_dir):
        return out

    if recursive:
        for root, _, files in os.walk(base_dir):
            for fn in files:
                if fn.lower().endswith(".json"):
                    out.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(base_dir):
            if fn.lower().endswith(".json"):
                out.append(os.path.join(base_dir, fn))
    return sorted(out)


def protocol_validity(bundle: Dict) -> Tuple[bool, List[str]]:
    """
    只做“硬性合规”的最小校验（为了能统计 invalid 的偏移情况）
    """
    reasons: List[str] = []

    if not isinstance(bundle, dict):
        return False, ["not_a_dict"]

    if "bundle_meta" not in bundle or not isinstance(bundle.get("bundle_meta"), dict):
        reasons.append("missing_or_bad_bundle_meta")

    if "per_file_scores" not in bundle or not isinstance(bundle.get("per_file_scores"), list):
        reasons.append("missing_or_bad_per_file_scores")
        return False, reasons

    pfs = bundle.get("per_file_scores", [])
    if len(pfs) == 0:
        reasons.append("per_file_scores_empty")
        return False, reasons

    # 抽查全部（数据量不大）
    for i, item in enumerate(pfs):
        if not isinstance(item, dict):
            reasons.append(f"per_file_scores[{i}].not_a_dict")
            continue

        sc = item.get("scores")
        if not isinstance(sc, dict):
            reasons.append(f"per_file_scores[{i}].missing_or_bad_scores")
            continue

        for k in DIM_KEYS:
            if k not in sc:
                reasons.append(f"per_file_scores[{i}].missing_{k}")
                continue
            v = sc.get(k)
            if not isinstance(v, int):
                reasons.append(f"per_file_scores[{i}].{k}_not_int")
            elif v < 0 or v > 2:
                reasons.append(f"per_file_scores[{i}].{k}_out_of_range")

        t = item.get("total")
        if t is not None and (not isinstance(t, int)):
            reasons.append(f"per_file_scores[{i}].total_not_int")

    return (len(reasons) == 0), reasons


def parse_main_pair_from_filename(path: str) -> Optional[Tuple[str, str]]:
    fn = os.path.basename(path)
    m = JUDGE_BUNDLE_RE.match(fn)
    if not m:
        return None
    judge = norm_name(m.group(1))
    generator = norm_name(m.group(2))
    return judge, generator


def parse_support_pair_from_filename_or_meta(path: str, bundle: Dict) -> Tuple[str, str]:
    """
    supporting_method：优先用文件名 ChatGPT/Claude/Gemini，否则回退 bundle_meta.model
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    stem_n = norm_name(stem)
    if stem_n in {"chatgpt", "claude", "gemini"}:
        return stem_n, stem_n

    meta_model = norm_name(bundle.get("bundle_meta", {}).get("model", ""))
    if meta_model in {"chatgpt", "claude", "gemini"}:
        return meta_model, meta_model

    # 兜底：当成 unknown 自评（仍然让它出现在 validity 表里）
    return stem_n or "unknown", stem_n or "unknown"


def rows_from_bundle(path: str, method: str, judge: str, generator: str, bundle: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for item in bundle.get("per_file_scores", []):
        file_name = item.get("file", "")
        m = FILE_RE.match(file_name)
        if not m:
            # 允许少数非标准文件名仍能落表（但标 unknown）
            q, ver, trig = "unknown", "unknown", "unknown"
        else:
            q, ver, trig = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()

        sc = item.get("scores", {}) if isinstance(item.get("scores"), dict) else {}
        row = {
            "method": method,
            "judge": judge,
            "generator": generator,
            "file": file_name,
            "question": q,
            "version": ver,
            "trigger": trig,
            "total": item.get("total", sum(int(sc.get(k, 0)) for k in DIM_KEYS)),
            "notes": item.get("notes", ""),
            "path": path,
        }
        for k in DIM_KEYS:
            row[k] = sc.get(k, None)
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame, out_prefix: str, out_dir: str) -> None:
    if df.empty:
        # 仍然写空表，避免你误判“没输出”
        df.to_csv(os.path.join(out_dir, f"{out_prefix}_by_row.csv"), index=False, encoding="utf-8-sig")
        return

    df.to_csv(os.path.join(out_dir, f"{out_prefix}_by_row.csv"), index=False, encoding="utf-8-sig")

    # by_pair：judge x generator
    def _mean_on(mask: pd.Series, s: pd.Series) -> float:
        sub = s[mask.reindex(s.index, fill_value=False)]
        return float(sub.mean()) if len(sub) else float("nan")

    g = df.groupby(["judge", "generator"], dropna=False)
    by_pair = g.agg(
        n=("total", "size"),
        avg_total=("total", "mean"),
        perfect_count=("total", lambda x: int((x == 10).sum())),
        zero_count=("total", lambda x: int((x == 0).sum())),
    ).reset_index()

    # explicit/implicit 平均
    trig = df["trigger"].astype(str).str.lower()
    by_pair["explicit_avg"] = by_pair.apply(
        lambda r: _mean_on((df["judge"] == r["judge"]) & (df["generator"] == r["generator"]) & (trig == "explicit"), df["total"]),
        axis=1,
    )
    by_pair["implicit_avg"] = by_pair.apply(
        lambda r: _mean_on((df["judge"] == r["judge"]) & (df["generator"] == r["generator"]) & (trig == "implicit"), df["total"]),
        axis=1,
    )

    by_pair.to_csv(os.path.join(out_dir, f"{out_prefix}_by_pair.csv"), index=False, encoding="utf-8-sig")

    # by_generator：只按 generator 汇总（跨 judge）
    gg = df.groupby(["generator"], dropna=False)
    by_gen = gg.agg(
        n=("total", "size"),
        avg_total=("total", "mean"),
        perfect_count=("total", lambda x: int((x == 10).sum())),
        zero_count=("total", lambda x: int((x == 0).sum())),
    ).reset_index()

    by_gen["explicit_avg"] = by_gen.apply(
        lambda r: _mean_on((df["generator"] == r["generator"]) & (trig == "explicit"), df["total"]),
        axis=1,
    )
    by_gen["implicit_avg"] = by_gen.apply(
        lambda r: _mean_on((df["generator"] == r["generator"]) & (trig == "implicit"), df["total"]),
        axis=1,
    )

    by_gen.to_csv(os.path.join(out_dir, f"{out_prefix}_by_generator.csv"), index=False, encoding="utf-8-sig")

    # inter-judge agreement（只对 main_method 有意义；supporting 自评不需要）
    if out_prefix == "main_method":
        key_cols = ["generator", "question", "version", "trigger"]
        piv = df.pivot_table(index=key_cols, columns="judge", values="total", aggfunc="mean")

        # 每个 generator：两两 judge 的 mean_abs_diff / exact_match_rate
        pair_rows: List[Dict] = []
        agg_rows: List[Dict] = []

        gens = sorted({idx[0] for idx in piv.index})
        for gen_name in gens:
            sub = piv.loc[gen_name]  # index: (question,version,trigger), columns: judge

            diffs = []
            exacts = []
            ranges = []

            # 逐条 item：拿到所有 judge 的分数
            for _, row in sub.iterrows():
                vals = row.dropna().values.tolist()
                if len(vals) < 2:
                    continue
                # overall range
                ranges.append(max(vals) - min(vals))
                # pairwise（一般这里就是 2 个 judge，但写成通用）
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        d = abs(vals[i] - vals[j])
                        diffs.append(d)
                        exacts.append(1 if d == 0 else 0)

            agg_rows.append({
                "generator": gen_name,
                "n_items_with_2plus_judges": int(sub.dropna(axis=0, how="all").shape[0]),
                "mean_range": float(pd.Series(ranges).mean()) if ranges else float("nan"),
                "mean_abs_diff": float(pd.Series(diffs).mean()) if diffs else float("nan"),
                "exact_match_rate": float(pd.Series(exacts).mean()) if exacts else float("nan"),
            })

            # judge-pair 明细（按列名两两）
            judges = [c for c in sub.columns if sub[c].notna().any()]
            for i in range(len(judges)):
                for j in range(i + 1, len(judges)):
                    j1, j2 = judges[i], judges[j]
                    sub2 = sub[[j1, j2]].dropna()
                    if sub2.empty:
                        continue
                    diff = (sub2[j1] - sub2[j2]).abs()
                    pair_rows.append({
                        "generator": gen_name,
                        "judge_1": j1,
                        "judge_2": j2,
                        "n": int(len(sub2)),
                        "mean_abs_diff": float(diff.mean()),
                        "exact_match_rate": float((diff == 0).mean()),
                    })

        pd.DataFrame(agg_rows).to_csv(
            os.path.join(out_dir, "main_method_inter_judge_agreement.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(pair_rows).to_csv(
            os.path.join(out_dir, "main_method_inter_judge_pairwise.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_dir", action="append", default=[], help="主方法目录（可重复传入多个）")
    ap.add_argument("--support_dir", action="append", default=[], help="补充方法目录（可重复传入多个）")
    ap.add_argument("--out_dir", required=True, help="输出目录（CSV 会写到这里）")
    ap.add_argument("--recursive", action="store_true", help="递归扫描目录")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bundle_validity_rows: List[Dict] = []
    main_rows: List[Dict] = []
    support_rows: List[Dict] = []

    # -------- main_method --------
    for d in args.main_dir:
        for path in iter_json_files(d, args.recursive):
            pair = parse_main_pair_from_filename(path)
            if pair is None:
                continue
            judge, generator = pair
            bundle = load_json(path)
            ok, reasons = protocol_validity(bundle)

            bundle_validity_rows.append({
                "method": "main_method",
                "judge": judge,
                "generator": generator,
                "file": os.path.basename(path),
                "path": path,
                "is_valid": int(ok),
                "reasons": "|".join(reasons),
            })

            if ok:
                main_rows.extend(rows_from_bundle(path, "main_method", judge, generator, bundle))

    # -------- supporting_method --------
    for d in args.support_dir:
        for path in iter_json_files(d, args.recursive):
            # supporting_method 只收“自评”文件：ChatGPT/Claude/Gemini 或 meta 可判别的 json
            bundle = load_json(path)
            judge, generator = parse_support_pair_from_filename_or_meta(path, bundle)

            # 允许 unknown 也落到 validity 表，便于排查目录里混入了什么
            ok, reasons = protocol_validity(bundle)
            bundle_validity_rows.append({
                "method": "supporting_method",
                "judge": judge,
                "generator": generator,
                "file": os.path.basename(path),
                "path": path,
                "is_valid": int(ok),
                "reasons": "|".join(reasons),
            })

            if ok:
                support_rows.extend(rows_from_bundle(path, "supporting_method", judge, generator, bundle))

    # 输出 validity 总表
    df_validity = pd.DataFrame(bundle_validity_rows)
    df_validity.to_csv(os.path.join(args.out_dir, "bundle_validity.csv"), index=False, encoding="utf-8-sig")

    # 输出主方法 / 补充方法汇总
    df_main = pd.DataFrame(main_rows)
    df_support = pd.DataFrame(support_rows)

    summarize(df_main, "main_method", args.out_dir)
    summarize(df_support, "supporting_method", args.out_dir)

    # 终端打印一个最简 summary（按 generator）
    def _print_gen_summary(df: pd.DataFrame, title: str) -> None:
        print(f"\n=== {title} ===")
        if df.empty:
            print("(empty)")
            return
        trig = df["trigger"].astype(str).str.lower()
        by_gen = df.groupby("generator").agg(
            n=("total", "size"),
            avg_total=("total", "mean"),
            explicit_avg=("total", lambda x: float(x[trig.reindex(x.index, fill_value=False) == "explicit"].mean()) if (trig.reindex(x.index, fill_value=False) == "explicit").any() else float("nan")),
            implicit_avg=("total", lambda x: float(x[trig.reindex(x.index, fill_value=False) == "implicit"].mean()) if (trig.reindex(x.index, fill_value=False) == "implicit").any() else float("nan")),
            zero_count=("total", lambda x: int((x == 0).sum())),
            perfect_count=("total", lambda x: int((x == 10).sum())),
        ).reset_index()
        print(by_gen.to_string(index=False))

    _print_gen_summary(df_main, "MAIN_METHOD (cross-model)")
    _print_gen_summary(df_support, "SUPPORTING_METHOD (self-eval)")

    print(f"\nDONE. CSV written to: {args.out_dir}")


if __name__ == "__main__":
    main()

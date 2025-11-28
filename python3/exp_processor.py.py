# -*- coding: utf-8 -*-
"""
exp_processor.py
汇总 11 个测试函数在 3 组设置下的实验结果，自动：
- 解析并合并不同来源/命名的 logs（CSV/JSON）
- 统一列名（hv/igd/evals/time/feasible/violations/asf/regret/…）
- 计算每函数的组内统计（均值、标准差、中位数、IQR、最佳）
- 生成 5 张“每函数数据表”：总体汇总、HV/IGD 曲线抽样、显著性检验、消融、冲突指标
- 生成跨函数的总表、图（箱线图、收敛曲线）、统计检验（Wilcoxon+Holm）
- 导出到 ./results/ 下：CSV、PNG、以及一个简明 PDF 附录（reportlab）

用法：
  python exp_processor.py \
      --root ./exp_data \
      --map baseline=baseline \
      --map group2=no_conflict_all \
      --map group3=conflict_all \
      --pattern "**/*.csv" --pattern "**/*.json"
"""
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

import os, re, json, math, argparse, glob, statistics
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# ---------- 列名标准化映射 ----------
CANON = {
    "func": ["func", "function", "name", "problem", "test_func", "test"],
    "group": ["group", "setting", "condition", "exp", "mode", "case"],
    "run": ["run", "seed", "trial", "rep", "replicate"],
    "hv": ["hv","hypervolume","hyper_vol","hv_last","hv_final"],
    "igd": ["igd","inverted_gd","igd_plus","igd_final","igd_last"],
    "evals": ["evals","evaluations","iter","iters","fe","nfe","nfes","evaluation"],
    "time": ["time","seconds","runtime","runtime_secs","walltime","elapsed"],
    "feas": ["feasible_rate","feasible","feas_rate","feas"],
    "vio": ["violations","constraint_violations","cv","mean_cv","total_cv"],
    "asf": ["asf","regret","pref_regret","distance","loss"],
    "conflict": ["conflict_detected","has_conflict","conflict"],
    "resolved": ["conflict_resolved","resolved","resolution"],
    "nl_n": ["nl_interventions","nl_cnt","nl_n","num_nl"],
    "slider_n": ["slider_interventions","slider_cnt","slider_n","num_slider"],
    "rank_n": ["rank_interventions","rank_cnt","rank_n","num_rank"],
}

# ---------- 统一列名 ----------
def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    def pick(target):
        for alias in CANON[target]:
            if alias in cols: return cols[alias]
        # 宽松匹配
        for k,v in cols.items():
            if re.fullmatch(rf"{target}(_\w+)?", k): return v
        return None
    mapping = {}
    for t in CANON:
        c = pick(t)
        if c: mapping[c] = t
    # 重命名
    df = df.rename(columns=mapping)
    # 类型兜底
    for t in ["hv","igd","evals","time","feas","vio","asf","nl_n","slider_n","rank_n"]:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce")
    # bool类
    for t in ["conflict","resolved"]:
        if t in df.columns:
            df[t] = df[t].astype(str).str.lower().map({"1":True,"true":True,"yes":True,"y":True,"t":True,"0":False,"false":False,"no":False,"n":False,"f":False})
    return df

# ---------- 尝试读文件（CSV/JSON） ----------
def read_any(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower()==".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower()==".json":
            with open(path,"r",encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "rows" in data:
                df = pd.DataFrame(data["rows"])
            else:
                df = pd.json_normalize(data)
        else:
            return None
        df["__src__"] = str(path)
        return df
    except Exception as e:
        print(f"[WARN] read fail: {path} -> {e}")
        return None

def try_infer_group_from_path(p: Path, group_map: Dict[str,str]) -> Optional[str]:
    s = str(p).replace("\\","/").lower()
    for g, key in group_map.items():
        if key in s:
            return g
    return None

def try_infer_func_from_path(p: Path) -> Optional[str]:
    s = p.name.lower()
    m = re.search(r"(zdt[12345]|dtlz[1-7]|wfg[1-9]|uf[1-9]|cf[1-9])", s)
    if m:
        return m.group(1).upper()
    # 退一步：目录名里找
    s2 = str(p.parent).lower()
    m2 = re.search(r"(zdt[12345]|dtlz[1-7]|wfg[1-9]|uf[1-9]|cf[1-9])", s2)
    if m2:
        return m2.group(1).upper()
    return None

# ---------- 统计工具 ----------
def mean_std(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x)==0: return (np.nan, np.nan)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x)>1 else 0.0

def median_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x)==0: return (np.nan, np.nan)
    q1, med, q3 = np.percentile(x, [25,50,75])
    return float(med), float(q3-q1)

def wilcoxon_signed(a, b):
    # 简化：若缺数据/长度不一致 -> 返回 nan
    try:
        from scipy.stats import wilcoxon
        a = np.asarray(a, dtype=float); b=np.asarray(b, dtype=float)
        n = min(len(a), len(b))
        if n<5: return np.nan
        return float(wilcoxon(a[:n], b[:n]).pvalue)
    except Exception:
        return np.nan

def holm_correction(pvals: Dict[Tuple[str,str], float]) -> Dict[Tuple[str,str], float]:
    items = [(k,v) for k,v in pvals.items() if not np.isnan(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out = {}
    for i,(k,p) in enumerate(items, start=1):
        out[k] = min(1.0, p*(m - i + 1))
    # 缺失的保持 nan
    for k in pvals:
        if k not in out: out[k] = np.nan
    return out

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./exp_data", help="解压后的数据根目录")
    ap.add_argument("--pattern", action="append", default=["**/*.csv","**/*.json"], help="匹配日志文件的 glob 模式，可多次指定")
    ap.add_argument("--map", action="append", default=["baseline=baseline","group2=no_conflict","group3=with_conflict"],
                    help="组名映射：左边是标准组名（baseline/group2/group3），右边是路径关键词")
    ap.add_argument("--out", type=str, default="./results", help="输出目录")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 组映射
    group_map = {}
    for m in args.map:
        k,v = m.split("=",1)
        group_map[k.strip()] = v.strip().lower()
    expected_groups = ["baseline","group2","group3"]

    # 查找文件
    files = []
    for pat in args.pattern:
        files += [Path(p) for p in glob.glob(str(root / pat), recursive=True)]
    files = [p for p in files if p.is_file()]

    if not files:
        print(f"[ERROR] 找不到日志文件。请确认 --root 与 --pattern 是否正确。")
        return

    rows = []
    for p in files:
        df = read_any(p)
        if df is None or len(df)==0:
            continue
        df = canonicalize_columns(df)

        # 填 func / group
        if "func" not in df.columns:
            f = try_infer_func_from_path(p)
            if f: df["func"] = f
        if "group" not in df.columns:
            g = try_infer_group_from_path(p, group_map)
            if g: df["group"] = g

        # 填 run
        if "run" not in df.columns:
            # 从文件名猜
            m = re.search(r"(seed|run|rep|trial)[-_]?(\d+)", p.name.lower())
            df["run"] = int(m.group(2)) if m else 0

        # 最低必要列：func & group
        if "func" not in df.columns or "group" not in df.columns:
            # 放弃这份
            continue

        # 只保留和我们关心的列
        keep = ["__src__","func","group","run","hv","igd","evals","time","feas","vio","asf",
                "conflict","resolved","nl_n","slider_n","rank_n"]
        for k in keep:
            if k not in df.columns: df[k] = np.nan
        rows.append(df[keep])

    if not rows:
        print("[ERROR] 未解析到任何有效数据；请检查命名或使用 --map/--pattern。")
        return

    all_df = pd.concat(rows, ignore_index=True)

    # --- 输出一份原始合并数据 ---
    all_df.to_csv(out / "all_merged.csv", index=False, encoding="utf-8-sig")

    # --- 每函数统计 & 生成 5 张表 ---
    per_func_dir = out / "per_function"; per_func_dir.mkdir(exist_ok=True)
    figures_dir = out / "figs"; figures_dir.mkdir(exist_ok=True)

    func_list = sorted(all_df["func"].dropna().unique().tolist())

    # 总汇总
    overall_summ_rows = []

    for func in func_list:
        dfF = all_df[all_df["func"]==func].copy()

        # 表1：总体汇总（均值±std）
        rows1 = []
        for g in expected_groups:
            dfg = dfF[dfF["group"]==g]
            hv_m, hv_s = mean_std(dfg["hv"])
            igd_m, igd_s = mean_std(dfg["igd"])
            time_m, time_s = mean_std(dfg["time"])
            feas_m, _ = mean_std(dfg["feas"])
            vio_m, _ = mean_std(dfg["vio"])
            asf_m, asf_s = mean_std(dfg["asf"])

            rows1.append({
                "func": func, "group": g,
                "hv_mean": hv_m, "hv_std": hv_s,
                "igd_mean": igd_m, "igd_std": igd_s,
                "time_mean": time_m, "time_std": time_s,
                "feas_mean": feas_m,
                "vio_mean": vio_m,
                "asf_mean": asf_m, "asf_std": asf_s,
                "n": int(len(dfg))
            })
        df1 = pd.DataFrame(rows1)
        df1.to_csv(per_func_dir / f"{func}_table1_summary.csv", index=False, encoding="utf-8-sig")

        # 表2：曲线抽样（若 evals 存在）：按 evals 取分位点
        df2 = pd.DataFrame()
        if dfF["evals"].notna().any():
            # 取每组在 evals 维度的若干点；为鲁棒，先把 evals 分箱
            bins = 10
            try:
                # 每组分箱统计 median(HV), median(IGD)
                recs = []
                for g in expected_groups:
                    dfg = dfF[dfF["group"]==g].copy()
                    if dfg["evals"].notna().any():
                        # 分箱
                        qs = np.quantile(dfg["evals"].dropna(), np.linspace(0,1,bins+1))
                        qs[0] = max(0, qs[0])
                        for i in range(bins):
                            lo, hi = qs[i], qs[i+1]
                            seg = dfg[(dfg["evals"]>=lo) & (dfg["evals"]<=hi)]
                            if len(seg)==0: continue
                            recs.append({
                                "func": func, "group": g,
                                "evals_bin_lo": float(lo), "evals_bin_hi": float(hi),
                                "evals_median": float(np.median(seg["evals"])),
                                "hv_median": float(np.nanmedian(seg["hv"])) if seg["hv"].notna().any() else np.nan,
                                "igd_median": float(np.nanmedian(seg["igd"])) if seg["igd"].notna().any() else np.nan
                            })
                df2 = pd.DataFrame(recs)
                df2.to_csv(per_func_dir / f"{func}_table2_curve_bins.csv", index=False, encoding="utf-8-sig")
            except Exception as e:
                pass

        # 表3：显著性检验（pairwise，基于每个 run 的末值）
        rows3 = []
        def last_by_run(dfg, col):
            # 假设同一 run 多行：取最大 evals 对应的 col
            if "evals" in dfg.columns and dfg["evals"].notna().any():
                idx = dfg.groupby("run")["evals"].transform("max")==dfg["evals"]
                return dfg[idx][["run",col]].dropna()
            else:
                return dfg[["run",col]].dropna()

        pairs = [("baseline","group2"), ("baseline","group3"), ("group2","group3")]
        pmap_hv, pmap_igd = {}, {}
        for a,b in pairs:
            a_hv = last_by_run(dfF[dfF["group"]==a], "hv")["hv"].values
            b_hv = last_by_run(dfF[dfF["group"]==b], "hv")["hv"].values
            a_ig = last_by_run(dfF[dfF["group"]==a], "igd")["igd"].values
            b_ig = last_by_run(dfF[dfF["group"]==b], "igd")["igd"].values

            p_hv = wilcoxon_signed(a_hv, b_hv)
            p_ig = wilcoxon_signed(a_ig, b_ig)
            pmap_hv[(a,b)] = p_hv
            pmap_igd[(a,b)] = p_ig

        pmap_hv_adj = holm_correction(pmap_hv)
        pmap_igd_adj = holm_correction(pmap_igd)
        for (a,b) in pairs:
            rows3.append({"func":func, "metric":"HV", "pair":f"{a} vs {b}",
                          "p_raw": pmap_hv[(a,b)], "p_holm": pmap_hv_adj[(a,b)]})
            rows3.append({"func":func, "metric":"IGD", "pair":f"{a} vs {b}",
                          "p_raw": pmap_igd[(a,b)], "p_holm": pmap_igd_adj[(a,b)]})
        df3 = pd.DataFrame(rows3)
        df3.to_csv(per_func_dir / f"{func}_table3_significance.csv", index=False, encoding="utf-8-sig")

        # 表4：三种辅助方式的消融（如果日志包含列 nl_n/slider_n/rank_n/asf，可统计干预与性能的相关）
        rows4 = []
        for g in expected_groups:
            dfg = dfF[dfF["group"]==g]
            rec = {"func":func, "group":g}
            for col in ["nl_n","slider_n","rank_n","asf","hv","igd","time"]:
                if col in dfg.columns and dfg[col].notna().any():
                    rec[col+"_mean"], rec[col+"_std"] = mean_std(dfg[col])
                else:
                    rec[col+"_mean"], rec[col+"_std"] = (np.nan, np.nan)
            rows4.append(rec)
        df4 = pd.DataFrame(rows4)
        df4.to_csv(per_func_dir / f"{func}_table4_ablation.csv", index=False, encoding="utf-8-sig")

        # 表5：冲突指标（冲突发生率/解决率/解决时长——如果有 time 或标注）
        rows5 = []
        for g in expected_groups:
            dfg = dfF[dfF["group"]==g]
            cnt = len(dfg)
            if cnt==0:
                rows5.append({"func":func,"group":g,"conflict_rate":np.nan,"resolve_rate":np.nan})
                continue
            conf_rate = float(dfg["conflict"].fillna(False).sum())/cnt if "conflict" in dfg.columns else np.nan
            reso_rate = float(dfg["resolved"].fillna(False).sum())/max(1, int(dfg["conflict"].fillna(False).sum())) if "resolved" in dfg.columns else np.nan
            rows5.append({"func":func,"group":g,"conflict_rate":conf_rate,"resolve_rate":reso_rate})
        df5 = pd.DataFrame(rows5)
        df5.to_csv(per_func_dir / f"{func}_table5_conflict.csv", index=False, encoding="utf-8-sig")

        # 累加到总体
        overall_summ_rows += rows1

        # 简单画图（箱线图）
        try:
            import matplotlib.pyplot as plt
            for m in ["hv","igd"]:
                if m not in dfF.columns: continue
                plt.figure()
                data = [dfF[dfF["group"]==g][m].dropna().values for g in expected_groups]
                plt.boxplot(data, labels=expected_groups, showmeans=True)
                plt.title(f"{func} - {m.upper()}")
                plt.ylabel(m.upper())
                plt.grid(True, alpha=.3)
                plt.tight_layout()
                plt.savefig(figures_dir / f"{func}_{m}.png", dpi=150)
                plt.close()
        except Exception as e:
            print(f"[WARN] plotting fail for {func}: {e}")

    # 导出总体汇总表
    df_all = pd.DataFrame(overall_summ_rows)
    df_all.to_csv(out / "overall_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n[OK] 完成。输出目录：{out.resolve()}")
    print("  - all_merged.csv")
    print("  - per_function/*table[1-5]_*.csv")
    print("  - figs/*.png")
    print("  - overall_summary.csv")

if __name__ == "__main__":
    main()

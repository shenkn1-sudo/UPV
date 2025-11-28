# exp_logger.py
from datetime import datetime
import os, json, csv
import numpy as np
import pandas as pd

HEADER = [
    "timestamp","run_id","mode","problem","algo","pop_size","gen_per_tick","seed",
    "tick","evals_cum","hv","igd","feasible_ratio","mean_penalty",
    "w1","w2","r1","r2","tau","cvar","cr_on",
    "n_atoms_nl","n_atoms_visual","n_pairs_rank","n_clarify_total"
]

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---- 解析式参考前沿：仅覆盖常用 2 目标基准 ----
def get_reference_pf(problem_id: str, n_points: int = 1000):
    pid = (problem_id or "").upper()
    t = np.linspace(0.0, 1.0, n_points)

    if pid == "ZDT1":
        f1 = t; f2 = 1.0 - np.sqrt(f1); return np.column_stack([f1, f2])
    if pid == "ZDT2":
        f1 = t; f2 = 1.0 - (f1 ** 2.0); return np.column_stack([f1, f2])
    if pid == "ZDT3":
        f1 = t; f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        mask = np.isfinite(f2); return np.column_stack([f1[mask], f2[mask]])
    if pid == "ZDT4":
        f1 = t; f2 = 1.0 - np.sqrt(f1); return np.column_stack([f1, f2])
    if pid == "ZDT6":
        x0 = t
        f1 = 1.0 - np.exp(-4.0 * x0) * (np.sin(6.0 * np.pi * x0) ** 6)
        f2 = 1.0 - (f1 ** 2.0)
        return np.column_stack([f1, f2])

    if pid in ("DTLZ2","DTLZ3","DTLZ4"):
        theta = 0.5 * np.pi * t
        f1, f2 = np.cos(theta), np.sin(theta)
        return np.column_stack([f1, f2])

    if pid == "DTLZ1":
        f1 = 0.5 * t; f2 = 0.5 - f1
        return np.column_stack([f1, f2])

    # KUR 无解析式 PF：返回 None（IGD 将置空）
    return None

def compute_igd(F, R):
    if R is None or F is None or len(F) == 0:
        return None

    def _is_nondominated(F):
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]: continue
            for j in range(n):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False; break
        return nd

    F_nd = F[_is_nondominated(F)]
    d = [np.min(np.linalg.norm(F_nd - r, axis=1)) for r in R]
    return float(np.mean(d))

class ExperimentLogger:
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = base_dir
        _ensure_dir(self.base_dir)
        self.run_id = None
        self.run_dir = None
        self.csv_path = None

    def start_run(self, run_id: str, config: dict):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or f"run-{ts}"
        self.run_dir = os.path.join(self.base_dir, self.run_id)
        _ensure_dir(self.run_dir)

        # 保存配置
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2,
                      default=lambda o: float(o) if hasattr(o, "__float__") else str(o))

        # 准备 CSV
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(HEADER)

    def log_tick(self, row: dict):
        if not self.csv_path: return
        vals = [row.get(h, "") if row.get(h, "") is not None else "" for h in HEADER]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(vals)

    def dump_population(self, X, F, tick: int | None = None):
        if self.run_dir is None: return
        name = "final_front" if tick is None else f"front_t{int(tick)}"
        df = pd.DataFrame(np.hstack([X, F]))
        df.to_csv(os.path.join(self.run_dir, f"{name}.csv"), index=False)

    def dump_upv(self, upv: dict, name: str = "upv_snapshot.json"):
        if self.run_dir is None: return
        with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as f:
            json.dump(upv, f, ensure_ascii=False, indent=2)

    def finalize(self):
        pass

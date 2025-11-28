from typing import Dict, Any, List, Tuple
import numpy as np
from NL2UPV import ConstraintAtom, llm_nl_to_patch, UPV
from NL2UPV import ConstraintAtom, llm_nl_to_patch


# 其他代码不变
def apply_patch_to_upv(current_upv: UPV, patch: Dict[str, Any], M: int) -> UPV:
    """
    将 LLM 产生的 patch 增量应用到 UPV 对象
    """
    if current_upv is None:
        raise ValueError("current_upv is None")

    # 初始化 w, r, b，确保 UPV 结构有默认值
    w = list(current_upv.w) if hasattr(current_upv.w, "__iter__") else [float(current_upv.w)]
    r = list(current_upv.r) if hasattr(current_upv.r, "__iter__") else [0.0] * M
    b = list(current_upv.b) if hasattr(current_upv.b, "__iter__") else [(-float("inf"), float("inf"))] * M
    if len(w) < M: w = (w + [1.0] * M)[:M]
    if len(r) < M: r = (r + [0.0] * M)[:M]
    if len(b) < M: b = (b + [(-float("inf"), float("inf"))] * M)[:M]

    # 补丁应用操作
    for op in patch.get("ops", []):
        t = op.get("op")

        if t == "set_weight":
            idx, val = int(op.get("index", -1)), float(op.get("value", 0.0))
            if 0 <= idx < M:
                w[idx] = max(0.0, val)

        # 更新目标参考点
        elif t == "set_reference":
            vals = op.get("values", [])
            if vals:
                r = (list(map(float, vals)) + [0.0] * M)[:M]

        # 更新目标区间
        elif t == "set_band":
            idx = int(op.get("obj_index", -1))
            lo = op.get("lo", -float("inf"))
            hi = op.get("hi", float("inf"))
            if 0 <= idx < M:
                b[idx] = (lo, hi)

        # 添加约束
        elif t == "add_constraint":
            kind = op.get("kind")
            idx = int(op.get("index"))
            sign = op.get("sign")
            value = float(op.get("value"))
            hard = op.get("hard", True)
            penalty = float(op.get("penalty", 10.0))
            constraint = ConstraintAtom(kind, idx, sign, value, hard, penalty)
            (current_upv.C_h if hard else current_upv.C_s).append(constraint)

    # 归一化目标权重
    w = _norm(w)

    # 更新 UPV 结构
    current_upv.w = np.asarray(w, dtype=float)
    current_upv.r = np.asarray(r, dtype=float)
    current_upv.b = [(float(lo), float(hi)) for (lo, hi) in b]

    return current_upv


from typing import List, Dict, Any
import openai
import json


# RAG 部分：从外部知识库进行检索
def knowledge_retrieval(query: str) -> List[str]:
    """
    模拟从外部知识库进行检索。
    这里我们假设调用一个API进行检索，返回与查询相关的信息。
    """
    # 这部分应该根据实际的知识库和检索算法进行实现
    # 例如：通过API检索并返回相关知识（例如：OpenAI、科大讯飞等）
    retrieved_info = [
        "在多目标优化中，目标权重的设定非常重要，它直接影响到优化结果的权衡。",
        "软约束通常用于在优化过程中引入某种程度的灵活性，允许算法寻找更广泛的解。",
        "根据用户提供的目标和约束条件，优化系统会自动调整搜索空间。"
    ]
    return retrieved_info


# LLM 将自然语言转化为补丁，并加入检索的知识
def llm_nl_to_patch_with_rag(user_text: str) -> Dict[str, Any]:
    """
    使用 RAG 来增强 LLM 的自然语言处理过程，并将其转化为补丁。
    """
    # 1. 使用 RAG 检索与用户文本相关的外部信息
    retrieved_info = knowledge_retrieval(user_text)

    # 2. 使用检索到的信息作为上下文，构造一个增强后的 LLM 提示词
    prompt = f"""
    以下是检索到的相关信息：
    {json.dumps(retrieved_info, ensure_ascii=False, indent=2)}

    用户请求的优化目标和约束： {user_text}

    请根据这些信息生成一个严格的 JSON 补丁，格式如下：
    {{
        "ops": [
            {{ "op": "set_weight", "index": 0, "value": 0.5 }},
            {{ "op": "add_constraint", "kind": "obj", "index": 1, "sign": "<=", "value": 0.8, "hard": true }}
        ]
    }}
    """
    # 3. 调用 LLM 生成补丁（这里假设我们用 OpenAI API 或科大讯飞）
    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # 假设使用 GPT-3
        prompt=prompt,
        max_tokens=150
    )

    # 假设返回的内容是一个有效的 JSON
    patch = json.loads(response.choices[0].text.strip())
    return patch


from NL2UPV import UPV  # 从NL2UPV.py文件导入UPV类
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor

# 现有优化函数
def optimize_with_bo(upv: UPV):
    # 使用 LLM + RAG 增强优化过程
    user_request = "希望优化速度"
    patch = llm_nl_to_patch_with_rag(user_request)  # 这里调用我们增强后的函数
    selected_point = patch["selected_point"]

    # 进行评估和更新UPV结构
    evaluations = evaluate(selected_point)
    upv = apply_patch_to_upv(upv, patch, M=len(evaluations))

    return upv


def evaluate(candidate: List[float]) -> List[float]:
    # 模拟目标评估（这里可以替换为实际评估过程）
    f1 = sum(candidate)
    f2 = 1.0 / sum(candidate)
    return [f1, f2]


def apply_patch_to_upv(current_upv: UPV, patch: Dict[str, Any], M: int) -> UPV:
    """
    将 LLM 产生的 patch 增量应用到 UPV 对象
    """
    if current_upv is None:
        raise ValueError("current_upv is None")

    # 初始化 w, r, b，确保 UPV 结构有默认值
    w = list(current_upv.w) if hasattr(current_upv.w, "__iter__") else [float(current_upv.w)]
    r = list(current_upv.r) if hasattr(current_upv.r, "__iter__") else [0.0] * M
    b = list(current_upv.b) if hasattr(current_upv.b, "__iter__") else [(-float("inf"), float("inf"))] * M
    if len(w) < M: w = (w + [1.0] * M)[:M]
    if len(r) < M: r = (r + [0.0] * M)[:M]
    if len(b) < M: b = (b + [(-float("inf"), float("inf"))] * M)[:M]

    # 补丁应用操作
    for op in patch.get("ops", []):
        t = op.get("op")

        # 更新目标权重
        if t == "set_weight":
            idx, val = int(op.get("index", -1)), float(op.get("value", 0.0))
            if 0 <= idx < M:
                w[idx] = max(0.0, val)

        # 更新目标参考点
        elif t == "set_reference":
            vals = op.get("values", [])
            if vals:
                r = (list(map(float, vals)) + [0.0] * M)[:M]

        # 更新目标区间
        elif t == "set_band":
            idx = int(op.get("obj_index", -1))
            lo = op.get("lo", -float("inf"))
            hi = op.get("hi", float("inf"))
            if 0 <= idx < M:
                b[idx] = (lo, hi)

        # 添加约束
        elif t == "add_constraint":
            kind = op.get("kind")
            idx = int(op.get("index"))
            sign = op.get("sign")
            value = float(op.get("value"))
            hard = op.get("hard", True)
            penalty = float(op.get("penalty", 10.0))
            constraint = ConstraintAtom(kind, idx, sign, value, hard, penalty)
            (current_upv.C_h if hard else current_upv.C_s).append(constraint)

    # 归一化目标权重
    w = _norm(w)

    # 更新 UPV 结构
    current_upv.w = np.asarray(w, dtype=float)
    current_upv.r = np.asarray(r, dtype=float)
    current_upv.b = [(float(lo), float(hi)) for (lo, hi) in b]

    return current_upv


from NL2UPV import ConstraintAtom, llm_nl_to_patch

import os
import re
import json
import threading
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

import dash
from dash import Dash, html, dcc, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

# ==== 拖拽排序：使用 dash-extensions.EventListener ====
HAS_DE = False
try:
    from dash_extensions import EventListener
    HAS_DE = True
except Exception:
    HAS_DE = False

# ==== 你的 NL→UPV 模块（保持不变） ====
from NL2UPV import parse_nl_to_atoms, atoms_to_upv
from NL2UPV import UPV as UPV_Pydantic




# ==== 实验记录器 & IGD 工具 ====
from exp_logger import ExperimentLogger, get_reference_pf, compute_igd
# ------------------ 问题定义：经典多目标测试集 + 工厂 ------------------

# —— ZDT 家族（2 目标）——
class ZDT1(ElementwiseProblem):
    def __init__(self, n_var=30):
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        out["F"] = np.array([f1, f2], dtype=float)

class ZDT2(ElementwiseProblem):
    def __init__(self, n_var=30):
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - (f1 / g) ** 2.0)
        out["F"] = np.array([f1, f2], dtype=float)

class ZDT3(ElementwiseProblem):
    def __init__(self, n_var=30):
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
        out["F"] = np.array([f1, f2], dtype=float)

class ZDT4(ElementwiseProblem):
    def __init__(self, n_var=10):
        bounds = np.zeros((n_var, 2), dtype=float)
        bounds[0] = [0.0, 1.0]
        bounds[1:] = [-5.0, 5.0]
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = x[0]
        g = 1.0 + 10.0 * (self.n_var - 1) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
        f2 = g * (1.0 - np.sqrt(f1 / g))
        out["F"] = np.array([f1, f2], dtype=float)

class ZDT6(ElementwiseProblem):
    def __init__(self, n_var=30):
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x)
        f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0])) ** 6
        g = 1.0 + 9.0 * (np.mean(x[1:]) ** 0.25)
        f2 = g * (1.0 - (f1 / g) ** 2.0)
        out["F"] = np.array([f1, f2], dtype=float)

# —— DTLZ（M=2）——
def _dtlz_split(n_var, M=2):
    return max(1, n_var - M + 1)

class DTLZ1(ElementwiseProblem):
    def __init__(self, n_var=6, M=2):
        self.M = 2
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        k = _dtlz_split(self.n_var, 2)
        g = 100.0 * (k + np.sum((x[self.n_var - k:] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[self.n_var - k:] - 0.5))))
        f1 = 0.5 * (1.0 + g) * x[0]
        f2 = 0.5 * (1.0 + g) * (1.0 - x[0])
        out["F"] = np.array([f1, f2], dtype=float)

class DTLZ2(ElementwiseProblem):
    def __init__(self, n_var=11, M=2):
        self.M = 2
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        k = _dtlz_split(self.n_var, 2)
        g = np.sum((x[self.n_var - k:] - 0.5) ** 2)
        theta = 0.5 * np.pi * x[0]
        f1 = (1.0 + g) * np.cos(theta)
        f2 = (1.0 + g) * np.sin(theta)
        out["F"] = np.array([f1, f2], dtype=float)

class DTLZ3(ElementwiseProblem):
    def __init__(self, n_var=11, M=2):
        self.M = 2
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        k = _dtlz_split(self.n_var, 2)
        g = 100.0 * (k + np.sum((x[self.n_var - k:] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[self.n_var - k:] - 0.5))))
        theta = 0.5 * np.pi * x[0]
        f1 = (1.0 + g) * np.cos(theta)
        f2 = (1.0 + g) * np.sin(theta)
        out["F"] = np.array([f1, f2], dtype=float)

class DTLZ4(ElementwiseProblem):
    def __init__(self, n_var=11, M=2, alpha=100.0):
        self.M = 2
        self.alpha = alpha
        bounds = np.array([[0.0, 1.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        k = _dtlz_split(self.n_var, 2)
        g = np.sum((x[self.n_var - k:] - 0.5) ** 2)
        theta = 0.5 * np.pi * (x[0] ** self.alpha)
        f1 = (1.0 + g) * np.cos(theta)
        f2 = (1.0 + g) * np.sin(theta)
        out["F"] = np.array([f1, f2], dtype=float)

# —— Kursawe（2 目标，3 变量）——
class KUR(ElementwiseProblem):
    def __init__(self, n_var=3):
        assert n_var >= 3, "Kursawe 通常使用 3 变量"
        bounds = np.array([[-5.0, 5.0]] * n_var, dtype=float)
        self.bounds = bounds
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=bounds[:, 0], xu=bounds[:, 1])
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        s1 = 0.0
        for i in range(self.n_var - 1):
            s1 += -10.0 * np.exp(-0.2 * np.sqrt(x[i] ** 2 + x[i + 1] ** 2))
        s2 = np.sum(np.abs(x) ** 0.8 + 5.0 * np.sin(x ** 3))
        out["F"] = np.array([s1, s2], dtype=float)

# —— 工厂 ——
PROBLEM_CATALOG = {
    "ZDT1":   {"cls": ZDT1,  "default_nvar": 30, "label": "ZDT1 (convex front)"},
    "ZDT2":   {"cls": ZDT2,  "default_nvar": 30, "label": "ZDT2 (non-convex)"},
    "ZDT3":   {"cls": ZDT3,  "default_nvar": 30, "label": "ZDT3 (disconnected)"},
    "ZDT4":   {"cls": ZDT4,  "default_nvar": 10, "label": "ZDT4 (multimodal)"},
    "ZDT6":   {"cls": ZDT6,  "default_nvar": 30, "label": "ZDT6 (bias f1)"},
    "DTLZ1":  {"cls": DTLZ1, "default_nvar": 6,  "label": "DTLZ1"},
    "DTLZ2":  {"cls": DTLZ2, "default_nvar": 11, "label": "DTLZ2"},
    "DTLZ3":  {"cls": DTLZ3, "default_nvar": 11, "label": "DTLZ3 (many local)"},
    "DTLZ4":  {"cls": DTLZ4, "default_nvar": 11, "label": "DTLZ4 (bias)"},
    "KUR":    {"cls": KUR,   "default_nvar": 3,  "label": "Kursawe"},
}

def make_problem(problem_id: str, n_var: int) -> ElementwiseProblem:
    pid = (problem_id or "ZDT1").upper()
    meta = PROBLEM_CATALOG.get(pid)
    if not meta:
        meta = PROBLEM_CATALOG["ZDT1"]
    cls = meta["cls"]
    try:
        return cls(n_var=int(n_var))
    except TypeError:
        return cls()


# ------------------ 内部 UPV 结构 ------------------
@dataclass
class ConstraintAtom:
    kind: str
    idx: int
    sign: str
    value: float
    hard: bool = True
    penalty: float = 10.0

@dataclass
class UPV:
    w: np.ndarray
    r: np.ndarray
    b: List[Tuple[float, float]]
    C_h: List[ConstraintAtom] = field(default_factory=list)
    C_s: List[ConstraintAtom] = field(default_factory=list)
    D: List[Tuple[float, float]] = field(default_factory=list)
    tau: float = 0.2
    rho: Dict[str, Any] = field(default_factory=lambda: {"cvar_alpha": 1.0, "noise_std": 0.0})
    pi: Dict[str, Any] = field(default_factory=lambda: {"explore_bias": 0.0})
    kappa: Dict[str, float] = field(default_factory=lambda: {"hard": 1e-6, "var": 0.2, "robust": 0.2})
    sigma: Dict[str, float] = field(default_factory=lambda: {"w": 0.5, "r": 0.5, "b": 0.5, "D": 0.5})
    provenance: List[str] = field(default_factory=list)
    def log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.provenance.append(f"[{stamp}] {msg}")

def pyd_upv_to_internal(pu: UPV_Pydantic, n_var: int) -> UPV:
    w = np.array(pu.w, dtype=float)
    r = np.array(pu.r, dtype=float)
    b = [(float(lo), float(hi)) for (lo, hi) in pu.b]
    D = [tuple(x) for x in pu.D] if len(pu.D) > 0 else []
    C_h, C_s = [], []
    for c in pu.C_h:
        C_h.append(ConstraintAtom(c.kind, int(c.idx), c.sign, float(c.value), True, float(c.penalty)))
    for c in pu.C_s:
        C_s.append(ConstraintAtom(c.kind, int(c.idx), c.sign, float(c.value), False, float(c.penalty)))
    return UPV(w=w, r=r, b=b, C_h=C_h, C_s=C_s, D=D, tau=float(pu.tau),
               rho=dict(pu.rho.model_dump()), pi=dict(pu.pi.model_dump()))

def normalize_w(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 1e-12)
    return w / np.sum(w)


# ------------------ 约束与指标工具 ------------------
def check_var_D(x: np.ndarray, D: List[Tuple[float, float]]) -> bool:
    if not D: return True
    for i, (lo, hi) in enumerate(D):
        if x[i] < lo - 1e-12 or x[i] > hi + 1e-12: return False
    return True

def check_constraint(x: np.ndarray, f: np.ndarray, c: ConstraintAtom) -> bool:
    val = x[c.idx] if c.kind == "var" else f[c.idx]
    if c.sign == "<=": return val <= c.value + 1e-12
    if c.sign == ">=": return val >= c.value - 1e-12
    return abs(val - c.value) <= 1e-9

def soft_penalty(x: np.ndarray, f: np.ndarray, c: ConstraintAtom) -> float:
    val = x[c.idx] if c.kind == "var" else f[c.idx]
    if c.sign == "<=": v = max(0.0, val - c.value)
    elif c.sign == ">=": v = max(0.0, c.value - val)
    else: v = abs(val - c.value)
    return c.penalty * v

def compute_penalty_and_feas(x: np.ndarray, f: np.ndarray, upv: UPV, cr_on: bool) -> Tuple[float, bool]:
    P = 0.0
    feas = True
    if upv.D and not check_var_D(x, upv.D):
        P += (1000.0 if cr_on else 10.0); feas = False
    for c in upv.C_h:
        ok = check_constraint(x, f, c)
        if not ok: P += (1000.0 if cr_on else 10.0); feas = False
    for c in upv.C_s:
        P += soft_penalty(x, f, c)
    return P, feas

def asf_distance(f: np.ndarray, w: np.ndarray, r: np.ndarray, rho: float = 1e-6) -> float:
    return float(np.max(w * np.abs(f - r)) + rho * np.sum(w * np.abs(f - r)))


# ------------------ L1~L4 冲突消解 ------------------
def asf_distance_batch(F: np.ndarray, w: np.ndarray, r: np.ndarray) -> Tuple[int, float]:
    d = np.array([asf_distance(F[i], w, r) for i in range(F.shape[0])])
    idx = int(np.argmin(d))
    return idx, float(d[idx])

def l1_project_aspiration(state: "OptState", threshold: float = 0.12, eta: float = 0.25):
    if state.pop_F is None or len(state.pop_F) == 0: return None
    idx, dmin = asf_distance_batch(state.pop_F, state.upv.w, state.upv.r)
    if dmin > threshold:
        alpha = eta * (1.0 - state.upv.tau)
        best = state.pop_F[idx]
        old_r = state.upv.r.copy()
        state.upv.r = (1 - alpha) * state.upv.r + alpha * best
        state.add_log("CR★L1", f"dmin={dmin:.4f}, alpha={alpha:.3f}, r: {old_r.tolist()} -> {state.upv.r.tolist()}")
        return True
    return False

def l2_reconcile_D_vs_Ch(state: "OptState"):
    changed = False
    notes = []
    D = list(state.upv.D) if state.upv.D else [tuple(x) for x in state.problem.bounds]
    for c in state.upv.C_h:
        if c.kind != "var": continue
        lo, hi = D[c.idx]
        old = (lo, hi)
        if c.sign == "<=" and hi > c.value: hi = c.value; changed = True
        elif c.sign == ">=" and lo < c.value: lo = c.value; changed = True
        elif c.sign == "==" and (lo > c.value or hi < c.value): lo = hi = c.value; changed = True
        lo = max(lo, float(state.problem.bounds[c.idx,0]))
        hi = min(hi, float(state.problem.bounds[c.idx,1]))
        if old != (lo, hi):
            notes.append(f"x{c.idx+1}: {old} -> {(lo,hi)} by hard {c.sign} {c.value}")
        D[c.idx] = (lo, hi)
    if changed:
        oldD = list(state.upv.D)
        state.upv.D = D
        state.add_log("CR★L2", f"D adjusted: {oldD} -> {D}; " + " | ".join(notes))
    return changed

def l3_adapt_soft_penalties(state: "OptState",
                            target_feas: Tuple[float,float]=(0.6,0.85),
                            up_gain: float=1.12, down_gain: float=0.95):
    if not state.upv.C_s or state.pop_X is None: return False
    X, F = state.pop_X, state.pop_F
    if len(X) == 0: return False

    lower, upper = target_feas
    m = len(state.upv.C_s)
    viol = np.zeros(m, dtype=int)
    for i in range(len(X)):
        for k, c in enumerate(state.upv.C_s):
            val = X[i, c.idx] if c.kind == "var" else F[i, c.idx]
            ok = (val <= c.value + 1e-12) if c.sign=="<=" else \
                 (val >= c.value - 1e-12) if c.sign==">=" else \
                 (abs(val - c.value) <= 1e-9)
            if not ok: viol[k] += 1
    viol_rate = viol / max(1, len(X))

    changes = []
    for k, c in enumerate(state.upv.C_s):
        old_pen = c.penalty
        if viol_rate[k] > (1 - upper):
            c.penalty = float(c.penalty * up_gain)
        elif viol_rate[k] < (1 - lower):
            c.penalty = float(c.penalty * down_gain)
        if abs(c.penalty - old_pen)/max(1e-12, old_pen) > 1e-6:
            changes.append(f"soft[{k}] pen {old_pen:.3f}->{c.penalty:.3f} (viol={viol_rate[k]:.2f})")
    if changes:
        state.add_log("CR★L3", " | ".join(changes))
        return True
    return False

def l4_goal_consistency(state: "OptState", clamp_alpha: float=0.5):
    changed = False
    notes = []
    for j in range(len(state.upv.b)):
        lo, hi = state.upv.b[j]
        if not np.isfinite(lo) or not np.isfinite(hi): continue
        rj = float(state.upv.r[j])
        if rj < lo or rj > hi:
            new_rj = (1 - clamp_alpha) * rj + clamp_alpha * np.clip(rj, lo, hi)
            notes.append(f"r{j+1} {rj:.3f}->{new_rj:.3f} within [{lo:.3f},{hi:.3f}]")
            state.upv.r[j] = new_rj
            changed = True
    w_old = state.upv.w.copy()
    state.upv.w = normalize_w(state.upv.w)
    if np.max(np.abs(w_old - state.upv.w)) > 1e-12:
        notes.append(f"w normalized: {w_old.tolist()} -> {state.upv.w.tolist()}")
        changed = True
    if changed:
        state.add_log("CR★L4", " | ".join(notes))
    return changed


# ------------------ 包装问题（泛化） ------------------
class WrappedProblem(ElementwiseProblem):
    def __init__(self, base: ElementwiseProblem, upv: UPV, enable_cr: bool,
                 penalty_scale: float = 1.0, bias_lambda: float = 0.05):
        super().__init__(n_var=base.n_var, n_obj=base.n_obj, n_constr=0, xl=base.xl, xu=base.xu)
        self.base = base
        self.upv = upv
        self.enable_cr = enable_cr
        self.penalty_scale = penalty_scale
        self.bias_lambda = bias_lambda if enable_cr else 0.0
    def _evaluate(self, x, out, *args, **kwargs):
        tmp = {}; self.base._evaluate(x, tmp)
        f = tmp["F"].astype(float)
        P, _ = compute_penalty_and_feas(x, f, self.upv, self.enable_cr)
        bias = asf_distance(f, self.upv.w, self.upv.r)
        out["F"] = f + (self.penalty_scale * P + self.bias_lambda * bias)


# ------------------ 优化器状态 ------------------
class OptState:
    def __init__(self, problem_id: str = "ZDT1", n_var: int = 30):
        self.lock = threading.Lock()
        self.problem_id = problem_id
        self.problem = make_problem(problem_id, n_var)
        self.n_var = int(self.problem.n_var)
        self.n_obj = int(self.problem.n_obj)

        self.upv = UPV(
            w=normalize_w(np.ones(self.n_obj)),
            r=np.zeros(self.n_obj),
            b=[(-np.inf, np.inf) for _ in range(self.n_obj)],
            D=[tuple(x) for x in self.problem.bounds]
        )

        # 算法 & 运行参数
        self.algo_name = "NSGA-II"      # "NSGA-II" | "NSGA-III" | "MOEA/D" | "BO" | "CP/MILP" | "RL"
        self.pop_size = 80
        self.gen_per_tick = 10
        self.seed = 123

        # EMO 操作符参数
        self.cx_prob = 0.9
        self.cx_eta = 15.0
        self.mut_eta = 20.0
        self.mut_prob = None  # None => 1/n_var

        # 具体算法参数容器（面板初值）
        self.emo_params = {
            "cx_prob": 0.9, "cx_eta": 15.0, "mut_eta": 20.0, "mut_prob": None,
            "moead_neighbors": 15
        }
        self.bo_params = {
            "n_local_factor": 6.0, "n_global": 40, "kernel_l": 0.3,
            "beta_override": None, "radius_scale": 1.0
        }
        self.cpm_params = {
            "use_pulp": False, "sample_size": 600, "lambda_pen": 1.0, "ridge_l2": 0.0
        }
        self.rl_params = {
            "N_mult": 12, "elite_frac": 0.25, "radius_scale": 1.0, "sigma": 0.15
        }
        try:
            import pulp  # noqa
            self.has_pulp = True
        except Exception:
            self.has_pulp = False

        # 运行数据
        self.pop_X = None
        self.pop_F = None
        self.hv_ref = np.array([1.0, 1.0], dtype=float)

        self.history_hv = []
        self.history_fmin = []
        self.history_feasible = []
        self.history_penalty = []
        self.history_upv = []
        self.history_logs: List[str] = []

        # ==== 实验模式与日志 ====
        self.expt_mode = "upv_full"  # 'single' | 'naive' | 'upv_full'
        self.logging_on = False
        self.run_id = None
        self.logger = None  # type: Optional[ExperimentLogger]

        # 计数器
        self.tick_count = 0
        self.eval_count = 0
        self.n_atoms_nl_total = 0
        self.n_atoms_visual_total = 0
        self.n_pairs_rank_total = 0
        self.n_clarify_total = 0

        # 收敛提示
        self.converged = False
        self.conv_hint = ""

        self.running = False
        self.stop_event = threading.Event()
        self.bt_pairs = []
        self.pcp_ranges: Dict[int, Tuple[float, float]] = {}

        # BO 状态
        self._bo_X = None
        self._bo_F = None

        # RL 状态
        self._rl_mean = None
        self._rl_std = None

        # L5 队列
        self.l5_queue: List[Dict[str, Any]] = []
        self._l5_counter: int = 0

    # ---- L5 接口 ----
    def enqueue_l5(self, kind: str, text: str, choices: Optional[List[str]] = None):
        self.l5_queue.append({"kind": kind, "text": text, "choices": choices or []})
    def pop_all_l5(self) -> List[Dict[str, Any]]:
        items = list(self.l5_queue)
        self.l5_queue.clear()
        return items
    def l5_scan_conflicts(self, latest_feas: Optional[float] = None):
        # 仅在 UPV Full + CR 时才启用主动澄清
        if not (self.enable_cr and getattr(self, "expt_mode", "upv_full") == "upv_full"):
            return
        # 1) D vs bounds
        for i, (lo, hi) in enumerate(self.upv.D or []):
            blo, bhi = float(self.problem.bounds[i, 0]), float(self.problem.bounds[i, 1])
            if lo > bhi or hi < blo:
                self.enqueue_l5("clarify", f"x{i+1} 区间 {lo:.3f}..{hi:.3f} 与边界 {blo:.3f}..{bhi:.3f} 无交集。",
                                ["自动裁剪到边界", "保持不变"])
        # 2) C_h(var) vs D
        for c in self.upv.C_h:
            if c.kind != "var": continue
            lo, hi = self.upv.D[c.idx]
            if c.sign == "<=" and lo > c.value:
                self.enqueue_l5("confirm", f"硬约束 x{c.idx+1} ≤ {c.value} 与当前 D 下界 {lo:.3f} 冲突。",
                                [f"把 D 上界裁到 {c.value}", "保持不变"])
            if c.sign == ">=" and hi < c.value:
                self.enqueue_l5("confirm", f"硬约束 x{c.idx+1} ≥ {c.value} 与当前 D 上界 {hi:.3f} 冲突。",
                                [f"把 D 下界提到 {c.value}", "保持不变"])
        # 3) r vs b
        for j, (blo, bhi) in enumerate(self.upv.b):
            if np.isfinite(blo) and self.upv.r[j] < blo:
                self.enqueue_l5("clarify", f"愿望 r{j+1}={self.upv.r[j]:.3f} 低于带宽下界 {blo:.3f}。",
                                ["夹取到带宽内", "保持不变"])
            if np.isfinite(bhi) and self.upv.r[j] > bhi:
                self.enqueue_l5("clarify", f"愿望 r{j+1}={self.upv.r[j]:.3f} 高于带宽上界 {bhi:.3f}。",
                                ["夹取到带宽内", "保持不变"])
        # 4) 可行率过低
        feas = latest_feas if latest_feas is not None else (self.history_feasible[-1] if self.history_feasible else 0.0)
        if feas < 0.1:
            self.enqueue_l5("warn", f"当前可行率仅 {feas:.2f}，建议放宽 D 或降低软罚。", ["放宽 D（+20%）", "降低软罚（×0.9）", "保持不变"])

    # 兼容旧调用名
    def _l5_proactive_check(self, latest_feas: Optional[float] = None):
        try:
            return self.l5_scan_conflicts(latest_feas=latest_feas)
        except Exception as e:
            self.add_log("L5", f"proactive_check error: {e}")
            return None

    # ---- 杂项 ----
    def add_log(self, kind: str, msg: str):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {kind}: {msg}"
        self.history_logs.append(line)
        if len(self.history_logs) > 1000:
            self.history_logs = self.history_logs[-1000:]

    def init_population(self):
        X0 = np.random.uniform(low=self.problem.xl, high=self.problem.xu, size=(self.pop_size, self.n_var))
        self.pop_X = X0
        F_list = []
        for i in range(self.pop_size):
            out = {}; self.problem._evaluate(X0[i], out); F_list.append(out["F"])
        self.pop_F = np.array(F_list)

        # HV 参考点：以初始种群最差值 + 10% 边距锁定
        F = self.pop_F
        maxv = np.max(F, axis=0)
        span = np.maximum(np.abs(maxv), 1.0)
        self.hv_ref = maxv + 0.1 * span

        self.log_upv()
        self.add_log("INIT", f"population={self.pop_size}, n_var={self.n_var}")

        # 初始化 BO/RL 状态
        self._bo_X = self.pop_X.copy()
        self._bo_F = self.pop_F.copy()
        self._rl_mean = np.mean(self.pop_X, axis=0)
        self._rl_std = np.std(self.pop_X, axis=0) + 1e-3

    def set_problem(self, problem_id: str, n_var: int):
        self.problem_id = problem_id
        self.problem = make_problem(problem_id, int(n_var))
        self.n_var = int(self.problem.n_var)
        self.n_obj = int(self.problem.n_obj)

        # 对齐 UPV
        self.upv.w = normalize_w(np.ones(self.n_obj))
        self.upv.r = np.zeros(self.n_obj)
        self.upv.b = [(-np.inf, np.inf) for _ in range(self.n_obj)]
        self.upv.D = [tuple(x) for x in self.problem.bounds]

        # 清理历史
        self.pop_X = None; self.pop_F = None
        self.history_hv.clear(); self.history_fmin.clear()
        self.history_feasible.clear(); self.history_penalty.clear()
        self.history_upv.clear()
        self.pcp_ranges.clear()

        self.add_log("PROBLEM", f"switch to {self.problem_id} (n_var={self.n_var})")

    def log_upv(self):
        self.history_upv.append({
            "w1": float(self.upv.w[0]), "w2": float(self.upv.w[1]),
            "r1": float(self.upv.r[0]), "r2": float(self.upv.r[1]),
            "tau": float(self.upv.tau),
            "cvar": float(self.upv.rho.get("cvar_alpha", 1.0))
        })

    # ---- EMO 构建 ----
    def _build_emo_algorithm(self):
        crossover = SBX(prob=self.cx_prob, eta=self.cx_eta)
        mut_prob = (1.0 / self.n_var) if (self.mut_prob is None) else float(self.mut_prob)
        mutation = PM(prob=mut_prob, eta=self.mut_eta)
        sampling = self.pop_X if self.pop_X is not None else FloatRandomSampling()

        if self.algo_name == "NSGA-III":
            ref_dirs = get_reference_directions("das-dennis", self.n_obj, n_points=self.pop_size)
            algo = NSGA3(pop_size=len(ref_dirs),
                         sampling=sampling, crossover=crossover, mutation=mutation,
                         eliminate_duplicates=True, ref_dirs=ref_dirs)
            return algo
        if self.algo_name == "MOEA/D":
            ref_dirs = get_reference_directions("das-dennis", self.n_obj, n_points=self.pop_size)
            algo = MOEAD(ref_dirs=ref_dirs, n_neighbors=min(self.emo_params["moead_neighbors"], len(ref_dirs)-1),
                         sampling=sampling, crossover=crossover, mutation=mutation,
                         eliminate_duplicates=True)
            return algo
        return NSGA2(pop_size=self.pop_size, sampling=sampling,
                     crossover=crossover, mutation=mutation, eliminate_duplicates=True)

    # ---- 单 step（根据算法分流）----
    def step_opt(self):
        if self.algo_name in ("NSGA-II", "NSGA-III", "MOEA/D"):
            self._step_emo()
        elif self.algo_name == "BO":
            self._step_bo()
        elif self.algo_name == "CP/MILP":
            self._step_cpm()
        elif self.algo_name == "RL":
            self._step_rl()
        else:
            self._step_emo()

    
    def _update_history(self):
        # 统计
        X, F = self.pop_X, self.pop_F
        hv = float(HV(ref_point=self.hv_ref)(F))
        fmin = np.min(F, axis=0).tolist()
        feas_flags, penalties = [], []
        for i in range(len(X)):
            tmp = {}; self.problem._evaluate(X[i], tmp)
            rawf = tmp["F"].astype(float)
            P, feas = compute_penalty_and_feas(X[i], rawf, self.upv, self.enable_cr)
            feas_flags.append(feas); penalties.append(P)
        feas_ratio = float(np.mean(feas_flags)) if len(feas_flags) else 0.0
        mean_penalty = float(np.mean(penalties)) if len(penalties) else 0.0

        self.history_hv.append(hv)
        self.history_fmin.append(fmin)
        self.history_feasible.append(feas_ratio)
        self.history_penalty.append(mean_penalty)

        # CR 动作（仅 Full）
        if self.enable_cr and getattr(self, "expt_mode", "upv_full") == "upv_full":
            l1_project_aspiration(self, threshold=0.12, eta=0.25)
            l3_adapt_soft_penalties(self, target_feas=(0.6,0.85), up_gain=1.12, down_gain=0.95)
        self.log_upv()

        # L5 主动澄清扫描（仅 Full）
        self.l5_scan_conflicts(latest_feas=feas_ratio)

        # IGD（有解析式参考前沿时）
        R = get_reference_pf(self.problem_id)
        igd = compute_igd(F, R) if R is not None else None

        # 累计 tick
        self.tick_count = int(getattr(self, "tick_count", 0)) + 1

        # 收敛检测：窗口=10，HV 提升<0.5%，且 IGD 提升<1% 或 IGD<1e-3，且近10轮可行率≥0.95
        try:
            W = 10
            if len(self.history_hv) >= W+1:
                hv0 = max(1e-9, abs(self.history_hv[-W-1]))
                d_hv = (self.history_hv[-1] - self.history_hv[-W-1]) / hv0
                igd_ok = True
                if len(self.history_penalty) >= W+1:
                    pass
                if R is not None and len([x for x in [igd] if x is not None])>0:
                    igd_hist = []
                    # 近 W+1 个 IGD 简单重算（开销小）
                    for k in range(max(0, len(self.history_hv)-W-1), len(self.history_hv)):
                        igd_hist.append(compute_igd(self.pop_F, R))
                    if len(igd_hist)>=2:
                        d_igd = (igd_hist[-1] - igd_hist[0]) / max(1e-9, abs(igd_hist[0]))
                        igd_ok = (abs(d_igd) < 0.01) or (igd is not None and igd < 1e-3)
                feas_recent = self.history_feasible[-W:]
                feas_ok = (np.median(feas_recent) >= 0.95) if len(feas_recent)==W else False
                self.converged = (d_hv < 0.005) and igd_ok and feas_ok
                if self.converged:
                    self.conv_hint = f"Converged: ΔHV({W})={d_hv:.3%}; feas≈{np.median(feas_recent):.2f}"
                else:
                    self.conv_hint = ""
        except Exception:
            # 保守处理
            self.converged = False

        # 记录行
        if getattr(self, "logging_on", False) and self.logger is not None:
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": self.run_id or "",
                "mode": self.expt_mode,
                "problem": self.problem_id,
                "algo": self.algo_name,
                "pop_size": self.pop_size,
                "gen_per_tick": self.gen_per_tick,
                "seed": self.seed,
                "tick": self.tick_count,
                "evals_cum": getattr(self, "eval_count", 0),
                "hv": hv,
                "igd": igd,
                "feasible_ratio": feas_ratio,
                "mean_penalty": mean_penalty,
                "w1": float(self.upv.w[0]), "w2": float(self.upv.w[1]),
                "r1": float(self.upv.r[0]), "r2": float(self.upv.r[1]),
                "tau": float(self.upv.tau),
                "cvar": float(self.upv.rho.get("cvar_alpha", 1.0)),
                "cr_on": int(self.enable_cr),
                "n_atoms_nl": getattr(self, "n_atoms_nl_total", 0),
                "n_atoms_visual": getattr(self, "n_atoms_visual_total", 0),
                "n_pairs_rank": getattr(self, "n_pairs_rank_total", 0),
                "n_clarify_total": getattr(self, "n_clarify_total", 0),
            }
            try:
                self.logger.log_tick(row)
                # 每 10 tick 保存一次前沿
                if self.tick_count % 10 == 0:
                    self.logger.dump_population(self.pop_X, self.pop_F, tick=self.tick_count)
            except Exception as e:
                self.add_log("LOG", f"write failed: {e}")

    def _step_emo(self):
        algo = self._build_emo_algorithm()
        term = get_termination("n_gen", self.gen_per_tick)
        wrapped = WrappedProblem(self.problem, self.upv, self.enable_cr, penalty_scale=1.0, bias_lambda=0.05)
        res = minimize(wrapped, algo, termination=term, seed=self.seed, verbose=False, save_history=False)
        new_pop = res.pop
        self.pop_X, self.pop_F = new_pop.get("X"), new_pop.get("F")
        
        # 累计评估数（估算）：pop_size × gen_per_tick
        self.eval_count = int(getattr(self, "eval_count", 0) + self.pop_size * self.gen_per_tick)

        # 累计评估数（估算）：pop_size × gen_per_tick
        self.eval_count = int(getattr(self, "eval_count", 0) + self.pop_size * self.gen_per_tick)
        self._update_history()

    def _step_bo(self):
        # 轻量 BO：GPR 每目标 + 候选（局部+全局） + ASF-UCB-罚项 评分
        rng = np.random.default_rng(self.seed + len(self.history_hv))
        X_obs = self._bo_X; F_obs = self._bo_F
        if X_obs is None or len(X_obs) < 5:
            X_obs, F_obs = self.pop_X, self.pop_F

        # GPR 拟合
        kernel = RBF(length_scale=self.bo_params["kernel_l"]) + WhiteKernel(noise_level=1e-6)
        gprs = []
        for j in range(self.n_obj):
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, random_state=self.seed)
            gpr.fit(X_obs, F_obs[:, j])
            gprs.append(gpr)

        # 候选：局部（基于 tau 半径）+ 全局
        n_local = int(self.bo_params["n_local_factor"] * self.pop_size)
        n_global = int(self.bo_params["n_global"])
        # 局部中心取最优 asf 个体
        d = np.array([asf_distance(F_obs[i], self.upv.w, self.upv.r) for i in range(len(F_obs))])
        x_center = X_obs[int(np.argmin(d))]
        radius = (0.05 + 0.45 * self.upv.tau) * float(self.bo_params["radius_scale"])
        xl, xu = self.problem.xl, self.problem.xu
        # 采样
        X_loc = np.clip(rng.normal(x_center, radius, size=(n_local, self.n_var)), xl, xu)
        X_glb = rng.uniform(xl, xu, size=(n_global, self.n_var))
        # 遵守 D
        if self.upv.D:
            for i,(lo,hi) in enumerate(self.upv.D):
                X_loc[:,i] = np.clip(X_loc[:,i], lo, hi)
                X_glb[:,i] = np.clip(X_glb[:,i], lo, hi)
        X_cand = np.vstack([X_loc, X_glb])

        # 评分：ASF - beta*std + 罚项
        alpha = float(self.upv.rho.get("cvar_alpha", 1.0))
        beta = float(self.bo_params["beta_override"]) if self.bo_params["beta_override"] is not None else (1.0 + 2.0*(1.0 - alpha))
        mu = []; std = []
        for j,gpr in enumerate(gprs):
            m, s = gpr.predict(X_cand, return_std=True)
            mu.append(m.reshape(-1,1)); std.append(s.reshape(-1,1))
        MU = np.hstack(mu); SD = np.hstack(std)
        ASF = np.array([asf_distance(MU[i], self.upv.w, self.upv.r) for i in range(len(X_cand))]).reshape(-1,1)
        score = ASF - beta * np.mean(SD, axis=1, keepdims=True)

        # 简单罚项（用均值做近似）
        P = []
        for i in range(len(X_cand)):
            f_hat = MU[i]
            p,_ = compute_penalty_and_feas(X_cand[i], f_hat, self.upv, self.enable_cr)
            P.append(p)
        P = np.array(P).reshape(-1,1)
        score = score + P

        # 选前 pop_size 个作为新种群并真实评估
        idx = np.argsort(score.ravel())[:self.pop_size]
        X_new = X_cand[idx]
        F_list = []
        for i in range(len(X_new)):
            out = {}; self.problem._evaluate(X_new[i], out); F_list.append(out["F"])
        self.pop_X, self.pop_F = X_new, np.array(F_list)

        # 更新 BO 数据集
        self._bo_X = np.vstack([X_obs, X_new])[-(3*self.pop_size):]
        self._bo_F = np.vstack([F_obs, self.pop_F])[-(3*self.pop_size):]

        
        # 累计评估数：本轮评估候选数量
        try:
            self.eval_count = int(getattr(self, "eval_count", 0) + len(X_new))
        except Exception:
            pass
        self._update_history()

    def _step_cpm(self):
        # 轻量 CP/MILP 近似：线性代理 + 候选采样 + 代理打分
        rng = np.random.default_rng(self.seed + len(self.history_hv))
        # 数据
        X_obs = self.pop_X; F_obs = self.pop_F
        # 线性代理
        models = []
        for j in range(self.n_obj):
            y = F_obs[:, j]
            if self.cpm_params["ridge_l2"] > 0:
                mdl = Ridge(alpha=self.cpm_params["ridge_l2"], fit_intercept=True)
            else:
                mdl = Ridge(alpha=1e-8, fit_intercept=True)
            mdl.fit(X_obs, y)
            models.append(mdl)

        # 候选
        N = self.cpm_params["sample_size"]
        xl, xu = self.problem.xl, self.problem.xu
        X_cand = rng.uniform(xl, xu, size=(N, self.n_var))
        if self.upv.D:
            for i,(lo,hi) in enumerate(self.upv.D):
                X_cand[:,i] = np.clip(X_cand[:,i], lo, hi)

        # 代理预测 + 打分（ASF + 罚项）
        MU = np.column_stack([mdl.predict(X_cand) for mdl in models])
        scores = []
        for i in range(len(X_cand)):
            f_hat = MU[i]
            a = asf_distance(f_hat, self.upv.w, self.upv.r)
            p,_ = compute_penalty_and_feas(X_cand[i], f_hat, self.upv, self.enable_cr)
            scores.append(a + self.cpm_params["lambda_pen"] * p)
        idx = np.argsort(scores)[:self.pop_size]
        X_new = X_cand[idx]
        F_list = []
        for i in range(len(X_new)):
            out = {}; self.problem._evaluate(X_new[i], out); F_list.append(out["F"])
        self.pop_X, self.pop_F = X_new, np.array(F_list)
        
        # 累计评估数：本轮评估候选数量
        try:
            self.eval_count = int(getattr(self, "eval_count", 0) + len(X_new))
        except Exception:
            pass

        # 累计评估数：本轮评估候选数量
        try:
            self.eval_count = int(getattr(self, "eval_count", 0) + len(X_new))
        except Exception:
            pass
        self._update_history()

    def _step_rl(self):
        # 轻量 CEM：以 ASF+罚项 为 cost，单轮更新
        rng = np.random.default_rng(self.seed + len(self.history_hv))
        N = max(int(self.rl_params["N_mult"] * self.pop_size), 300)
        elite_frac = float(self.rl_params["elite_frac"])
        n_elite = max(10, int(elite_frac * N))
        radius = (0.05 + 0.45 * self.upv.tau) * float(self.rl_params["radius_scale"])
        xl, xu = self.problem.xl, self.problem.xu

        m = self._rl_mean if self._rl_mean is not None else np.mean(self.pop_X, axis=0)
        s = self._rl_std if self._rl_std is not None else np.std(self.pop_X, axis=0) + 1e-3
        s = np.maximum(s, 1e-3)

        X = rng.normal(m, radius * s, size=(N, self.n_var))
        X = np.clip(X, xl, xu)
        if self.upv.D:
            for i,(lo,hi) in enumerate(self.upv.D):
                X[:,i] = np.clip(X[:,i], lo, hi)

        # 评分
        costs = []
        F_list = []
        for i in range(N):
            out = {}; self.problem._evaluate(X[i], out); f = out["F"].astype(float)
            F_list.append(f)
            a = asf_distance(f, self.upv.w, self.upv.r)
            p,_ = compute_penalty_and_feas(X[i], f, self.upv, self.enable_cr)
            costs.append(a + p)
        F = np.array(F_list)
        idx = np.argsort(costs)[:n_elite]
        elite = X[idx]
        # 更新分布
        self._rl_mean = 0.7 * m + 0.3 * np.mean(elite, axis=0)
        self._rl_std  = 0.7 * s + 0.3 * (np.std(elite, axis=0) + 1e-3)

        # 新种群 = 最优 pop_size 个
        idx2 = np.argsort(costs)[:self.pop_size]
        self.pop_X = X[idx2]
        self.pop_F = F[idx2]
        
        # 累计评估数：评估 N 个采样
        try:
            self.eval_count = int(getattr(self, "eval_count", 0) + N)
        except Exception:
            pass

        # 累计评估数：评估 N 个采样
        try:
            self.eval_count = int(getattr(self, "eval_count", 0) + N)
        except Exception:
            pass
        self._update_history()


# ------------------ Pareto 工具 ------------------
def is_nondominated(F: np.ndarray) -> np.ndarray:
    n = len(F)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]: continue
        for j in range(n):
            if i == j: continue
            if all(F[j] <= F[i]) and any(F[j] < F[i]):
                nd[i] = False
                break
    return nd

def crowding_distance(F: np.ndarray) -> np.ndarray:
    m, n_obj = F.shape
    cd = np.zeros(m)
    if m == 0: return cd
    if m <= 2: cd[:] = np.inf; return cd
    for j in range(n_obj):
        order = np.argsort(F[:, j])
        cd[order[0]] = cd[order[-1]] = np.inf
        fmin, fmax = F[order, j][0], F[order, j][-1]
        denom = (fmax - fmin) if (fmax - fmin) > 1e-12 else 1.0
        for k in range(1, m - 1):
            cd[order[k]] += (F[order[k + 1], j] - F[order[k - 1], j]) / denom
    return cd

def select_representative_indices(F: np.ndarray, k: int = 5) -> List[int]:
    n = len(F)
    if n == 0: return []
    nd_mask = is_nondominated(F)
    nd_idx = np.where(nd_mask)[0]
    if len(nd_idx) >= k:
        F_nd = F[nd_idx]
        cd = crowding_distance(F_nd)
        order_local = np.argsort(-cd)[:k]
        return [int(nd_idx[i]) for i in order_local]
    taken = list(map(int, nd_idx.tolist()))
    remain = [i for i in range(n) if i not in taken]
    if remain:
        w, r = state.upv.w, state.upv.r
        d = np.array([asf_distance(F[i], w, r) for i in remain])
        need = max(0, k - len(taken))
        picks = list(np.argsort(d)[:need])
        taken += [int(remain[p]) for p in picks]
    return taken[:k]

def _render_cards_fallback(id_list):
    """
    将候选解的索引列表渲染为一组 dbc.Card。
    要求:
      - state.pop_F: numpy.ndarray，形状 (N, M) 目标值
      - state.pop_X: numpy.ndarray，形状 (N, D) 决策变量
    返回: list[Component]
    """
    children = []

    # 保护：空列表直接返回
    if not id_list:
        return children

    F = getattr(state, "pop_F", None)
    X = getattr(state, "pop_X", None)

    # 为了读起来更整齐，最多展示前 5 个变量（多了会太宽）
    def _fmt_vars(row, max_k=5):
        try:
            vals = [float(v) for v in row[:max_k]]
        except Exception:
            return "x: N/A"
        s = ", ".join(f"{v:.3g}" for v in vals)
        return f"x[:{min(max_k, len(row))}] = [{s}]"

    for idx in id_list:
        try:
            i = int(idx)
        except Exception:
            continue

        # 标题与子标题（目标值）
        title = f"Candidate #{i}"
        subtitle = "f: N/A"

        if F is not None and 0 <= i < len(F):
            fvals = []
            try:
                fvals = [float(v) for v in F[i]]
            except Exception:
                fvals = []
            if len(fvals) >= 2:
                subtitle = f"f1={fvals[0]:.4f}, f2={fvals[1]:.4f}"
            elif len(fvals) == 1:
                subtitle = f"f1={fvals[0]:.4f}"

        body_children = [
            html.H6(title, className="card-title mb-1"),
            html.P(subtitle, className="card-subtitle text-muted small mb-2"),
        ]

        # 变量预览
        if X is not None and 0 <= i < len(X):
            body_children.append(
                html.Div(_fmt_vars(X[i]), className="small text-muted")
            )

        # 卡片
        card = dbc.Card(
            dbc.CardBody(body_children),
            class_name="me-2 mb-2",
            style={"minWidth": "220px"},
        )
        children.append(card)

    return children

# ------------------ BT 权重学习 ------------------
def fit_bt_weights(pairs: List[Tuple[np.ndarray, np.ndarray, int]], w0: np.ndarray) -> np.ndarray:
    if not pairs: return w0
    X_list, y_list = [], []
    for fA, fB, yAB in pairs:
        X_list.append((fB - fA).reshape(1, -1))
        y_list.append(1 if int(yAB) == 1 else 0)
    X = np.vstack(X_list); y = np.array(y_list, dtype=int)
    if np.unique(y).size < 2:
        X = np.vstack([X, -X]); y = np.concatenate([y, 1 - y])
    if np.allclose(X, 0): return w0
    clf = LogisticRegression(C=10.0, fit_intercept=False, solver="lbfgs", max_iter=500)
    clf.fit(X, y)
    w = clf.coef_.reshape(-1)
    w = np.maximum(w, 1e-12); w = w / w.sum()
    return w if np.all(np.isfinite(w)) else w0


# ------------------ 后台优化线程 ------------------
def optimizer_loop(state: "OptState"):
    while not state.stop_event.is_set():
        with state.lock:
            if state.running:
                state.step_opt()
        time.sleep(0.15)


# ------------------ 构建 App ------------------
external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "UPV Interactive Optimization"
app.config.suppress_callback_exceptions = True
# 允许重复输出的初始重复
app.config.prevent_initial_callbacks = 'initial_duplicate'

# 内联 CSS（包含聊天区固定高度 & 拖拽样式）
INLINE_CSS = """
.neo-slider .rc-slider-track { background: linear-gradient(90deg,#00eaff,#8a2be2); height: 8px; box-shadow: 0 0 12px rgba(0,234,255,.5); }
.neo-slider .rc-slider-rail  { background: rgba(255,255,255,.15); height: 8px; backdrop-filter: blur(2px); }
.neo-slider .rc-slider-handle {
  border: 2px solid rgba(255,255,255,.6); width: 18px; height: 18px; margin-top: -5px;
  background: radial-gradient(circle at 30% 30%, #ffffff 0%, #d9f7ff 35%, #7ee0ff 70%, #00b3ff 100%);
  box-shadow: 0 0 18px rgba(0,179,255,.8), inset 0 0 4px rgba(255,255,255,.8);
}
.neo-slider .rc-slider-handle-dragging { box-shadow: 0 0 22px rgba(138,43,226,.9), inset 0 0 6px rgba(255,255,255,.9); }
.neo-slider .rc-slider-dot { display: none; }
.neo-slider .rc-slider-mark { font-size: 11px; }

/* 聊天卡片：固定高度 + 上方滚动区、下方输入区 */
.chat-card-body { height: 820px; display: flex; flex-direction: column; }
.chat-pane { flex: 1 1 auto; height: auto; max-height: none; overflow-y: auto; overflow-x: hidden;
  padding: 10px 8px; background: rgba(255,255,255,.55); border-radius: 14px; }
.bubble { max-width: 86%; padding: 10px 12px; border-radius: 14px; margin: 8px 0; white-space: pre-wrap; word-break: break-word; }
.bubble.user { margin-left: auto; background: linear-gradient(135deg,#c7f5ff,#eaf7ff); box-shadow: 0 0 10px rgba(0,174,255,.25); }
.bubble.bot  { margin-right: auto; background: linear-gradient(135deg,#efe9ff,#f6f1ff); box-shadow: 0 0 10px rgba(128,0,255,.18); }
.nl-input-row { margin-top: 8px; }
textarea#nl-input { resize: vertical; min-height: 80px; max-height: 200px; }

/* 方案排序拖拽 */
.rank-li { list-style: none; border: 1px solid #eee; background: #fff; border-radius: 12px;
           padding: 10px 12px; margin-bottom: 8px; display: flex; align-items: center; gap: 10px; }
.drag-handle { cursor: grab; user-select: none; -webkit-user-drag: element; display: inline-block; padding: 0 6px; font-weight: 600; }
.drag-handle:active { cursor: grabbing; }
.rank-li.dragging { opacity: .72; transform: translateZ(0) scale(1.01); }
#rank-list { list-style: none; padding-left: 0; margin: 0; }

/* 右侧列更紧凑 */
.right-col .card { margin-bottom: 16px; }
"""

app.index_string = f"""
<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{INLINE_CSS}</style>
    <script>
    // 原生 HTML5 DnD：拖拽的是 <li> 本体，但只允许从 .drag-handle 发起
    window.__rankDnDInit = function() {{
        var list = document.getElementById('rank-list');
        if (!list || list.__dnd_inited) return;
        list.__dnd_inited = true;

        function bindOneLI(li) {{
            if (!li || li.__dnd_bound) return;
            li.__dnd_bound = true;
            var handle = li.querySelector('.drag-handle');
            li.setAttribute('draggable', 'true');
            li.addEventListener('dragstart', function(e) {{
                if (!handle || !handle.contains(e.target)) {{
                    e.preventDefault(); return false;
                }}
                try {{ e.dataTransfer.setData('text/plain', 'rank'); }} catch (err) {{}}
                e.dataTransfer.effectAllowed = 'move';
                try {{ e.dataTransfer.dropEffect = 'move'; }} catch (err) {{}}
                li.classList.add('dragging');
            }});
            li.addEventListener('dragend', function(e) {{
                li.classList.remove('dragging');
                var order = Array.from(list.querySelectorAll('li[data-id]')).map(function(x) {{
                    return parseInt(x.dataset.id);
                }});
                var ev = new CustomEvent('sortupdate', {{ detail: {{ order: order }}, bubbles: true }});
                list.dispatchEvent(ev);
            }});
        }}

        function getAfterElement(container, y) {{
            var els = Array.from(container.querySelectorAll('li[data-id]:not(.dragging)'));
            var closest = null;
            var closestOffset = Number.NEGATIVE_INFINITY;
            for (var i = 0; i < els.length; i++) {{
                var box = els[i].getBoundingClientRect();
                var offset = y - box.top - box.height / 2;
                if (offset < 0 && offset > closestOffset) {{
                    closestOffset = offset; closest = els[i];
                }}
            }}
            return closest;
        }}

        list.addEventListener('dragover', function(e) {{
            e.preventDefault();
            try {{ e.dataTransfer.dropEffect = 'move'; }} catch (err) {{}}
            var dragging = list.querySelector('li.dragging');
            if (!dragging) return;
            var after = getAfterElement(list, e.clientY);
            if (after == null) {{ list.appendChild(dragging); }}
            else {{ list.insertBefore(dragging, after); }}
        }});
        list.addEventListener('dragenter', function(e) {{ e.preventDefault(); }});
        list.addEventListener('drop', function(e) {{ e.preventDefault(); }});

        Array.from(list.querySelectorAll('li[data-id]')).forEach(bindOneLI);
        var obs = new MutationObserver(function() {{
            Array.from(list.querySelectorAll('li[data-id]')).forEach(bindOneLI);
        }});
        obs.observe(list, {{ childList: true, subtree: true }});
    }};

    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(window.__rankDnDInit, 50);
    }});
    </script>
  </head>
  <body>
    {{%app_entry%}}
    <footer>
      {{%config%}}
      {{%scripts%}}
      {{%renderer%}}
    </footer>
  </body>
</html>
"""

# 刻度常量
MARKS_01 = {0: "0", 0.5: "0.5", 1: "1"}


# ===== 图表工厂 =====
# === Safe figure helpers: never assume `state` exists at layout time ===
import numpy as np
import plotly.graph_objs as go

def _get_state():
    """Return global state if it exists, else None."""
    return globals().get("state", None)

def _has_attr_list(st, attr_name: str) -> bool:
    """Check st.attr_name exists and is a non-empty sequence."""
    return (st is not None) and hasattr(st, attr_name) and bool(getattr(st, attr_name))

def _empty_fig(title: str):
    """An empty placeholder figure used before state/data is available."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis_title=None, yaxis_title=None
    )
    fig.add_annotation(
        text="Waiting for data…",
        showarrow=False,
        x=0.5, y=0.5, xref="paper", yref="paper",
        font=dict(color="gray")
    )
    return fig

def _series_fig(y, title: str, ylab: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines+markers", name=title))
    fig.update_layout(
        title=title,
        xaxis_title="Tick",
        yaxis_title=ylab,
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ---- Replace your original figure_hv with this safe version ----
def figure_hv(st=None):
    st = st or _get_state()
    if not _has_attr_list(st, "history_hv"):
        return _empty_fig("Hypervolume")
    y = [float(v) for v in getattr(st, "history_hv", [])]
    return _series_fig(y, "Hypervolume", "HV")

# If you have IGD time series:
def figure_igd(st=None):
    st = st or _get_state()
    if not _has_attr_list(st, "history_igd"):
        return _empty_fig("IGD")
    y = [float(v) for v in getattr(st, "history_igd", [])]
    return _series_fig(y, "IGD", "IGD")

# Feasible ratio:
def figure_feasible(st=None):
    st = st or _get_state()
    if not _has_attr_list(st, "history_feasible"):
        return _empty_fig("Feasible Ratio")
    y = [float(v) for v in getattr(st, "history_feasible", [])]
    return _series_fig(y, "Feasible Ratio", "ratio")

# Mean penalty:
def figure_penalty(st=None):
    st = st or _get_state()
    if not _has_attr_list(st, "history_penalty"):
        return _empty_fig("Penalty")
    y = [float(v) for v in getattr(st, "history_penalty", [])]
    return _series_fig(y, "Penalty", "avg penalty")

# 2D Pareto front (F1 vs F2):
def figure_pf2d(st=None):
    st = st or _get_state()
    F = getattr(st, "pop_F", None)
    if F is None or len(F) == 0:
        return _empty_fig("Pareto Front (F1 vs F2)")
    F = np.asarray(F)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=F[:, 0], y=F[:, 1],
        mode="markers",
        name="Population",
        marker=dict(size=6, opacity=0.8)
    ))
    fig.update_layout(
        title="Pareto Front (F1 vs F2)",
        xaxis_title="f1", yaxis_title="f2",
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=30)
    )
    return fig


def figure_fmin(st=None):
    """
    安全版：布局阶段不依赖全局 state。
    - 若 state 不存在或没有 history_fmin，则返回占位图；
    - 支持 1 个或 2 个目标（arr 形状兼容）。
    """
    # 允许外部传入 st；否则尝试取全局 state
    st = st or globals().get("state", None)

    # 没有 state 或没有数据：返回占位图
    if st is None or not getattr(st, "history_fmin", None):
        fig = go.Figure()
        fig.update_layout(
            title="Best Objective Values",
            template="plotly_white",
            height=240,
            margin=dict(l=8, r=8, t=28, b=8),
        )
        fig.add_annotation(
            text="Waiting for data…",
            showarrow=False,
            x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(color="gray"),
        )
        return fig

    # 有数据：生成折线
    arr = np.array(getattr(st, "history_fmin", []), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    fig = go.Figure()
    if arr.shape[1] >= 1:
        fig.add_trace(go.Scatter(y=arr[:, 0], mode="lines+markers", name="min f1"))
    if arr.shape[1] >= 2:
        fig.add_trace(go.Scatter(y=arr[:, 1], mode="lines+markers", name="min f2"))

    fig.update_layout(
        title="Best Objective Values",
        template="plotly_white",
        height=240,
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis_title="Tick",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def figure_nd_scatter(st=None):
    """
    安全版：布局阶段不依赖全局 state。
    - 若 state/pop_F 不可用，返回占位图；
    - 有数据时绘制被支配点 & 非支配前沿，并标注愿望点 r。
    """
    # 允许外部传入 st；否则尝试取全局 state
    st = st or globals().get("state", None)

    # 没有 state 或没有种群数据：返回占位图
    if st is None or getattr(st, "pop_F", None) is None or len(getattr(st, "pop_F")) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Non-dominated Front",
            template="plotly_white",
            height=320,
            margin=dict(l=8, r=8, t=28, b=8),
            xaxis_title="f1",
            yaxis_title="f2",
        )
        fig.add_annotation(
            text="Waiting for data…",
            showarrow=False,
            x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(color="gray"),
        )
        return fig

    # 有数据：开始绘制
    F = np.asarray(st.pop_F, dtype=float)
    nd = is_nondominated(F)

    xlo, xhi = float(np.min(F[:, 0])), float(np.max(F[:, 0]))
    ylo, yhi = float(np.min(F[:, 1])), float(np.max(F[:, 1]))
    dx = max(1e-6, xhi - xlo)
    dy = max(1e-6, yhi - ylo)
    xr = [xlo - 0.1 * dx, xhi + 0.1 * dx]
    yr = [ylo - 0.1 * dy, yhi + 0.1 * dy]

    fig = go.Figure()
    # 被支配点
    if np.any(~nd):
        fig.add_trace(go.Scatter(
            x=F[~nd, 0], y=F[~nd, 1],
            mode="markers",
            marker=dict(size=7, color="#A0AEC0"),
            name="dominated"
        ))
    # 非支配前沿
    fig.add_trace(go.Scatter(
        x=F[nd, 0], y=F[nd, 1],
        mode="markers",
        marker=dict(size=9, color="#E74C3C", line=dict(color="#B03A2E", width=1)),
        name="Pareto front"
    ))
    # 愿望点 r（如果可用）
    try:
        r = np.asarray(st.upv.r, dtype=float)
        if r.size >= 2:
            fig.add_trace(go.Scatter(
                x=[float(r[0])], y=[float(r[1])],
                mode="markers",
                marker=dict(size=12, symbol="star", color="#2E86C1"),
                name="aspiration r"
            ))
    except Exception:
        pass

    fig.update_layout(
        height=320,
        margin=dict(l=8, r=8, t=28, b=8),
        template="plotly_white",
        title=f"Non-dominated Front ({getattr(st, 'problem_id', '?')})",
        xaxis_title="f1",
        yaxis_title="f2",
    )
    fig.update_xaxes(range=xr)
    fig.update_yaxes(range=yr)
    return fig


def figure_feas_penalty(st=None):
    """
    安全版：布局阶段不依赖全局 state。
    - 若没有数据，返回带“Waiting for data…”注释的占位图；
    - 有数据时同时画可行率（y1）与平均罚项（y2）。
    """
    st = st or globals().get("state", None)

    feas = list(getattr(st, "history_feasible", []) or [])
    pen  = list(getattr(st, "history_penalty", [])  or [])

    # 无数据：占位图
    if st is None or (len(feas) == 0 and len(pen) == 0):
        fig = go.Figure()
        fig.update_layout(
            title="Feasibility & Mean Penalty",
            template="plotly_white",
            height=240,
            margin=dict(l=8, r=8, t=28, b=8),
            yaxis=dict(title="Feasible ratio", range=[0, 1]),
            yaxis2=dict(title="Mean penalty", overlaying="y", side="right"),
        )
        fig.add_annotation(
            text="Waiting for data…",
            showarrow=False,
            x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(color="gray"),
        )
        return fig

    # 对齐长度（以较短的为准，避免长度不一致报错）
    n = min(len(feas), len(pen)) if (len(feas) > 0 and len(pen) > 0) else max(len(feas), len(pen))
    feas = feas[:n] if len(feas) >= n else feas
    pen  = pen[:n]  if len(pen)  >= n else pen
    x = list(range(1, n + 1))

    fig = go.Figure()
    if len(feas) > 0:
        fig.add_trace(go.Scatter(x=x, y=feas, mode="lines+markers", name="Feasible ratio", yaxis="y1"))
    if len(pen) > 0:
        fig.add_trace(go.Scatter(x=x, y=pen,  mode="lines+markers", name="Mean penalty",   yaxis="y2"))

    fig.update_layout(
        template="plotly_white",
        height=240,
        margin=dict(l=8, r=8, t=28, b=8),
        title="Feasibility & Mean Penalty",
        xaxis_title="Tick",
        yaxis=dict(title="Feasible ratio", range=[0, 1]),
        yaxis2=dict(title="Mean penalty", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def figure_upv_timeline(st=None):
    """
    安全版：布局阶段不依赖全局 state。
    - 无 state 或无 history_upv 时，返回占位图；
    - 有数据时，按 w1,w2,r1,r2,tau,cvar 依次绘线（缺列自动跳过）。
    """
    st = st or globals().get("state", None)

    # 无数据：占位图
    if st is None or not getattr(st, "history_upv", None):
        fig = go.Figure()
        fig.update_layout(
            title="UPV Slots Timeline",
            template="plotly_white",
            height=240,
            margin=dict(l=8, r=8, t=28, b=8),
        )
        fig.add_annotation(
            text="Waiting for data…",
            showarrow=False,
            x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(color="gray"),
        )
        return fig

    # 有数据：构图（容错：缺列跳过；缺值前向填充）
    df = pd.DataFrame(getattr(st, "history_upv", []))
    if not len(df):
        # 兜底：若 df 为空仍返回占位
        fig = go.Figure()
        fig.update_layout(
            title="UPV Slots Timeline",
            template="plotly_white",
            height=240,
            margin=dict(l=8, r=8, t=28, b=8),
        )
        return fig

    df = df.ffill().bfill()

    fig = go.Figure()
    for k in ["w1", "w2", "r1", "r2", "tau", "cvar"]:
        if k in df.columns:
            fig.add_trace(go.Scatter(y=df[k].astype(float), mode="lines+markers", name=k))

    fig.update_layout(
        template="plotly_white",
        height=240,
        margin=dict(l=8, r=8, t=28, b=8),
        title="UPV Slots Timeline",
        xaxis_title="Tick",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def build_pcp_fig(selected_axes=None):
    var_names = [f"x{i+1}" for i in range(6)] + ["f1","f2"]
    if selected_axes is None:
        selected_axes = var_names
    if state.pop_X is None or state.pop_F is None:
        return go.Figure()

    k = min(80, len(state.pop_X))
    data = {}
    for i in range(6):
        if i < state.n_var:
            data[f"x{i+1}"] = state.pop_X[:k, i]
        else:
            data[f"x{i+1}"] = np.zeros(k)
    data["f1"] = state.pop_F[:k, 0]
    data["f2"] = state.pop_F[:k, 1]
    df = pd.DataFrame(data)

    dims = []
    for i, name in enumerate(var_names):
        vals = df[name].values
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if np.isclose(hi - lo, 0):
            lo -= 1e-6; hi += 1e-6
        dim = dict(
            label=name, values=vals, range=[lo, hi],
            visible=(name in selected_axes) and not (name.startswith("x") and int(name[1:]) > state.n_var)
        )
        if i in state.pcp_ranges:
            dim["constraintrange"] = [state.pcp_ranges[i][0], state.pcp_ranges[i][1]]
        dims.append(dim)

    fig = go.Figure(data=[go.Parcoords(
        line=dict(color=df["f1"].values, colorscale="Viridis", showscale=False),
        dimensions=dims, labelfont=dict(size=12),
    )])
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=40, b=8), template="plotly_white")
    return fig


# ===== 聊天工具 =====
def render_chat_bubbles(messages: List[Dict[str, str]]):
    if not messages:
        return [html.Div("👋 请输入自然语言指令，例如：将 f2 的权重提高到 0.7，并把 x1 限制在 [0.2,0.5]。", className="bubble bot")]
    out = []
    for m in messages:
        role = m.get("role","bot")
        out.append(html.Div(m.get("text",""), className=f"bubble {'user' if role=='user' else 'bot'}"))
    return out

from typing import Dict, Any
import os

def apply_nl_and_update_upv(nl_text: str) -> Dict[str, Any]:
    """
    使用星火(Spark)大模型将自然语言解析为“补丁”(patch)，
    并直接对当前统一偏好向量(UPV)进行【增量】修改。
    """
    # 1) LLM -> patch（讯飞 OpenAI 兼容 /v1/chat/completions）
    patch = llm_nl_to_patch(nl_text)

    # 2) 应用补丁到“当前” UPV（保持内部结构不变）
    with state.lock:
        state.add_log("NL", f"text={nl_text}; patch={patch}")
        state.upv = apply_patch_to_upv(state.upv, patch, M=state.n_obj)

        # （可选）冲突消解与扫描
        if getattr(state, "enable_cr", False):
            try:
                l2_reconcile_D_vs_Ch(state)
                l4_goal_consistency(state)
            except Exception as _e:
                print("[CR] skipped:", _e)
        try:
            state.l5_scan_conflicts()
        except Exception:
            pass

        # 3) 回传“当前”UPV（非模板，不重置）
        u = state.upv
        upv_json = {
            "w": np.asarray(u.w, dtype=float).tolist(),
            "r": np.asarray(u.r, dtype=float).tolist(),
            "b": [[float(lo), float(hi)] for (lo, hi) in list(u.b)],
            "D": [list(map(float, d)) for d in (list(u.D) if getattr(u, "D", None) else [])],
            "tau": float(u.tau),
            "rho": {k: float(v) if isinstance(v, (int, float)) else v for k, v in dict(u.rho).items()},
        }
    return upv_json



# ===== 代表性方案 / 拖拽工具 =====
def render_rank_list_items(order_ids: List[int]):
    """把候选渲染成 <li>：只显示方案编号与目标值"""
    items = []
    for idx in order_ids:
        iidx = int(idx)
        f1, f2 = float(state.pop_F[iidx, 0]), float(state.pop_F[iidx, 1])
        items.append(
            html.Li(
                [
                    # html.Span("☰", className="drag-handle", draggable=False),
                    html.Span("☰", className="drag-handle"),
                    html.Span(f"方案 #{iidx}", className="fw-bold"),
                    html.Span(f"f1={f1:.4f}  |  f2={f2:.4f}", className="text-muted")
                ],
                className="rank-li",
                **{"data-id": iidx}
            )
        )
    return items


# ===== 顶部栏 =====
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            # 品牌标题
            dbc.Col(
                dbc.NavbarBrand(
                    "UPV-based Interactive Multi-Objective Optimization",
                    class_name="ms-2"
                ),
                md="auto"
            ),

            # 左侧：模式 + CR + 日志 开关
            dbc.Col(
                html.Div([
                    html.Div([
                        html.Span("模式", className="text-white-50 small me-2"),
                        dcc.Dropdown(
                            id="cfg-expt-mode",
                            options=[
                                {"label": "Single（PBO，仅排序）", "value": "single"},
                                {"label": "Naive（朴素多类型）", "value": "naive"},
                                {"label": "UPV (Full CR)", "value": "upv_full"},
                            ],
                            value="upv_full",
                            clearable=False,
                            style={"width": "220px"},
                        ),
                    ], className="d-flex align-items-center me-2"),

                    dbc.Checklist(
                        options=[{"label": " Enable Conflict Resolution (CR)", "value": "cr"}],
                        value=["cr"],
                        id="cr-toggle",
                        switch=True,
                        class_name="me-3"
                    ),

                    dbc.Checklist(
                        options=[{"label": " 记录日志", "value": "on"}],
                        value=["on"],                          # ✅ 默认开启日志
                        id="cfg-log-enable",
                        switch=True,
                        class_name="me-2 text-white"
                    ),
                ], className="d-flex align-items-center flex-wrap gap-2"),
                md=6
            ),

            # 中间：Run ID
            dbc.Col(
                dbc.Input(
                    id="cfg-run-id",
                    type="text",
                    placeholder="Run ID（可选）",
                    size="sm",
                    style={"width": "180px"}
                ),
                md="auto"
            ),

            # 右侧：操作按钮组
            dbc.Col(
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button("开始优化", id="btn-start", color="success", size="sm"),
                        dbc.Button("暂停", id="btn-pause", color="secondary", size="sm"),
                        dbc.Button("停止", id="btn-stop", color="danger", size="sm"),
                        dbc.Button("步进10代", id="btn-step", color="info", size="sm"),
                    ]),
                    html.Span(id="conv-hint", className="badge bg-success ms-3", style={"display": "none"})
                ], className="d-flex align-items-center"),
                md="auto"
            ),
        ], align="center", class_name="g-2 w-100")
    ]),
    color="primary",
    dark=True,
    class_name="mb-3 shadow-sm"
)

# —— 控件说明（气泡提示，无需回调）——
navbar_tooltips = html.Div([
    dbc.Tooltip(
        "选择实验模式：Single=单通道基线（不使用CR）；Naive=朴素多类型整合（不建模冲突）；UPV=我们的方法（开启L2/L4冲突协调与L5主动扫描）。注意：真正开始时以“确认开始”时的模式为准，系统会据此设置CR开关。",
        target="cfg-expt-mode",
        placement="bottom"
    ),
    dbc.Tooltip(
        "冲突消解开关（CR）。启动后会按模式自动设置：UPV=ON，其它=OFF；你也可以在运行中手动开关（若代码逻辑支持）。",
        target="cr-toggle",
        placement="bottom"
    ),
    dbc.Tooltip(
        "记录关键指标与事件到 logs/<run_id>/（若留空则自动生成 run_id）。默认开启。",
        target="cfg-log-enable",
        placement="bottom"
    ),
    dbc.Tooltip("可选的运行标识（用于日志目录命名）。", target="cfg-run-id", placement="bottom"),
    dbc.Tooltip("开始/继续优化。", target="btn-start", placement="bottom"),
    dbc.Tooltip("暂停优化线程。", target="btn-pause", placement="bottom"),
    dbc.Tooltip("停止并清理状态。", target="btn-stop", placement="bottom"),
    dbc.Tooltip("仅前进 10 代（调试用）。", target="btn-step", placement="bottom"),
])



# —— 默认值保持不变（若已在别处定义，可略过）——
DEFAULT_EMO_PARAMS = {
    "pop_size": 64, "gen_per_tick": 5,
    "cx_prob": 0.9, "cx_eta": 15.0, "mut_prob": 0.1, "mut_eta": 20.0,
    "moead_neighbors": 15
}
DEFAULT_BO_PARAMS  = {
    "init_points": 8, "batch_size": 4,
    "n_local_factor": 6.0, "n_global": 40,
    "kernel_l": 0.3, "beta_override": None, "radius_scale": 1.0
}
DEFAULT_CPM_PARAMS = {"max_nodes": 2000}
DEFAULT_RL_PARAMS  = {"sigma": 0.3}

def build_all_algo_param_panels(st=None):
    st = st or globals().get("state", None)
    ep = getattr(st, "emo_params", DEFAULT_EMO_PARAMS)
    bp = getattr(st, "bo_params", DEFAULT_BO_PARAMS)
    cp = getattr(st, "cpm_params", DEFAULT_CPM_PARAMS)
    rp = getattr(st, "rl_params", DEFAULT_RL_PARAMS)

    # EMO（交叉/变异）
    emo_ops = dbc.Row([
        dbc.Col([html.Div("交叉概率 (SBX)"),
                 dbc.Input(id="cfg-cxprob", type="number", min=0, max=1, step=0.01,
                           value=float(ep.get("cx_prob", DEFAULT_EMO_PARAMS["cx_prob"])))], width=3),
        dbc.Col([html.Div("交叉 η (SBX)"),
                 dbc.Input(id="cfg-cxeta", type="number", min=1, step=1,
                           value=float(ep.get("cx_eta", DEFAULT_EMO_PARAMS["cx_eta"])))], width=3),
        dbc.Col([html.Div("变异概率 (PM)"),
                 dbc.Input(id="cfg-mutprob", type="number", min=0, max=1, step=0.01,
                           value=float(ep.get("mut_prob", DEFAULT_EMO_PARAMS["mut_prob"])))], width=3),
        dbc.Col([html.Div("变异 η (PM)"),
                 dbc.Input(id="cfg-muteta", type="number", min=1, step=1,
                           value=float(ep.get("mut_eta", DEFAULT_EMO_PARAMS["mut_eta"])))], width=3),
    ], class_name="mb-3")

    moead_row = dbc.Row([
        dbc.Col([html.Div("MOEA/D 邻居数"),
                 dbc.Input(id="cfg-moead-nei", type="number", min=2, step=1,
                           value=int(ep.get("moead_neighbors", DEFAULT_EMO_PARAMS["moead_neighbors"])))], width=3),
    ], class_name="mb-3")

    # BO（保留已有基础 + 进阶）
    bo_basic = dbc.Row([
        dbc.Col([html.Div("BO 初始点数"),
                 dbc.Input(id="cfg-bo-init", type="number", min=1, step=1,
                           value=int(bp.get("init_points", DEFAULT_BO_PARAMS["init_points"])))], width=3),
        dbc.Col([html.Div("BO 批大小"),
                 dbc.Input(id="cfg-bo-batch", type="number", min=1, step=1,
                           value=int(bp.get("batch_size", DEFAULT_BO_PARAMS["batch_size"])))], width=3),
    ], class_name="mb-3")

    bo_advanced = dbc.Row([
        dbc.Col([html.Div("局部探索倍数 n_local_factor"),
                 dbc.Input(id="cfg-bo-nlocal", type="number", min=1, step=0.5,
                           value=float(bp.get("n_local_factor", DEFAULT_BO_PARAMS["n_local_factor"])))], width=4),
        dbc.Col([html.Div("全局候选数 n_global"),
                 dbc.Input(id="cfg-bo-nglobal", type="number", min=1, step=1,
                           value=int(bp.get("n_global", DEFAULT_BO_PARAMS["n_global"])))], width=4),
        dbc.Col([html.Div("核长度尺度 ℓ"),
                 dbc.Input(id="cfg-bo-l", type="number", min=1e-3, step=1e-2,
                           value=float(bp.get("kernel_l", DEFAULT_BO_PARAMS["kernel_l"])))], width=4),
    ], class_name="mb-3")

    bo_advanced2 = dbc.Row([
        dbc.Col([html.Div("β（留空=自动 by α）"),
                 dbc.Input(id="cfg-bo-beta", type="text",
                           value=("" if (bp.get("beta_override", None) is None) else str(bp["beta_override"])))], width=4),
        dbc.Col([html.Div("半径缩放 radius_scale"),
                 dbc.Input(id="cfg-bo-radius", type="number", min=0.1, step=0.1,
                           value=float(bp.get("radius_scale", DEFAULT_BO_PARAMS["radius_scale"])))], width=4),
    ], class_name="mb-3")

    cpm_row = dbc.Row([
        dbc.Col([html.Div("CP/MILP 最大节点数"),
                 dbc.Input(id="cfg-cpm-maxnodes", type="number", min=100, step=100,
                           value=int(cp.get("max_nodes", DEFAULT_CPM_PARAMS["max_nodes"])))], width=4),
    ], class_name="mb-3")

    rl_row = dbc.Row([
        dbc.Col([html.Div("RL σ (探索强度)"),
                 dbc.Input(id="cfg-rl-sigma", type="number", min=0.01, step=0.01,
                           value=float(rp.get("sigma", DEFAULT_RL_PARAMS["sigma"])))], width=4),
    ], class_name="mb-1")

    # ★ 关键：包一层带固定 id 的容器，供回调切换可见性
    return [
        html.Div([emo_ops, moead_row], id="panel-emo", style={"display": "block"}),        # 默认显示 EMO
        html.Div([bo_basic, bo_advanced, bo_advanced2], id="panel-bo", style={"display": "none"}),
        html.Div([cpm_row], id="panel-cpm", style={"display": "none"}),
        html.Div([rl_row], id="panel-rl", style={"display": "none"}),
    ]




# ===== 启动配置弹窗 =====
cfg_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("优化模型配置与确认")),
    dbc.ModalBody([
        dbc.Card([
            dbc.CardHeader("当前模型信息（只读）"),
            dbc.CardBody([html.Div(id="cfg-info", style={"whiteSpace":"pre-wrap","fontSize":"12px"})])
        ], class_name="mb-3"),

        dbc.Card([
            dbc.CardHeader("算法与参数设置"),
            dbc.CardBody([
                # 新增：优化问题选择
                dbc.Row([
                    dbc.Col([
                        html.Div("优化问题 / 测试函数"),
                        dcc.Dropdown(
                            id="cfg-problem",
                            options=[{"label": meta["label"], "value": pid} for pid, meta in PROBLEM_CATALOG.items()],
                            value="ZDT1", clearable=False
                        )
                    ], width=6),
                    dbc.Col([
                        html.Div("变量个数 n_var"),
                        dbc.Input(id="cfg-nvar", type="number", min=2, step=1, value=30)
                    ], width=3),
                ], class_name="mb-3"),

                dbc.Row([
                    dbc.Col([
                        html.Div("算法选择"),
                        dcc.Dropdown(
                            id="cfg-algo",
                            options=[
                                {"label":"NSGA-II (EMO)","value":"NSGA-II"},
                                {"label":"NSGA-III (EMO)","value":"NSGA-III"},
                                {"label":"MOEA/D (EMO)","value":"MOEA/D"},
                                {"label":"BO (EHVI / ASF)","value":"BO"},
                                {"label":"CP/MILP (Linear surrogate)","value":"CP/MILP"},
                                {"label":"RL (CEM + CVaR)","value":"RL"},
                            ],
                            value="NSGA-II",
                            clearable=False
                        ),
                    ], width=4),
                    dbc.Col([html.Div("种群规模 pop_size"), dbc.Input(id="cfg-pop", type="number", min=10, step=10, value=80)], width=4),
                    dbc.Col([html.Div("每 tick 代数 n_gen"), dbc.Input(id="cfg-gen", type="number", min=1, step=1, value=10)], width=4),
                ], class_name="mb-3"),

                # 常驻参数面板（只切换可见性）
                html.Div(build_all_algo_param_panels(), id="cfg-algo-params")
            ])
        ], class_name="mb-3"),

        dbc.Alert(id="cfg-summary", color="secondary", class_name="mb-0", style={"whiteSpace":"pre-wrap","fontSize":"12px"})
    ]),
    dbc.ModalFooter([
        dbc.Button("确认开始", id="btn-start-confirm", color="success", class_name="me-2"),
        dbc.Button("取消", id="btn-start-cancel", color="secondary")
    ])
], id="cfg-modal", is_open=False, size="lg", backdrop="static")


# ===== 左列：对话 + 日志 =====
chatbar = dbc.Col([
    dbc.Card([
        dbc.CardHeader("自然语言 ➜ UPV（对话）"),
        dbc.CardBody([
            dcc.Store(id="nl-chat-store", data=[]),
            html.Div(id="nl-chat-view", className="chat-pane"),
            dbc.InputGroup([
                dbc.Textarea(id="nl-input", placeholder="像 GPT 一样说出你的偏好/约束/目标…（Shift+Enter 换行，点击右侧发送）", rows=3),
                dbc.Button("发送", id="nl-send", color="primary"),
            ], class_name="nl-input-row"),
            html.Small(id="nl-chat-status", className="text-success mt-2 d-block"),
            html.Div(id="nl-scroll-dummy", style={"display":"none"})
        ], className="chat-card-body"),
    ], class_name="mb-3"),

    dbc.Card([
        dbc.CardHeader("交互与 CR 日志"),
        dbc.CardBody(html.Pre(id="log-view", style={"whiteSpace":"pre-wrap","fontSize":"12px", "maxHeight":"260px", "overflowY":"auto"}))
    ])
], width=3)


# ===== 右列：滑块 + 排序 + UPV Snapshot =====
rightbar = dbc.Col([
    dbc.Card([
        dbc.CardHeader("Weights / Trust / Robustness"),
        dbc.CardBody([
            html.Div("Weight w1"),
            dcc.Slider(id="w1", min=0, max=1, step=0.01, value=0.5, className="neo-slider", marks=MARKS_01, tooltip={"placement":"bottom"}),
            html.Div("Weight w2", className="mt-2"),
            dcc.Slider(id="w2", min=0, max=1, step=0.01, value=0.5, className="neo-slider", marks=MARKS_01, tooltip={"placement":"bottom"}),
            html.Div("Trust (τ)", className="mt-2"),
            dcc.Slider(id="trust", min=0, max=1, step=0.01, value=0.2, className="neo-slider", marks=MARKS_01, tooltip={"placement":"bottom"}),
            html.Div("CVaR α", className="mt-2"),
            dcc.Slider(id="cvar", min=0, max=1, step=0.01, value=1.0, className="neo-slider", marks=MARKS_01, tooltip={"placement":"bottom"}),
            html.Small(id="w-display", className="text-muted mt-2 d-block")
        ])
    ], class_name="mb-3"),

    dbc.Card([
        dbc.CardHeader("方案排序（5 个，可拖拽；提交后学习权重）"),
        dbc.CardBody([
            dcc.Store(id="rank-order-store"),
            dcc.Store(id="rank-candidate-set"),
            html.Div(id="rank-warning", className="text-danger mb-2"),
            (EventListener(id="rank-el", events=[{"event":"sortupdate"}],
                           children=html.Ul(id="rank-list", className="p-0 m-0"))
             if HAS_DE else html.Div(id="rank-cards")),
            html.Div(id="rank-preview", className="mt-2"),
            dbc.Button("提交排序", id="rank-submit", color="primary", class_name="mt-2"),
            html.Div(id="pref-msg", className="text-muted mt-2"),
            html.Div(id="rank-init-dummy", style={"position":"relative", "zIndex": 1,"display":"none"})
        ])
    ], class_name="mb-3"),

    dbc.Card([
        dbc.CardHeader("UPV Snapshot"),
        dbc.CardBody(html.Pre(id="upv-snapshot", style={"WhiteSpace":"pre-wrap","fontSize":"12px","color":"#333"}))
    ])
], width=3, className="right-col")


# ===== 中列：图表 + PCP =====
pcp_header = dbc.Row([
    dbc.Col(html.Div("Parallel Coordinates (brush to set D/b)"), width=6),
    dbc.Col(dcc.Dropdown(
        id="pcp-axes",
        options=[{"label":n,"value":n} for n in ["x1","x2","x3","x4","x5","x6","f1","f2"]],
        value=["x1","x2","x3","x4","x5","x6","f1","f2"], multi=True, clearable=False
    ), width=6)
])

center_area = dbc.Col([
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Hypervolume"), dbc.CardBody(dcc.Graph(id="fig-hv", figure=figure_hv()))]), width=6),
        dbc.Col(dbc.Card([dbc.CardHeader("Best Objective Values"), dbc.CardBody(dcc.Graph(id="fig-fmin", figure=figure_fmin()))]), width=6),
    ], class_name="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Non-dominated Front"), dbc.CardBody(dcc.Graph(id="fig-nd", figure=figure_nd_scatter()))]), width=6),
        dbc.Col(dbc.Card([dbc.CardHeader(pcp_header), dbc.CardBody(dcc.Graph(id="pcp"))]), width=6),
    ], class_name="mb-2"),
    dbc.Row([
        dbc.Col(),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.Div("PCP Selection Status", className="fw-semibold mb-1"),
            html.Pre(id="pcp-status", style={"whiteSpace":"pre-wrap","fontSize":"12px","color":"#333","marginBottom":"6px"}),
            dbc.Button("Clear PCP Brushes", id="btn-clear-pcp", color="warning", size="sm"),
        ])]), width=6)
    ], class_name="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Feasibility & Mean Penalty"), dbc.CardBody(dcc.Graph(id="fig-feas", figure=figure_feas_penalty()))]), width=6),
        dbc.Col(dbc.Card([dbc.CardHeader("UPV Slots Timeline"), dbc.CardBody(dcc.Graph(id="fig-upv", figure=figure_upv_timeline()))]), width=6),
    ], class_name="mb-3"),
], width=6)


# ===== 顶层布局 =====
app.layout = html.Div([
    navbar,
    dbc.Container([dbc.Row([chatbar, center_area, rightbar])], fluid=True),
    cfg_modal,
    dcc.Interval(id="tick", interval=1200, n_intervals=0)
])


# ------------------ 回调：滑块、聊天、PCP、刷新 ------------------
@app.callback(
    Output("w-display","children"),
    Input("w1","value"), Input("w2","value"),
    Input("trust","value"), Input("cvar","value"),
    Input("cr-toggle","value"),
    prevent_initial_call=False
)
def update_sliders(w1, w2, trust, cvar, cr_val):
    with state.lock:
        w = np.array([float(w1 or 0.5), float(w2 or 0.5)], dtype=float)
        state.upv.w = normalize_w(w)
        state.upv.tau = float(trust)
        state.upv.rho["cvar_alpha"] = float(cvar)
        state.enable_cr = ("cr" in (cr_val or []))
        return f"w={state.upv.w.tolist()} | τ={state.upv.tau:.2f} | CVaR α={state.upv.rho['cvar_alpha']:.2f} | CR={'ON' if state.enable_cr else 'OFF'}"

@app.callback(
    Output("nl-chat-store","data"),
    Output("nl-chat-status","children"),
    Output("w1","value", allow_duplicate=True),
    Output("w2","value", allow_duplicate=True),
    Output("trust","value", allow_duplicate=True),
    Output("cvar","value", allow_duplicate=True),
    Input("nl-send","n_clicks"),
    State("nl-input","value"),
    State("nl-chat-store","data"),
    prevent_initial_call=True
)
def on_nl_send(n, text, store):
    if not text:
        raise PreventUpdate
    store = store or []
    store.append({"role":"user","text":text})
    try:
        upv_json = apply_nl_and_update_upv(text)
        reply = "好的，已经将您的需求整合到 UPV 中。\n" + json.dumps(
            {k: upv_json.get(k) for k in ["w","r","b","D","tau","rho"]}, ensure_ascii=False, indent=2)
        store.append({"role":"bot","text":reply})
        w1 = float(state.upv.w[0]); w2 = float(state.upv.w[1])
        tau = float(state.upv.tau); cvar = float(state.upv.rho.get("cvar_alpha",1.0))
        return store, "NL→UPV 已应用。", w1, w2, tau, cvar
    except Exception as e:
        store.append({"role":"bot","text":f"[NL解析失败] {e}"})
        return store, f"NL 失败：{e}", no_update, no_update, no_update, no_update

@app.callback(
    Output("nl-chat-view","children"),
    Input("nl-chat-store","data"),
    prevent_initial_call=False
)
def render_chat(store):
    return render_chat_bubbles(store or [])

# 聊天自动滚底
app.clientside_callback(
    """
    function(children){
        var pane = document.getElementById('nl-chat-view');
        if (pane) { pane.scrollTop = pane.scrollHeight; }
        return "";
    }
    """,
    Output("nl-scroll-dummy","children"),
    Input("nl-chat-view","children"),
    prevent_initial_call=False
)

# L5 队列 → 聊天窗口
@app.callback(
    Output("nl-chat-store","data", allow_duplicate=True),
    Input("tick","n_intervals"),
    State("nl-chat-store","data"),
    prevent_initial_call=True
)
def flush_l5_into_chat(_n, store):
    msgs = state.pop_all_l5()
    if not msgs:
        raise PreventUpdate
    with state.lock:
        state.n_clarify_total = int(getattr(state, "n_clarify_total", 0) + len(msgs))
    store = (store or [])[:]
    prefix_map = {"clarify":"[澄清建议] ", "confirm":"[请确认] ", "warn":"[提示] "}
    for m in msgs:
        if isinstance(m, dict):
            text = m.get("text", "")
            kind = m.get("kind", "clarify")
            choices = m.get("choices", [])
            prefix = prefix_map.get(kind, "[提示] ")
            opts = ""
            if choices:
                opts = "\n可选项：" + " | ".join([f"{i+1}) {c}" for i, c in enumerate(choices)])
            store.append({"role": "bot", "text": f"{prefix}{text}{opts}"})
        else:
            store.append({"role": "bot", "text": f"[提示] {str(m)}"})
    return store


# PCP 刷选 & 清空
@app.callback(
    Output("pcp-status", "children"),
    Input("pcp", "relayoutData"),
    Input("btn-clear-pcp", "n_clicks"),
    prevent_initial_call=True
)
def pcp_update(relayout, n_clear):
    trig = ctx.triggered_id
    if trig == "btn-clear-pcp":
        with state.lock:
            state.pcp_ranges.clear()
            oldD = list(state.upv.D)
            oldb = list(state.upv.b)
            state.upv.D = [tuple(x) for x in state.problem.bounds]
            state.upv.b = [(-np.inf, np.inf) for _ in range(state.n_obj)]
            state.add_log("PCP-CLEAR", f"D: {oldD} -> {state.upv.D}; b: {oldb} -> {state.upv.b}")
            if state.enable_cr:
                l2_reconcile_D_vs_Ch(state)
                l4_goal_consistency(state)
        return "Cleared."
    if trig == "pcp" and relayout:
        parsed: Dict[int, Tuple[float, float]] = {}
        def _fold_range(r):
            if isinstance(r, list) and len(r) > 0:
                if isinstance(r[0], list):
                    lo = min(rr[0] for rr in r); hi = max(rr[1] for rr in r)
                    return float(min(lo, hi)), float(max(lo, hi))
                elif len(r) == 2:
                    lo, hi = float(r[0]), float(r[1])
                    return (lo, hi) if lo <= hi else (hi, lo)
            return None
        for k, v in (relayout or {}).items():
            m = re.match(r"dimensions\[(\d+)\]\.constraintrange(?:\[\d+\])?", k)
            if m:
                idx = int(m.group(1))
                rng = _fold_range(v)
                if rng is not None:
                    parsed[idx] = rng
        if not parsed:
            return no_update
        with state.lock:
            state.pcp_ranges.update(parsed)
            state.n_atoms_visual_total = int(getattr(state, "n_atoms_visual_total", 0) + len(parsed))
            
            D = list(state.upv.D) if state.upv.D else [tuple(x) for x in state.problem.bounds]
            changes = []
            for i, (lo, hi) in parsed.items():
                if 0 <= i <= 5 and i < state.n_var:
                    lo_ = max(float(state.problem.bounds[i,0]), float(lo))
                    hi_ = min(float(state.problem.bounds[i,1]), float(hi))
                    changes.append(f"x{i+1} [{D[i][0]:.3f},{D[i][1]:.3f}] -> [{lo_:.3f},{hi_:.3f}]")
                    D[i] = (lo_, hi_)
                elif i == 6:
                    changes.append(f"f1 band {state.upv.b[0]} -> {(float(lo), float(hi))}")
                    state.upv.b[0] = (float(lo), float(hi))
                elif i == 7:
                    changes.append(f"f2 band {state.upv.b[1]} -> {(float(lo), float(hi))}")
                    state.upv.b[1] = (float(lo), float(hi))
            state.upv.D = D
            state.add_log("PCP", " | ".join(changes))
            if state.enable_cr:
                l2_reconcile_D_vs_Ch(state)
                l4_goal_consistency(state)
            lines = []
            for i in range(min(6, state.n_var)):
                if i in state.pcp_ranges:
                    lines.append(f"x{i+1} in [{state.upv.D[i][0]:.3f}, {state.upv.D[i][1]:.3f}]")
            if 6 in state.pcp_ranges:
                lines.append(f"f1 in [{state.upv.b[0][0]:.3f}, {state.upv.b[0][1]:.3f}]")
            if 7 in state.pcp_ranges:
                lines.append(f"f2 in [{state.upv.b[1][0]:.3f}, {state.upv.b[1][1]:.3f}]")
        return " | ".join(lines) if lines else "No active brush."
    return no_update


# 统一刷新
@app.callback(
    Output("pcp","figure"),
    Output("fig-hv","figure"),
    Output("fig-fmin","figure"),
    Output("fig-nd","figure"),
    Output("fig-feas","figure"),
    Output("fig-upv","figure"),
    Output("upv-snapshot","children"),
    Output("log-view","children"),
    Output("conv-hint","children"),
    Output("conv-hint","style"),
    Input("tick","n_intervals"),
    Input("pcp-axes","value"),
    prevent_initial_call=False
)
def refresh_all(_n, axes):
    with state.lock:
        upv_snap = {
            "problem": state.problem_id,
            "w": state.upv.w.tolist(),
            "r": state.upv.r.tolist(),
            "b": state.upv.b,
            "tau": state.upv.tau,
            "cvar": state.upv.rho.get("cvar_alpha", 1.0),
            "D0..5": state.upv.D[:6] if state.upv.D else []
        }
        return (
            build_pcp_fig(selected_axes=axes),
            figure_hv(),
            figure_fmin(),
            figure_nd_scatter(),
            figure_feas_penalty(),
            figure_upv_timeline(),
            json.dumps(upv_snap, ensure_ascii=False, indent=2),
            "\n".join(state.history_logs[-200:]) if state.history_logs else "No logs yet.",
            ("Converged ✓" if state.converged else ""),
            ({"display":"inline-block"} if state.converged else {"display":"none"})
        )


# ===== 弹窗开关 & 参数展示 =====
@app.callback(
    # ----- Outputs（去掉重复的 cfg-moead-nei）-----
    Output("cfg-modal","is_open"),
    Output("cfg-info","children"),
    Output("cfg-summary","children"),
    Output("cfg-algo","value"),
    Output("cfg-pop","value"),
    Output("cfg-gen","value"),
    Output("cfg-cxprob","value"),
    Output("cfg-cxeta","value"),
    Output("cfg-muteta","value"),
    Output("cfg-mutprob","value"),
    Output("cfg-moead-nei","value"),
    Output("cfg-problem","value"),
    Output("cfg-nvar","value"),
    Output("cfg-bo-nlocal","value"),
    Output("cfg-bo-nglobal","value"),
    Output("cfg-bo-l","value"),
    Output("cfg-bo-beta","value"),
    Output("cfg-bo-radius","value"),
    # ----- Inputs & State -----
    Input("btn-start","n_clicks"),
    Input("btn-start-cancel","n_clicks"),
    State("cfg-modal","is_open"),
    prevent_initial_call=True
)
def toggle_cfg_modal(n_open, n_cancel, is_open):
    trig = ctx.triggered_id
    if trig == "btn-start":
        with state.lock:
            var_lines = [
                f"x{i+1}: [{float(state.problem.bounds[i,0]):.3f}, {float(state.problem.bounds[i,1]):.3f}]"
                for i in range(min(10, state.n_var))
            ]
            info = (
                f"问题: {state.problem_id}\n"
                f"目标个数: {state.n_obj}\n"
                f"变量个数: {state.n_var}\n"
                f"CR: {'ON' if state.enable_cr else 'OFF'} ; τ={state.upv.tau:.2f} ; CVaR α={state.upv.rho.get('cvar_alpha',1.0):.2f}\n"
                f"当前 w={state.upv.w.tolist()}  r={state.upv.r.tolist()}\n"
                f"变量范围（前 10 项预览）:\n" + "\n".join(var_lines)
            )
            summary = (
                f"算法: {state.algo_name}\n"
                f"pop_size={state.pop_size}, n_gen/tick={state.gen_per_tick}, seed={state.seed}\n"
                f"SBX: prob={state.cx_prob}, eta={state.cx_eta} | PM: prob={state.mut_prob or '1/n'}, eta={state.mut_eta}"
            )
            return (
                True,                      # cfg-modal.is_open
                info,                      # cfg-info.children
                summary,                   # cfg-summary.children
                state.algo_name,           # cfg-algo.value
                state.pop_size,            # cfg-pop.value
                state.gen_per_tick,        # cfg-gen.value
                state.cx_prob,             # cfg-cxprob.value
                state.cx_eta,              # cfg-cxeta.value
                state.mut_eta,             # cfg-muteta.value
                state.mut_prob,            # cfg-mutprob.value
                state.emo_params.get("moead_neighbors", 15),             # cfg-moead-nei.value
                state.problem_id,          # cfg-problem.value
                state.n_var,               # cfg-nvar.value
                float(state.bo_params.get("n_local_factor", 6.0)),       # cfg-bo-nlocal.value
                int(state.bo_params.get("n_global", 40)),                # cfg-bo-nglobal.value
                float(state.bo_params.get("kernel_l", 0.3)),             # cfg-bo-l.value
                ("" if state.bo_params.get("beta_override", None) is None
                   else float(state.bo_params["beta_override"])),         # cfg-bo-beta.value
                float(state.bo_params.get("radius_scale", 1.0)),         # cfg-bo-radius.value
            )
    # 取消或其它情况：保持关闭 & 其它不更新（注意返回数量=18）
    return (
        False, no_update, no_update, no_update, no_update, no_update,
        no_update, no_update, no_update, no_update, no_update,
        no_update, no_update, no_update, no_update, no_update, no_update, no_update
    )

# 面板可见性切换 + 摘要更新
@app.callback(
    Output("panel-emo","style"),
    Output("panel-bo","style"),
    Output("panel-cpm","style"),
    Output("panel-rl","style"),
    Output("cfg-summary","children", allow_duplicate=True),
    Input("cfg-algo","value"),
    State("cfg-pop","value"),
    State("cfg-gen","value"),
    prevent_initial_call='initial_duplicate'
)
def on_algo_change_toggle_panels(algo, pop, ngen):
    show = {}; hide = {"display": "none"}
    if algo in ("NSGA-II","NSGA-III","MOEA/D"):
        styles = (show, hide, hide, hide)
        p = state.emo_params
        extra = f"SBX: prob={p['cx_prob']}, eta={p['cx_eta']} | PM: prob={p['mut_prob'] or '1/n'}, eta={p['mut_eta']}"
        if algo == "MOEA/D":
            extra += f" | neighbors={p['moead_neighbors']}"
    elif algo == "BO":
        styles = (hide, show, hide, hide)
        p = state.bo_params
        extra = f"n_local≈{p['n_local_factor']}×pop, n_global≥{p['n_global']}, l={p['kernel_l']}, β={p['beta_override'] or '(auto by α)'} , r_scale={p['radius_scale']}"
    elif algo == "CP/MILP":
        styles = (hide, hide, show, hide)
        p = state.cpm_params
        extra = f"pulp={'ON' if (p['use_pulp'] and state.has_pulp) else 'OFF'}, sample={p['sample_size']}, λ={p['lambda_pen']}, ridge_L2={p['ridge_l2']}"
    else:
        styles = (hide, hide, hide, show)
        p = state.rl_params
        extra = f"N_mult={p['N_mult']}, elite={p['elite_frac']}, r_scale={p['radius_scale']}, σ={p['sigma']}"
    summary = (f"算法: {algo}\n"
               f"pop_size={pop}, n_gen/tick={ngen}\n"
               f"{extra}")
    return (*styles, summary)


# ===== 应用参数 & 开始优化 =====
opt_thread = None
def ensure_thread_running():
    global opt_thread
    if opt_thread is None or not opt_thread.is_alive():
        state.stop_event.clear()
        opt_thread = threading.Thread(target=optimizer_loop, args=(state,), daemon=True)
        opt_thread.start()
# @app.callback(
#     Output("panel-emo", "style"),
#     Output("panel-bo", "style"),
#     Output("panel-cpm", "style"),
#     Output("panel-rl", "style"),
#     Input("cfg-algo", "value"),
#     prevent_initial_call=False
# )
# def _toggle_algo_panels(algo):
#     show = {"display": "block"}
#     hide = {"display": "none"}
#     if algo in ("NSGA-II", "NSGA-III", "MOEA/D"):
#         return show, hide, hide, hide
#     if algo == "BO":
#         return hide, show, hide, hide
#     if algo == "CP/MILP":
#         return hide, hide, show, hide
#     if algo == "RL":
#         return hide, hide, hide, show
#     return show, hide, hide, hide  # 默认回退到 EMO

@app.callback(
    Output("btn-start", "children"),
    Output("cfg-modal", "is_open", allow_duplicate=True),
    Output("btn-pause", "children", allow_duplicate=True),
    Input("btn-start-confirm", "n_clicks"),
    # —— 通用 ——
    State("cfg-algo", "value"),
    State("cfg-pop", "value"),
    State("cfg-gen", "value"),
    State("cfg-problem", "value"),
    State("cfg-nvar", "value"),
    # —— EMO ——
    State("cfg-cxprob", "value"),
    State("cfg-cxeta", "value"),
    State("cfg-muteta", "value"),
    State("cfg-mutprob", "value"),
    State("cfg-moead-nei", "value"),
    # —— BO（与你布局一致） ——
    State("cfg-bo-nlocal", "value"),
    State("cfg-bo-nglobal", "value"),
    State("cfg-bo-l", "value"),
    State("cfg-bo-beta", "value"),
    State("cfg-bo-radius", "value"),
    # —— CPM（布局里只有这个） ——
    State("cfg-cpm-maxnodes", "value"),
    # —— RL（布局里只有这个） ——
    State("cfg-rl-sigma", "value"),
    # —— 其它 ——
    State("cfg-expt-mode", "value"),
    State("cfg-log-enable", "value"),
    State("cfg-run-id", "value"),
    prevent_initial_call=True
)
def on_start_confirm(n, algo, pop, ngen, prob_id, nvar,
                     cxp, cxeta, mueta, muprob, moead_nei,
                     bo_nlocal, bo_nglobal, bo_l, bo_beta, bo_radius,
                     cpm_maxnodes,
                     rl_sigma,
                     expt_mode, log_enable, run_id):
    if not n:
        raise PreventUpdate

    with state.lock:
        # 切换问题
        state.set_problem(prob_id or "ZDT1", int(nvar or state.n_var))

        # —— 通用配置 ——
        state.algo_name = algo or "NSGA-II"
        state.pop_size = int(pop or state.pop_size)
        state.gen_per_tick = int(ngen or state.gen_per_tick)

        # —— EMO ——
        state.cx_prob = float(cxp if cxp is not None else state.cx_prob)
        state.cx_eta = float(cxeta if cxeta is not None else state.cx_eta)
        state.mut_eta = float(mueta if mueta is not None else state.mut_eta)
        state.mut_prob = (float(muprob) if (muprob is not None and str(muprob) != "") else None)
        state.emo_params.update({
            "cx_prob": state.cx_prob,
            "cx_eta": state.cx_eta,
            "mut_eta": state.mut_eta,
            "mut_prob": state.mut_prob,
            "moead_neighbors": int(moead_nei) if moead_nei is not None else state.emo_params.get("moead_neighbors", 15),
        })

        # —— BO ——
        state.bo_params.update({
            "n_local_factor": float(bo_nlocal) if bo_nlocal is not None else state.bo_params.get("n_local_factor", 6.0),
            "n_global": int(bo_nglobal) if bo_nglobal is not None else state.bo_params.get("n_global", 40),
            "kernel_l": float(bo_l) if bo_l is not None else state.bo_params.get("kernel_l", 0.3),
            "beta_override": (None if (bo_beta is None or str(bo_beta).strip() == "") else float(bo_beta)),
            "radius_scale": float(bo_radius) if bo_radius is not None else state.bo_params.get("radius_scale", 1.0),
        })

        # —— CP/MILP ——（当前 UI 仅有 max_nodes，可先记下；真正用不用看你后端逻辑）
        state.cpm_params["max_nodes"] = int(cpm_maxnodes) if cpm_maxnodes is not None else state.cpm_params.get("max_nodes", 2000)

        # —— RL ——（UI 只提供 sigma，其它沿用 state 现值）
        if rl_sigma is not None:
            state.rl_params["sigma"] = float(rl_sigma)

        # —— 实验模式 & 日志 ——
        state.expt_mode = (expt_mode or "upv_full")
        state.enable_cr = True if state.expt_mode == "upv_full" else False

        # 计数器复位
        state.tick_count = 0
        state.eval_count = 0
        state.n_atoms_nl_total = 0
        state.n_atoms_visual_total = 0
        state.n_pairs_rank_total = 0
        state.n_clarify_total = 0
        state.converged = False
        state.conv_hint = ""

        # Single：清空 D/b/r（使排序权重主导）
        if state.expt_mode == "single":
            state.upv.r = np.zeros(state.n_obj)
            state.upv.b = [(-np.inf, np.inf) for _ in range(state.n_obj)]
            state.upv.D = [tuple(x) for x in state.problem.bounds]

        # 日志初始化
        state.logging_on = bool(log_enable) and ("on" in (log_enable or []))
        state.run_id = (run_id or "").strip() or None
        if state.logging_on:
            try:
                state.logger = ExperimentLogger(base_dir="logs")
                cfg = {
                    "mode": state.expt_mode,
                    "problem": state.problem_id,
                    "algo": state.algo_name,
                    "pop_size": state.pop_size,
                    "gen_per_tick": state.gen_per_tick,
                    "seed": state.seed,
                }
                state.logger.start_run(state.run_id, cfg)
                state.run_id = state.logger.run_id
                state.add_log("LOG", f"logging -> logs/{state.run_id}/metrics.csv")
            except Exception as e:
                state.add_log("LOG", f"logger init failed: {e}")
                state.logging_on = False
                state.logger = None

        # 重新初始化并启动
        state.init_population()
        ensure_thread_running()
        state.running = True
        algo_msg = f"EMO({state.algo_name})" if state.algo_name in ("NSGA-II","NSGA-III","MOEA/D") else state.algo_name
        state.add_log("RUN",
            f"Start problem={state.problem_id} (n_var={state.n_var}); algo={algo_msg}, pop={state.pop_size}, n_gen/tick={state.gen_per_tick}"
        )

    return "运行中...", False, "暂停"



# 暂停/继续
@app.callback(
    Output("btn-pause", "children"),
    Output("btn-start", "children", allow_duplicate=True),
    Input("btn-pause", "n_clicks"),
    prevent_initial_call=True
)
def toggle_pause(n):
    if not n:
        raise PreventUpdate
    with state.lock:
        state.running = not state.running
        state.add_log("RUN", "Resume" if state.running else "Pause")
        if state.running:
            pause_label = "暂停"; start_label = "运行中..."
        else:
            pause_label = "继续优化"; start_label = "已暂停"
    return pause_label, start_label

@app.callback(
    Output("btn-stop","children"),
    Input("btn-stop","n_clicks"),
    prevent_initial_call=True
)

def on_stop(n):
    with state.lock:
        state.running = False
        state.stop_event.set()
        state.add_log("RUN", "Stop")
        # 日志快照
        if getattr(state, "logging_on", False) and state.logger is not None and state.pop_X is not None and state.pop_F is not None:
            try:
                state.logger.dump_population(state.pop_X, state.pop_F, tick=state.tick_count)
                upv_snap = {
                    "problem": state.problem_id,
                    "w": state.upv.w.tolist(),
                    "r": state.upv.r.tolist(),
                    "b": state.upv.b,
                    "tau": state.upv.tau,
                    "cvar": state.upv.rho.get("cvar_alpha", 1.0),
                    "D": state.upv.D
                }
                state.logger.dump_upv(upv_snap, "upv_snapshot.json")
                state.logger.finalize()
            except Exception as e:
                state.add_log("LOG", f"finalize failed: {e}")
    return "已停止"


@app.callback(
    Output("btn-step","children"),
    Input("btn-step","n_clicks"),
    prevent_initial_call=True
)
def on_step(n):
    with state.lock:
        state.step_opt()
        state.add_log("RUN", f"Step {state.gen_per_tick} gen")
    return f"已步进 {state.gen_per_tick} 代"


# ===== 方案排序：刷新 =====
def get_rank_candidates_k5() -> List[int]:
    if state.pop_F is None:
        return []
    return select_representative_indices(state.pop_F, k=5)

if HAS_DE:
    @app.callback(
        Output("rank-list","children"),
        Output("rank-order-store","data"),
        Output("rank-candidate-set","data"),
        Output("rank-warning","children"),
        Input("tick","n_intervals"),
        State("rank-order-store","data"),
        State("rank-candidate-set","data"),
        prevent_initial_call=False
    )
    def refresh_rank_list(_n, order_store, candidate_set):
        with state.lock:
            new_cands = get_rank_candidates_k5()
            warn = "" if len(new_cands) == 5 else "当前非支配解不足 5 个，已用偏好最近邻补足。"
            old_set = set(map(int, candidate_set or []))
            new_set = set(map(int, new_cands))
            if (not old_set) or (new_set != old_set) or (not order_store):
                base = [int(i) for i in (order_store or []) if int(i) in new_set]
                for i in new_cands:
                    if int(i) not in base:
                        base.append(int(i))
                items = render_rank_list_items(base)
                return items, base, list(map(int, new_cands)), warn
            return no_update, no_update, no_update, warn
else:
    @app.callback(
        Output("rank-cards", "children"),
        Output("rank-order-store", "data"),
        Output("rank-candidate-set", "data"),
        Output("rank-warning", "children"),
        Input("tick", "n_intervals"),
        State("rank-order-store", "data"),
        State("rank-candidate-set", "data"),
        prevent_initial_call=False
    )
    def refresh_rank_cards(_n, order_store, candidate_set):
        with state.lock:
            # 获取候选（容错：为空则用空列表）
            new_cands_raw = get_rank_candidates_k5() or []
            # 统一转成 Python int 且去重（保持顺序）
            new_cands = list(dict.fromkeys(int(i) for i in new_cands_raw))

            warn = "" if len(new_cands) == 5 else "当前非支配解不足 5 个，已用偏好最近邻补足。"

            # 用于判断是否需要重绘
            old_set = set(int(i) for i in (candidate_set or []))
            new_set = set(new_cands)

            if (not old_set) or (new_set != old_set) or (not order_store):
                # 保留旧的排序顺序中的仍有效项，然后追加新出现的
                base = [int(i) for i in (order_store or []) if int(i) in new_set]
                for i in new_cands:
                    if i not in base:
                        base.append(i)

                # 渲染卡片（容错：若返回 tuple/单组件/None，全都规整成 list）
                cards = _render_cards_fallback(base)
                if isinstance(cards, tuple):
                    cards = cards[0]
                if cards is None:
                    cards = []
                elif not isinstance(cards, (list, tuple)):
                    cards = [cards]

                return list(cards), base, new_cands, warn

            # 候选无变化且已有顺序缓存 -> 不更新 UI
            return no_update, no_update, no_update, warn


if HAS_DE:
    @app.callback(
        Output("rank-preview","children"),
        Output("rank-order-store","data", allow_duplicate=True),
        Input("rank-el","event"),
        State("rank-order-store","data"),
        prevent_initial_call=True
    )
    def on_sortupdate(evt, old_order):
        if not evt or "order" not in evt:
            raise PreventUpdate
        new_order = [int(i) for i in evt["order"]]
        if not new_order:
            raise PreventUpdate
        return "当前顺序： " + "  →  ".join([f"#{i}" for i in new_order]), new_order
else:
    @app.callback(
        Output("rank-order-store","data", allow_duplicate=True),
        Input({"t":"rank-up","i":dash.ALL}, "n_clicks"),
        Input({"t":"rank-down","i":dash.ALL}, "n_clicks"),
        State("rank-order-store","data"),
        prevent_initial_call=True
    )
    def move_card(up_clicks, down_clicks, order):
        trig = ctx.triggered_id
        if not trig or not order:
            raise PreventUpdate
        idx = int(trig["i"]); t = trig["t"]
        order = [int(i) for i in order]
        if idx not in order: raise PreventUpdate
        pos = order.index(idx)
        if t == "rank-up" and pos>0:
            order[pos-1], order[pos] = order[pos], order[pos-1]
        if t == "rank-down" and pos < len(order)-1:
            order[pos+1], order[pos] = order[pos], order[pos+1]
        return [int(i) for i in order]

    @app.callback(
        Output("rank-preview","children"),
        Input("rank-order-store","data"),
        prevent_initial_call=False
    )
    def preview_rank(order):
        if not order: return "无排序。"
        return "当前顺序： " + "  →  ".join([f"#{int(i)}" for i in order])

# 初始化前端 Sortable
app.clientside_callback(
    """
    function(children){
        if (window && window.__rankDnDInit){ window.__rankDnDInit(); }
        return "";
    }
    """,
    Output("rank-init-dummy","children"),
    Input("rank-list","children"),
    prevent_initial_call=False
)

# 提交排序 → 学习 w
@app.callback(
    Output("pref-msg","children"),
    Output("w1","value", allow_duplicate=True),
    Output("w2","value", allow_duplicate=True),
    Input("rank-submit","n_clicks"),
    State("rank-order-store","data"),
    prevent_initial_call=True
)
def submit_rank(n, order):
    if not order or len(order) < 2:
        return "请先拖拽/调整顺序后再提交。", no_update, no_update
    with state.lock:
        order = [int(i) for i in order]
        pairs = []
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                ia, ib = order[i], order[j]
                fA, fB = state.pop_F[ia], state.pop_F[ib]
                pairs.append((fA, fB, 0))
        state.n_pairs_rank_total = int(getattr(state, "n_pairs_rank_total", 0) + len(pairs))
        w_hat = fit_bt_weights(pairs, state.upv.w.copy())
        eta = 0.35
        w_old = state.upv.w.copy()
        state.upv.w = normalize_w((1 - eta) * state.upv.w + eta * w_hat)
        state.add_log("RANK-LIST", f"order={order}, w: {w_old.tolist()} -> {state.upv.w.tolist()} (eta={eta})")
        msg = f"已学习权重：w={state.upv.w.tolist()}（η={eta}）；排序已纳入偏好模型。"
        return msg, float(state.upv.w[0]), float(state.upv.w[1])


# 同步 n_var 默认（可选）
@app.callback(
    Output("cfg-nvar","value", allow_duplicate=True),
    Input("cfg-problem","value"),
    prevent_initial_call=True
)
def sync_nvar_default(pid):
    meta = PROBLEM_CATALOG.get((pid or "").upper())
    if not meta: raise PreventUpdate
    return int(meta["default_nvar"])


# 后台线程首次拉起
def ensure_thread_once():
    global opt_thread
    if 'opt_thread' not in globals() or opt_thread is None or not opt_thread.is_alive():
        state.stop_event.clear()
        globals()['opt_thread'] = threading.Thread(target=optimizer_loop, args=(state,), daemon=True)
        globals()['opt_thread'].start()
# ------------------ 启动 ------------------

state = OptState(problem_id="ZDT1", n_var=30)
state.enable_cr = True
state.init_population()
ensure_thread_once()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=False)

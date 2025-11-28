# -*- coding: utf-8 -*-
"""
nl2upv_demo.py

Standalone demo: Natural Language → Unified Preference Vector (UPV)
- Calls OpenAI Chat Completions if OPENAI_API_KEY is set (or if API_KEY_OVERRIDE below is set),
  otherwise falls back to a robust regex parser.
- Returns strict JSON with normalized weights, 1-based→0-based index conversion,
  and basic validation/cleanup.

Author: (your name)
License: MIT
"""

import os
import re
import json
import argparse
import time
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

# ------------------------------
# (可选) 直接把Key写在这里，或留空用环境变量
# ------------------------------

# ==== HARD-CODED DEFAULTS (必须放在文件最前面) ====
API_KEY_OVERRIDE = "TfzRMdEiJqPDwXQKeoji:dCwCNTjKSbgHWOmWvEly"   # Bearer Key
DEFAULT_BASE_URL = "https://spark-api-open.xf-yun.com/v1"       # SDK会自己拼 /chat/completions，我们这里也手动拼
DEFAULT_MODEL = "generalv3.5"                                 # 例如 generalv3.5 / 4.0Ultra / Pro / Max / Lite
# =================================================

DEBUG_LLM = True
from typing import List, Dict, Any

class ConstraintAtom:
    """
    约束条件对象，包含硬约束与软约束信息。
    """
    def __init__(self, kind: str, idx: int, sign: str, value: float, hard: bool = True, penalty: float = 10.0):
        self.kind = kind  # 'obj' 或 'var'
        self.idx = idx  # 目标索引
        self.sign = sign  # 约束符号： '<=', '>=', '=='
        self.value = value  # 约束值
        self.hard = hard  # 是否为硬约束
        self.penalty = penalty  # 软约束的惩罚因子

    def __repr__(self):
        return f"ConstraintAtom(kind={self.kind}, idx={self.idx}, sign={self.sign}, value={self.value}, hard={self.hard}, penalty={self.penalty})"

# 其他相关的函数和类，如 llm_nl_to_patch 等
def llm_nl_to_patch(user_text: str) -> Dict[str, Any]:
    """
    解析自然语言为结构化补丁。
    """
    # 使用 LLM 或其他方法将自然语言转换为补丁格式
    patch = {
        "ops": [
            {
                "op": "set_weight",
                "index": 0,
                "value": 0.5
            },
            {
                "op": "set_reference",
                "values": [0.0, 1.0]
            },
            {
                "op": "add_constraint",
                "kind": "obj",
                "index": 1,
                "sign": "<=",
                "value": 0.8,
                "hard": True,
                "penalty": 10.0
            }
        ]
    }
    return patch

# ------------------------------
# UPV 数据结构（最小可用子集）
# ------------------------------
class ConstraintAtom(BaseModel):
    kind: str                    # 'var' | 'obj'，kind='var' 表示变量约束、'obj' 表示目标约束。
    idx: int                     # 0-based
    sign: str                    # '<=' | '>=' | '=='
    value: float
    hard: bool = True
    penalty: float = 10.0        # used if soft

    @field_validator("kind")
    @classmethod
    def _kind_ok(cls, v):
        if v not in ("var", "obj"):
            raise ValueError("kind must be 'var' or 'obj'")
        return v

    @field_validator("sign")
    @classmethod
    def _sign_ok(cls, v):
        if v not in ("<=", ">=", "=="):
            raise ValueError("sign must be one of <=, >=, ==")
        return v


class Robustness(BaseModel):
    cvar_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    noise_std: float = Field(default=0.0, ge=0.0)


class Strategy(BaseModel):
    trust: float = Field(default=0.2, ge=0.0, le=1.0)
    explore_bias: float = 0.0


class UPV:
    w: np.ndarray
    r: np.ndarray
    b: List[Tuple[float, float]]
    C_h: List[ConstraintAtom] = Field(default_factory=list)
    C_s: List[ConstraintAtom] = Field(default_factory=list)
    D: List[Tuple[float, float]] = Field(default_factory=list)
    tau: float = 0.2
    rho: Dict[str, Any] = Field(default_factory=lambda: {"cvar_alpha": 1.0, "noise_std": 0.0})
    pi: Dict[str, Any] = Field(default_factory=lambda: {"explore_bias": 0.0})
    kappa: Dict[str, float] = Field(default_factory=lambda: {"hard": 1e-6, "var": 0.2, "robust": 0.2})
    sigma: Dict[str, float] = Field(default_factory=lambda: {"w": 0.5, "r": 0.5, "b": 0.5, "D": 0.5})
    provenance: List[str] = Field(default_factory=list)
    def log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.provenance.append(f"[{stamp}] {msg}")

def call_xf_api(user_text):
    pass


def parse_nl_to_upv(user_text: str) -> UPV:
    data = call_xf_api(user_text)  # 使用科大讯飞API解析自然语言
    w = [max(0.0, float(x)) for x in data["weights"]]  # 获取目标权重
    r = [float(x) for x in data["reference"]]  # 获取目标参考点
    b = [(float(lo), float(hi)) for lo, hi in data["bands"]]  # 获取目标区间
    upv = UPV(w=w, r=r, b=b)
    return upv

def normalize_weights(w: List[float]) -> List[float]:
    arr = np.array([max(0.0, float(x)) for x in w], dtype=float)
    s = arr.sum()
    if s <= 0:
        # fallback: uniform
        arr = np.ones_like(arr) / len(arr)
    else:
        arr = arr / s
    return arr.tolist()


# ------------------------------
# LLM 解析器 + 正则回退
# ------------------------------
def build_llm_prompt(user_text: str) -> str:
    schema = {
        "weights": "array len M, non-negative, sum→1 (omit if absent)",
        "reference": "array len M (omit if absent)",
        "bands": "list of {obj_index: int(1-based), lo: float, hi: float}",
        "constraints": "list of {kind: 'var'|'obj', idx: int(1-based), sign: '<='|'>='|'==', value: float, hard: bool, penalty: float(optional)}",
        "robustness": "{cvar_alpha: float in (0,1], noise_std: float(optional)}",
        "strategy": "{trust: float in (0,1], explore_bias: float(optional)}"
    }
    prompt = f"""
You are an expert optimization assistant. Convert the following Chinese natural language into a STRICT JSON object for a multi-objective optimizer. 
Return ONLY a JSON object, with no additional text.

Fields (omit missing ones):
{json.dumps(schema, indent=2, ensure_ascii=False)}

Rules:
- Convert all indices from 1-based to 0-based in the JSON.
- If weights provided but don't sum to 1, normalize them (keep non-negative).
- If text uses '必须/要求/必须满足/shall' → hard=true; '希望/尽量/最好/prefer' → hard=false and provide a reasonable penalty (default 10).
- If trust/cvar specified, map to strategy.trust / robustness.cvar_alpha respectively.

User text:
\"\"\"{user_text}\"\"\"
"""
    return prompt.strip()


import json, urllib.request, urllib.error

def call_openai_chat(user_text: str, model: str | None, api_key: str | None, base_url: str | None = None) -> dict | None:
    """
    仅使用标准库HTTP调用 OpenAI兼容接口（科大讯飞 Spark 可用）。
    - base_url: 设为 https://spark-api-open.xf-yun.com/v1
    - POST /chat/completions
    返回严格 JSON（失败返回 None，交由 regex 回退）。
    """
    api_key = api_key or API_KEY_OVERRIDE
    base_url = base_url or DEFAULT_BASE_URL
    model = model or DEFAULT_MODEL
    if not api_key:
        print("[LLM] api_key is empty"); return None

    # 规范 /v1
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    url = base_url.rstrip("/") + "/chat/completions"

    # ---- 提示词（保持你原有 schema/规则）----
    def build_llm_prompt(user_text: str) -> str:
        schema = {
            "weights": "array len M, non-negative, sum→1 (omit if absent)",
            "reference": "array len M (omit if absent)",
            "bands": "list of {obj_index: int(1-based), lo: float, hi: float}",
            "constraints": "list of {kind: 'var'|'obj', idx: int(1-based), sign: '<='|'>='|'==', value: float, hard: bool, penalty: float(optional)}",
            "robustness": "{cvar_alpha: float in (0,1], noise_std: float(optional)}",
            "strategy": "{trust: float in (0,1], explore_bias: float(optional)}"
        }

        prompt = f"""
        你是一名多目标优化助手。请把下面的**中文自然语言**转换为一个**严格 JSON 对象**，用于多目标优化求解器。
        **只输出 JSON**，不要任何说明文字、不要使用 Markdown 代码块。

        【字段（缺省可省略；只产你确定的字段）】
        {json.dumps(schema, indent=2, ensure_ascii=False)}

        【语义到字段的映射（对应左侧 UI 模块）】
        1) 目标权重（“xx 目标更重要”）→ weights: [w0, w1, ...]，非负并归一化。
        2) 参考点（“xx 在 xx 时较好”）→ reference: [r0, r1, ...]。
        3) 硬/软约束（“必须/要求/应当/shall”=硬；“希望/尽量/最好/prefer”=软）→
           constraints: [
             {{"kind":"obj"|"var", "idx":<0-based>, "sign":">="|"<="|"==", "value":<number>, "hard":<bool>, "penalty":<number 可选>}}
           ]
           - 软约束未给 penalty 时默认 10。
        4) 稳健度（“方案更稳一点/保守/风险小/分位/CVaR”）→ robustness.cvar_alpha ∈ (0,1]。
        5) 搜索策略（“先广后精/先粗搜再细调/探索/开发/信任度”）→ strategy.trust ∈ [0,1]。
        6) 几何屏蔽（“不要动前舵/横梁/某变量不用”）→ 可转为对变量/目标的带宽或等式约束：
           - 目标带宽 bands: [[lo, hi], ...]；允许 "-inf"/"inf" 字符串表示无界。
           - 或 constraints 中的等式/不等式限制（kind="var"）。
        7) 解释请求（“为什么选 xx 方案”）→ 不在此 JSON 输出；此处忽略。

        【硬规则】
        - 索引统一**1→0**（“w1/目标2/x3”→ index=0/1/2）。
        - 只在指令**明确**时填写字段；否则**不要猜**，保持缺省（不要编造数值）。
        - 若提供了 weights 但不和为 1：保留相对比例，按非负归一化。
        - 若出现负权重，先截到 0，然后再归一化（全 0 时设为均匀分布）。
        - 数值一律用十进制浮点数；不要 NaN；无穷用字符串 "inf"/"-inf"。
        - 约束语义：
          - “至少/不小于/≥/不得低于”→ sign=">="；“至多/不大于/≤/不超过”→ sign="<="；“等于/固定为/严格在点上/==”→ sign="=="。
          - “必须/要求/必须满足/应当/shall”→ hard=true；“希望/尽量/最好/偏好/prefer”→ hard=false（penalty 缺省为 10）。
        - 稳健度/风险：
          - “更稳/更保守/抗波动/风险小”→ 较小的 cvar_alpha（例如 0.1–0.3）。
          - “愿意冒险/激进/风险高”→ 较大的 cvar_alpha（例如 0.7–1.0）。
          若用户给了具体数值则以用户为准。
        - 搜索策略：
          - “先广搜/粗搜/探索多一些”→ strategy.trust 偏低（如 0.2–0.4）。
          - “再细调/精调/收敛/开发多一些”→ strategy.trust 偏高（如 0.6–0.8）。
          若用户给了具体数值则以用户为准。
        - 如果用户给了相互矛盾的指令（例如同一目标同时“≥0.8 且 ≤0.5”）：
          - 优先保留包含“必须/应当/shall”的硬约束；
          - 软约束可适当放宽或省略（并不要输出解释文字）。
        - 输出必须是**一个** JSON 对象；不得输出数组、字符串或多段对象；不得包含注释或多余键。

        【校验与修复顺序】
        1) 先解析用户意图到临时结构；
        2) 应用 1→0 索引规则；
        3) 规范 weights（截负→归一化；全 0→均匀分布）；
        4) 清洗 bands 值：缺失用 "-inf"/"inf"；保证 lo≤hi；
        5) 清洗 constraints：确保 kind/idx/sign/value 类型正确；soft 缺 penalty→10；
        6) 仅保留在 schema 中允许的键；禁止输出未定义键；
        7) 最终输出 JSON。

        【少量示例】
        - 示例 1（权重+参考点+硬约束）：
          输入： “目标1更重要，目标2一般；参考点 f1=0.2, f2=0.5；f1 必须 ≤ 0.6”
          输出：
          {{
            "weights": [0.7, 0.3],
            "reference": [0.2, 0.5],
            "constraints": [
              {{"kind":"obj","idx":0,"sign":"<=","value":0.6,"hard":true}}
            ]
          }}

        - 示例 2（稳健度与策略、软约束、带宽）：
          输入： “更稳健一些，先广搜后细调；目标2尽量 ≥0.4；f2 关注区间 [0.3, 0.7]”
          输出：
          {{
            "robustness": {{"cvar_alpha": 0.3}},
            "strategy":   {{"trust": 0.35}},
            "constraints": [
              {{"kind":"obj","idx":1,"sign":">=","value":0.4,"hard":false,"penalty":10}}
            ],
            "bands": [["-inf","inf"], [0.3, 0.7]]
          }}

        - 示例 3（变量屏蔽/几何屏蔽 → 变量等式/带宽）：
          输入： “不要动第 3 个变量；第 1 个变量在 [0.2,0.6]”
          输出：
          {{
            "constraints": [
              {{"kind":"var","idx":2,"sign":"==","value":0.0,"hard":true}}
            ],
            "var_bands": [[0.2,0.6]]  // 若你的 schema 支持变量带宽；否则只保留等式/不等式约束
          }}

        【用户输入】
        \"\"\"{user_text}\"\"\"
        """

        return prompt.strip()

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You convert natural language to strict JSON for optimization preferences."},
            {"role": "user",   "content": build_llm_prompt(user_text)}
        ]
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="ignore"))

        # 标准兼容结构：choices[0].message.content
        content = None
        try:
            content = obj["choices"][0]["message"]["content"]
        except Exception:
            # 兼容某些服务实现
            content = obj.get("content") or json.dumps(obj)

        # ---- 解析 JSON（容忍 ```json ...``` 包裹）----
        def _extract_json_block(s: str):
            if not isinstance(s, str): return None
            import re as _re
            m = _re.search(r"```json\s*(\{[\s\S]*?\})\s*```", s, _re.IGNORECASE)
            if m:
                try: return json.loads(m.group(1))
                except Exception: pass
            i, j = s.find("{"), s.rfind("}")
            if i != -1 and j != -1 and j > i:
                frag = s[i:j+1]
                try: return json.loads(frag)
                except Exception: pass
            try: return json.loads(s)
            except Exception: return None

        data = _extract_json_block(content)
        if data is None:
            print("[LLM] cannot parse JSON from content:", str(content)[:200])
            return None

        # ---- 1-based -> 0-based & 权重归一 ----
        if "bands" in data:
            for b in data["bands"]:
                b["obj_index"] = int(b["obj_index"]) - 1
        if "constraints" in data:
            for c in data["constraints"]:
                c["idx"] = int(c["idx"]) - 1
        if "weights" in data:
            w = [max(0.0, float(x)) for x in data["weights"]]
            s = sum(w); data["weights"] = [x/s for x in w] if s > 0 else [1.0/len(w)]*len(w)

        return data

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print("[LLM] HTTPError:", e.code, body[:300]); return None
    except Exception as e:
        print("[LLM] Request error:", e); return None



def regex_parse(user_text: str) -> Dict[str, Any]:
    t = user_text.strip().lower()
    out: Dict[str, Any] = {}

    m = re.search(r"w\s*=\s*\[([0-9eE\.\,\s\-]+)\]", t)
    if m:
        out["weights"] = [float(x) for x in m.group(1).split(",")]

    m = re.search(r"r\s*=\s*\[([0-9eE\.\,\s\-]+)\]", t)
    if m:
        out["reference"] = [float(x) for x in m.group(1).split(",")]

    m = re.findall(r"band\s*f(\d+)\s*\[\s*([0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)\s*\]", t)
    if m:
        out["bands"] = [{"obj_index": int(i)-1, "lo": float(lo), "hi": float(hi)} for i, lo, hi in m]

    m = re.findall(r"(hard|soft)\s+constraint\s*:\s*([xf])(\d+)\s*(<=|>=|==)\s*([0-9eE\.\-]+)(?:\s*penalty\s*=\s*([0-9eE\.\-]+))?", t)
    if m:
        cons = []
        for hs, xo, idx, sign, val, pen in m:
            cons.append({
                "kind": "var" if xo == "x" else "obj",
                "idx": int(idx) - 1,
                "sign": sign,
                "value": float(val),
                "hard": hs == "hard",
                "penalty": float(pen) if (pen and hs == "soft") else 10.0
            })
        out["constraints"] = cons

    m = re.search(r"cvar\s*=\s*([0-9]*\.?[0-9]+)", t)
    if m:
        out["robustness"] = {"cvar_alpha": float(m.group(1))}

    m = re.search(r"trust\s*=\s*([0-9]*\.?[0-9]+)", t)
    if m:
        out.setdefault("strategy", {})
        out["strategy"]["trust"] = float(m.group(1))

    return out


from typing import Optional, Dict, Any

def parse_nl_to_atoms(user_text: str,
                      model: Optional[str] = None,
                      api_key: Optional[str] = None,
                      base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    先走 LLM（讯飞 OpenAI 兼容端），失败就回退 regex；永不返回未定义变量。
    """
    model   = model   or DEFAULT_MODEL
    api_key = api_key or API_KEY_OVERRIDE
    base_url= base_url or DEFAULT_BASE_URL

    data: Optional[Dict[str, Any]] = None
    try:
        data = call_openai_chat(user_text, model=model, api_key=api_key, base_url=base_url)
    except Exception as e:
        print("[NL] call_openai_chat raised exception -> fallback to regex. Err:", e)

    if not data:
        data = regex_parse(user_text)

    if "weights" in data:
        # 再做一次归一（兜底）
        w = [max(0.0, float(x)) for x in data["weights"]]
        s = sum(w); data["weights"] = [x/s for x in w] if s > 0 else [1.0/len(w)]*len(w)

    # —— 放在 parse_nl_to_atoms 的 return data 之前 ——
    # 兜底：轻量识别 'w1=0.1' / '将w1改为0.1' / 'w1 0.1'
    if (not data) or (not isinstance(data, dict)):
        import re as _re
        t = str(user_text)
        m = _re.search(r'[wW]\s*1\s*[:=]?\s*([0-9]*\.?[0-9]+)', t)
        if not m:
            m = _re.search(r'将\s*[wW]1\s*(?:改为|设为|设置为|=|:)?\s*([0-9]*\.?[0-9]+)', t)
        if m:
            try:
                v = float(m.group(1))
                data = {"weights": [v]}  # 只给 w1，其它在主流程里“增量合并”
            except Exception:
                data = {}
        else:
            data = {}

    return data



# ------------------------------
# 将 Atoms 标准化为 UPV（仅做填充，不做优化）
# ------------------------------
from typing import List, Tuple

def atoms_to_upv(atoms: Dict[str, Any], M: int, Nvar: int) -> UPV:
    # 1) 权重
    if isinstance(atoms.get("weights"), list) and atoms["weights"]:
        w = [max(0.0, float(x)) for x in atoms["weights"]]
        s = sum(w); w = [x/s for x in w] if s > 0 else [1.0/max(1,M)]*max(1,M)
    else:
        w = [1.0/max(1,M)]*max(1,M)
    if len(w) != M:
        w = (w + [1.0/M]*M)[:M]

    # 2) 参考点
    r = atoms.get("reference", [0.0]*M)
    r = (list(r) + [0.0]*M)[:M]
    r = [float(x) for x in r]

    # 3) 目标带宽
    bands: List[Tuple[float,float]] = [(-float("inf"), float("inf")) for _ in range(M)]
    for band in atoms.get("bands", []) or []:
        try:
            j = int(band.get("obj_index", -1))
            if 0 <= j < M:
                lo = float(band.get("lo", -float("inf")))
                hi = float(band.get("hi",  float("inf")))
                bands[j] = (lo, hi)
        except Exception:
            pass

    # 4) 约束（硬/软）
    C_h: List[ConstraintAtom] = []
    C_s: List[ConstraintAtom] = []
    for c in atoms.get("constraints", []) or []:
        try:
            ca = ConstraintAtom(
                kind=c["kind"], idx=int(c["idx"]), sign=c["sign"],
                value=float(c["value"]), hard=bool(c.get("hard", True)),
                penalty=float(c.get("penalty", 10.0))
            )
            if ca.hard: C_h.append(ca)
            else:       C_s.append(ca)
        except Exception as e:
            print("[atoms_to_upv] skip constraint due to error:", e)

    # 5) 变量域（留空→由问题边界填充）
    D: List[Tuple[float,float]] = []

    # 6) 风险/策略
    rho = Robustness(**(atoms.get("robustness") or {})) if "robustness" in atoms else Robustness()
    tau = float((atoms.get("strategy") or {}).get("trust", 0.2))
    pi  = Strategy(trust=tau, explore_bias=float((atoms.get("strategy") or {}).get("explore_bias", 0.0)))

    return UPV(w=w, r=r, b=bands, C_h=C_h, C_s=C_s, D=D, tau=tau, rho=rho, pi=pi, meta={"source": "NL"})



# ------------------------------
# 示例语句（中文，自然语言）
# ------------------------------
EXAMPLES = [
    # 权重+参考点+软约束+信赖域+CVaR
    "把第一目标权重设为0.7，第二目标设为0.3；参考点 r=[0.2,0.6]；f2 至少 0.5 是软约束，罚项 20；信赖域 0.2；CVaR=0.9。",
    # 仅参考点与区间
    "希望 f1 落在 [0.1,0.3]，f2 落在 [0.4,0.7]；r=[0.2,0.5]。",
    # 硬约束（变量或目标）
    "必须满足 x1 <= 0.5；同时 f2 <= 0.8 必须满足；其它都尽量靠近 r=[0.2,0.6]。",
    # 混合表达
    "尽量把 f1 做小一些（权重高一点），f2 也要考虑（不要太差）。我希望信赖域小一些，比如 0.15，此外 CVaR 设成 0.85。",
    # 英文混合（便于跨语）
    "Weights w=[0.55,0.45]; soft constraint: f2>=0.55 penalty=15; trust=0.2; cvar=0.9.",
    # 仅权重
    "我更在意第一目标，大概 0.6 比 0.4 吧。",
    # 仅CVaR/信赖域
    "稳健性要强一点，CVaR=0.95；搜索别太跳，信赖域 0.1。",
    # 仅软约束
    "f1 最好不要超过 0.3（软约束，罚项 10），f2 尽量不低于 0.5（软约束，罚项 25）。",
    # 仅硬约束
    "必须满足 x1 >= 0.2 且 f2 <= 0.75。",
    # 口语表达
    "第一目标优先（比第二重要），希望目标2不低于0.55，能做到就好，做不到也行。"
]


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="", help="单条自然语言；留空进入交互模式或指定 --demo")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--M", type=int, default=2, help="目标个数（仅用于UPV默认长度）")
    parser.add_argument("--Nvar", type=int, default=30, help="变量个数（仅用于占位）")
    parser.add_argument("--demo", action="store_true", help="使用内置示例批量演示")
    args = parser.parse_args()

    api_key = args.api_key or API_KEY_OVERRIDE or os.getenv("OPENAI_API_KEY", "")

    def run_one(utter: str):
        print("\n=== NL 输入 ===")
        print(utter)
        atoms = parse_nl_to_atoms(utter, model=args.model, api_key=api_key)
        print("\n--- LLM/正则 解析为 Atoms(JSON) ---")
        print(json.dumps(atoms, ensure_ascii=False, indent=2))
        try:
            upv = atoms_to_upv(atoms, M=args.M, Nvar=args.Nvar)
            print("\n--- 标准化 UPV(JSON) ---")
            # print(upv.model_dump_json(indent=2, ensure_ascii=False))
            print(json.dumps(upv.model_dump(), indent=2, ensure_ascii=False))

        except ValidationError as ve:
            print("\n[校验失败] 无法转成UPV：", ve)

    if args.text:
        run_one(args.text)
        return

    if args.demo:
        for ut in EXAMPLES:
            run_one(ut)
        return

    print("进入交互模式（回车直接退出）。提示：可输入示例如：")
    print("  w=[0.6,0.4]；r=[0.2,0.6]；soft constraint: f2 >= 0.5 penalty=20；trust=0.2；cvar=0.9")
    print("  必须满足 x1 <= 0.5；f2 至少 0.55（软约束罚项15）; 我更在意第一目标(0.7:0.3)")
    while True:
        s = input("\nNL> ").strip()
        if not s:
            print("已退出。")
            break
        run_one(s)


if __name__ == "__main__":
    main()



def llm_nl_to_patch(user_text: str,
                    model: str | None = None,
                    api_key: str | None = None,
                    base_url: str | None = None) -> dict:
    """
    Convert Chinese NL to a STRICT JSON patch via LLM (XFYUN OpenAI-compatible /v1/chat/completions).
    No regex fallback; raises on failure.
    Patch schema:
    {
      "mode": "patch",
      "ops": [
        {"op":"set_weight", "index":0, "value":0.1},
        {"op":"set_cvar",   "value":0.1},
        {"op":"set_trust",  "value":0.2},
        {"op":"set_reference", "values":[0.2,0.6]},
        {"op":"set_band",   "obj_index":1, "lo":-1.0, "hi":2.0},
        {"op":"add_constraint","kind":"obj","idx":1,"sign":">=","value":0.5,"hard":false,"penalty":20.0}
      ]
    }
    """
    api_key = api_key or API_KEY_OVERRIDE
    base_url = base_url or DEFAULT_BASE_URL
    model    = model    or DEFAULT_MODEL
    if not api_key:
        raise RuntimeError("LLM API Key is empty. Set API_KEY_OVERRIDE in NL2UPV.py.")

    # normalize base_url to /v1 and endpoint
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    url = base_url.rstrip("/") + "/chat/completions"

    SYSTEM = "You convert Chinese natural language into STRICT JSON patches for an optimizer."
    PROMPT = f"""
只输出 JSON（不要解释文字）。将用户中文需求转为补丁（patch），用于修改一个已存在的 UPV。
格式：
{{
  "mode": "patch",
  "ops": [
    {{"op":"set_weight","index":0,"value":0.1}},
    {{"op":"set_cvar","value":0.1}},
    {{"op":"set_trust","value":0.2}},
    {{"op":"set_reference","values":[0.2,0.6]}},
    {{"op":"set_band","obj_index":1,"lo":-1.0,"hi":2.0}},
    {{"op":"add_constraint","kind":"obj","idx":1,"sign":">=","value":0.5,"hard":false,"penalty":20.0}}
  ]
}}
要求：
- 仅返回与本次修改相关的操作；不要输出当前 UPV 的快照。
- 若自然语言用 1-based（例如 w1 / 目标2），请在 JSON 中改为 0-based。
用户输入：
\"\"\"{user_text}\"\"\"
""".strip()

    payload = {
        "model": model, "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": PROMPT},
        ]
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="ignore"))
        content = obj["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM HTTPError {e.code}: {e.read().decode('utf-8', errors='ignore')[:300]}")
    except Exception as e:
        raise RuntimeError(f"LLM Request Error: {e}")

    # Extract strict JSON
    def _extract_json(s: str):
        import re
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", s, re.IGNORECASE)
        if m:
            return json.loads(m.group(1))
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(s[i:j+1])
        return json.loads(s)

    data = _extract_json(content)
    if not isinstance(data, dict) or data.get("mode") != "patch" or not isinstance(data.get("ops"), list):
        raise RuntimeError(f"LLM did not return a valid patch: {content[:200]}")
    return data


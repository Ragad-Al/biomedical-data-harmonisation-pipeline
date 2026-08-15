# onboarding/transform_library.py
from __future__ import annotations
from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np

def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[col]

def passthrough(df: pd.DataFrame, inputs: Union[str, List[str], Dict[str, str]], params: Dict[str, Any]) -> pd.Series:
    if isinstance(inputs, list):
        col = inputs[0] if inputs else ""
    elif isinstance(inputs, dict):
        col = next(iter(inputs.values()), "")
    else:
        col = inputs
    return _series(df, col)

def choose_first_non_null(df: pd.DataFrame, inputs: List[str], params: Dict[str, Any]) -> pd.Series:
    out = pd.Series([""] * len(df), index=df.index)
    for col in inputs:
        if not col or col not in df.columns:
            continue
        s = df[col].where(~df[col].isna(), "").astype(str)
        mask = (out == "") & (s != "")
        out.loc[mask] = s.loc[mask]
    if params.get("uppercase"):
        out = out.astype(str).str.upper()
    return out

def strip_chr_prefix(df: pd.DataFrame, inputs: Union[str, List[str]], params: Dict[str, Any]) -> pd.Series:
    col = inputs[0] if isinstance(inputs, list) else inputs
    s = _series(df, col).astype(str)
    if params.get("case_insensitive", True):
        s = s.str.replace(r"^chr", "", regex=True, case=False)
    else:
        s = s.str.replace(r"^chr", "", regex=True)
    return s.str.strip()

def uppercase(df: pd.DataFrame, inputs: Union[str, List[str]], params: Dict[str, Any]) -> pd.Series:
    col = inputs[0] if isinstance(inputs, list) else inputs
    return _series(df, col).astype(str).str.upper().str.strip()

def clamp_0_1(df: pd.DataFrame, inputs: Union[str, List[str]], params: Dict[str, Any]) -> pd.Series:
    col = inputs[0] if isinstance(inputs, list) else inputs
    invalid_to = params.get("invalid_to", "")
    s = pd.to_numeric(_series(df, col), errors="coerce")
    s = s.where((s >= 0) & (s <= 1), np.nan)
    return s.where(~s.isna(), invalid_to)

def to_int(df: pd.DataFrame, inputs: Union[str, List[str]], params: Dict[str, Any]) -> pd.Series:
    col = inputs[0] if isinstance(inputs, list) else inputs
    invalid_to = params.get("invalid_to", "")
    s = pd.to_numeric(_series(df, col), errors="coerce")
    s = s.round(0)
    s = s.astype("Int64")  # nullable integer
    return s.where(~s.isna(), invalid_to)

def to_float(df: pd.DataFrame, inputs: Union[str, List[str]], params: Dict[str, Any]) -> pd.Series:
    col = inputs[0] if isinstance(inputs, list) else inputs
    invalid_to = params.get("invalid_to", "")
    s = pd.to_numeric(_series(df, col), errors="coerce")
    return s.where(~s.isna(), invalid_to)

import re

def strip_suffix_regex(df: pd.DataFrame, inputs, params: Dict[str, Any]) -> pd.Series:
    col = inputs[0] if isinstance(inputs, list) else inputs
    pattern = params.get("pattern", r"\.[0-9]+$")
    s = _series(df, col).astype(str)
    return s.str.replace(pattern, "", regex=True)

def to_float(df: pd.DataFrame, inputs, params: Dict[str, Any]) -> pd.Series:
    col = inputs[0] if isinstance(inputs, list) else inputs
    invalid_to = params.get("invalid_to", "")
    s = pd.to_numeric(_series(df, col), errors="coerce")
    return s.where(~s.isna(), invalid_to)

TEMPLATES = {
    "passthrough": passthrough,
    "choose_first_non_null": choose_first_non_null,
    "strip_chr_prefix": strip_chr_prefix,
    "uppercase": uppercase,
    "clamp_0_1": clamp_0_1,
}

TEMPLATES.update({
    "to_int": to_int,
    "to_float": to_float,
    "strip_suffix_regex": strip_suffix_regex,
})
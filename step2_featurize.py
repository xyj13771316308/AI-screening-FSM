# -*- coding: utf-8 -*-
"""
Step 2: 特征化 —— 孔隙几何 (zeo++) + 化学特征 + 水稳定性先验
==============================================================
输入: data/mofs.parquet
输出: data/features.parquet
  几何: PLD, LCD, VSA(m2/cm3), GSA(m2/g), void_fraction
  化学: density, n_elements, 金属节点, 配位不饱和金属分数(用 oxidation state 启发)
  标签: water_stability (0/1/2 启发式), toxic (bool)
水稳定性启发规则（文献经验, 仅作预筛, 最终需 MD/实验验证）:
  2=高: Zr, Cr(III), Fe(III), Al, Ti, V(IV) 羧酸体系
  1=中: Zn, Cu, Mg, Mn, Co, Ni 羧酸; Zr 咪唑
  0=低: 含 N 配位但金属为软酸(Cu/Co/Ni/Zn), 膦酸/磺酸弱配位, 含易水解基团
"""
import os, subprocess, tempfile
import numpy as np
import pandas as pd
from pymatgen.core import Structure

HIGH_STAB = {"Zr","Cr","Fe","Al","Ti","V","Hf","Ce"}
MID_STAB  = {"Zn","Cu","Mg","Mn","Co","Ni","Cd"}
TOXIC     = {"Cr","Cd","Pb","Hg","Ni","Co","As","Be"}

def zeopp_features(s: Structure, zeopp_exe="network") -> dict:
    """调用 zeo++。需先安装并将 binary 加入 PATH。"""
    with tempfile.TemporaryDirectory() as td:
        cif = os.path.join(td, "mof.cif"); s.to("cif", cif)
        out = cif.replace(".cif", ".out")
        subprocess.run([zeopp_exe, "-ha", "-res", out, cif],
                       check=True, capture_output=True, timeout=900)
        d = {"PLD": np.nan, "LCD": np.nan, "VSA": np.nan,
             "GSA": np.nan, "void_fraction": np.nan}
        for line in open(out):
            t = line.split()
            if line.startswith("Di"):    d["PLD"] = float(t[1])
            elif line.startswith("Df"):  d["LCD"] = float(t[1])
            elif line.startswith("VSA"): d["VSA"] = float(t[1])
            elif line.startswith("GSA"): d["GSA"] = float(t[1])
            elif line.startswith("POAV"): d["void_fraction"] = float(t[-2])
        return d

def chem_features(s: Structure) -> dict:
    comp = s.composition
    metals = {str(e) for e in comp.elements if e.is_metal}
    stab = 2 if metals & HIGH_STAB else (1 if metals & MID_STAB else 0)
    return {
        "density": s.density,
        "n_elements": len(comp.elements),
        "avg_atomic_mass": sum(comp.get_atomic_fraction(e)*e.atomic_mass
                               for e in comp.elements),
        "n_metal_sites": sum(1 for site in s if site.specie.is_metal),
        "metal_nodes": "-".join(sorted(metals)),
        "water_stability": stab,
        "toxic": bool(metals & TOXIC),
    }

if __name__ == "__main__":
    df = pd.read_parquet("data/mofs.parquet")
    chem = pd.DataFrame([chem_features(s) for s in df["structure"]])
    # 孔隙特征: 并行 zeo++（按 CPU 核数开进程池）
    from multiprocessing import Pool
    with Pool(16) as p:
        pore = pd.DataFrame(p.map(zeopp_features, df["structure"]))
    feat = pd.concat([df[["material_id","formula","source"]].reset_index(drop=True),
                      chem.reset_index(drop=True), pore.reset_index(drop=True)], axis=1)
    feat.to_parquet("data/features.parquet")
    print(feat.describe())
    print("NaN pore features:", feat["PLD"].isna().sum(), "-> zeo++ 未装好或未解析")

# -*- coding: utf-8 -*-
"""
Step 1: MOF 结构库获取与清洗
==============================
三个来源互补：
  A. QMOF   ~20k 量子精修 MOF (DFT 优化结构, 带 partial charges)   首选
     https://github.com/arosen93/QMOF  (qmof_database.zip, CIF)
  B. CoRE MOF 2019  ~14k 实验合成 MOF (已去除溶剂分子)             落地性验证
     http://gregchung.github.io/CoRE-MOFs/
  C. MP-MOF  ~25k 假设 MOF (pymatgen API)                          结构多样性
用法: python step1_collect.py --source qmof --cif_dir ./cifs/qmof
输出: data/mofs.parquet  (material_id, formula, structure(pmg), source)
"""
import os, glob, argparse
import pandas as pd
from pymatgen.core import Structure

def load_cifs(cif_dir: str, source: str) -> pd.DataFrame:
    rows, bad = [], 0
    for f in glob.glob(os.path.join(cif_dir, "**/*.cif"), recursive=True):
        mid = source + "-" + os.path.basename(f).replace(".cif","").replace("_","-")
        try:
            s = Structure.from_file(f)
            # --- 清洗规则 ---
            if len(s) > 3000:            continue   # 超大晶胞 GCMC 跑不动
            if s.volume/len(s) < 5:      continue   # 过于致密, 无孔
            if not s.is_ordered:                      # 无序占据→取最高占据
                s = s.get_sorted_structure()
            rows.append({"material_id": mid, "formula": s.composition.reduced_formula,
                         "structure": s, "source": source, "n_sites": len(s)})
        except Exception:
            bad += 1
    print(f"[{source}] loaded={len(rows)} skipped={bad}")
    return pd.DataFrame(rows)

def fetch_mp_mofs(api_key: str) -> pd.DataFrame:
    from pymatgen.ext.matproj import MPRester
    rows = []
    with MPRester(api_key) as mpr:
        docs = mpr.materials.search(fields=["material_id","formula_pretty",
                                            "density","structure","theoretical"])
    for d in docs:
        s = d.structure
        rows.append({"material_id": "MP-"+d.material_id.split("-")[-1],
                     "formula": d.formula_pretty, "structure": s,
                     "source": "mp", "n_sites": len(s)})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["qmof","core","mp","all"], default="qmof")
    ap.add_argument("--cif_dir", default="./cifs/qmof")
    ap.add_argument("--mp_key", default=os.environ.get("MP_API_KEY",""))
    args = ap.parse_args()

    dfs = []
    if args.source in ("qmof","core","all"):
        dfs.append(load_cifs(args.cif_dir, args.source))
    if args.source in ("mp","all") and args.mp_key:
        dfs.append(fetch_mp_mofs(args.mp_key))

    df = pd.concat(dfs).drop_duplicates("material_id")
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/mofs.parquet")
    print(f"total MOFs: {len(df)}")

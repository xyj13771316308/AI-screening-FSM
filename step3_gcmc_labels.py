# -*- coding: utf-8 -*-
"""
Step 3: GCMC 标签生成（RASPA2）—— 全流程最贵的步骤，只跑被选中子集
====================================================================
对每个选中 MOF 计算两个标签：
  y1 loading_mmol_g     氟砜草胺在 298K, 目标活度下的平衡载药量 (mmol/g)
  y2 ads_enthalpy_kJ_mol 低覆盖极限吸附焓 (kJ/mol, 负值)
场景设定（叶面喷施）:
  - 模拟"露水润湿叶面"：水活度 a_w=0.9, 药物活度 a_FSM=0.1, 竞争吸附
  - 输出两个工况: 干基载药(纯 FSM 蒸气) + 湿基竞争(水+FSM)
力场:
  MOF: UFF4MOF (框架柔性关闭, 固定骨架)
  FSM: GAFF2 + AM1-BCC 电荷 (step0 的 mol2), S=O 亚砜注意 antechamber 类型 so
  水: TIP4P/2005
运行: 单 MOF ~20-60 min (1e7+1e7 步, 24 核). 100 个 MOF 并行集群更佳.
"""
import os, subprocess, shutil, json, argparse, re
import pandas as pd

TEMPLATE = """
SimulationType                MonteCarlo
NumberOfCycles                10000000
NumberOfInitializationCycles  10000000
PrintEvery                    100000
RestartFile                   no

Forcefield                    UFF4MOF
ChargeMethod                  Ewald
CutOffVDW                     12.0
CutOffCoulomb                 12.0

Framework 0
FrameworkName {name}
UnitCells 1 1 1
HeliumVoidFraction {voidf}
ExternalTemperature 298.0
ExternalPressure {press_pa:.1f}    // 由活度换算: p = a * p_sat(FSM,298K)

Component 0 MoleculeName             flusulfinam
            MoleculeDefinition       local
            FugacityCoefficient      1.0
            TranslationProbability   0.3
            RotationProbability      0.3
            ReinsertionProbability   0.2
            SwapProbability          0.2
            CreateNumberOfMolecules  0

Component 1 MoleculeName             water_tip4p
            MoleculeDefinition       local
            TranslationProbability   0.3
            RotationProbability      0.3
            SwapProbability          0.4
            CreateNumberOfMolecules  0
"""

PSAT_FSM_PA = 0.02   # 估算: MW379 低挥发除草剂, 298K 蒸气压 ~10^-4 mmHg ≈ 0.013 Pa; 取保守值
                     # 请用 Antoine/估算方法替换为文献/计算值

def run_one(cif_path, workdir, press_pa, voidf, raspa_exe="simulate"):
    os.makedirs(workdir, exist_ok=True)
    shutil.copy(cif_path, os.path.join(workdir, "framework.cif"))
    with open(os.path.join(workdir, "simulation.input"), "w") as f:
        f.write(TEMPLATE.format(name="framework", voidf=voidf, press_pa=press_pa))
    subprocess.run([raspa_exe], cwd=workdir, check=True,
                   capture_output=True, timeout=86400)
    return parse_raspa(os.path.join(workdir, "Output",
                       "System_0", "output_framework_298.000000_*.txt"))

def parse_raspa(out_glob):
    import glob
    loading, enthalpy = None, None
    for fp in glob.glob(out_glob):
        for line in open(fp):
            if "Average loading excess [mmol/g]" in line:
                loading = float(line.split()[-1])
            if "Average enthalpy of adsorption" in line and "kJ/mol" in line:
                enthalpy = float(line.split()[-2])
    return loading, enthalpy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="候选 material_id 列表, 逗号分隔")
    ap.add_argument("--cif_dir", default="./cifs/qmof")
    args = ap.parse_args()
    feat = pd.read_parquet("data/features.parquet")
    feat = feat[feat["material_id"].isin(args.ids.split(","))]
    rows = []
    for _, r in feat.iterrows():
        wd = f"raspa_runs/{r.material_id}"
        loading, enth = run_one(os.path.join(args.cif_dir, r.material_id + ".cif"),
                                wd, press_pa=0.1*PSAT_FSM_PA,
                                voidf=r.get("void_fraction") or 0.5)
        rows.append({"material_id": r.material_id,
                     "loading_mmol_g": loading, "ads_enthalpy_kJ_mol": enth})
        pd.DataFrame(rows).to_csv("data/labels.csv", index=False)  # 断点续传
    print(pd.DataFrame(rows))

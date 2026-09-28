# -*- coding: utf-8 -*-
"""
Step 0: 氟砜草胺分子处理 —— 描述符 + 3D 构象 + RESP 电荷文件
==============================================================
输出:
  data/flusulfinam_3d.sdf      优化构象
  data/flusulfinam.mol2        GAFF 力场 + RESP 电荷（RASPA 用）
  data/drug_descriptors.json   全部描述符与孔径阈值
"""
import os, json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

SMILES = "CCCS(=O)c1c(F)c(ccc1C(F)(F)F)C(=O)Nc2oc(C)nn2"
OUT = "data"; os.makedirs(OUT, exist_ok=True)

mol = Chem.MolFromSmiles(SMILES)
mol = Chem.AddHs(mol)
params = AllChem.ETKDGv3(); params.randomSeed = 42
AllChem.EmbedMolecule(mol, params)
AllChem.MMFFOptimizeMolecule(mol, maxIters=5000)

# ---- 描述符 ----
conf = mol.GetConformer()
coords = conf.GetPositions()
r_gyr = float(np.sqrt(((coords - coords.mean(0))**2).sum() / len(coords)))
vdW_vol = AllChem.ComputeMolVolume(mol, confId=0)

desc = {
    "name": "flusulfinam", "cas": "2428458-82-4", "smiles": SMILES,
    "MW": Descriptors.MolWt(mol),
    "LogP": Descriptors.MolLogP(mol),
    "TPSA": Descriptors.TPSA(mol),
    "HBD": Descriptors.NumHDonors(mol),
    "HBA": Descriptors.NumHAcceptors(mol),
    "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
    "vdW_volume_A3": vdW_vol,
    "r_gyr_A": r_gyr,
    "equiv_sphere_d_A": (6*vdW_vol/np.pi)**(1/3),
    "kinetic_d_est_A": 2.0*r_gyr,
}
# ---- 孔径阈值（叶面喷施场景经验规则）----
desc["MIN_PLD_A"] = max(10.0, 1.4 * desc["kinetic_d_est_A"])   # 孔喉: 扩散下限
desc["MIN_LCD_A"] = 1.2 * desc["equiv_sphere_d_A"]             # 空腔: 容纳下限
json.dump(desc, open(f"{OUT}/drug_descriptors.json","w"), indent=2)
print(json.dumps({k: round(v,2) for k,v in desc.items()}, ensure_ascii=False, indent=1))

with Chem.SDWriter(f"{OUT}/flusulfinam_3d.sdf") as w:
    w.write(mol)

# ---- RESP 电荷（RASPA 需要 .mol2；Psi4/ORCA 算 RESP，此处给 antechamber 流程）----
"""
本地执行（需要 ambertools + Gaussian/ORCA 走 RESP）:
  # 1) 生成 RESP 输入（以 ORCA 为例）
  orca resp.inp  -> 得到 .gbw 与偶极信息
  # 2) 或快速近似：am1-bcc 电荷对 GCMC 通常足够
  antechamber -i data/flusulfinam_3d.sdf -fi sdf -o data/flusulfinam.mol2 -fo mol2 \
              -c bcc -at gaff2 -nc 0 -rn FSM
  # 3) 检查参数完整性
  parmchk2 -i data/flusulfinam.mol2 -f mol2 -o data/FSM.frcmod
"""

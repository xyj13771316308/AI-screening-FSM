# -*- coding: utf-8 -*-
"""
Step 4: ALIGNN 图神经网络训练（GPU）
=====================================
为什么用 GNN 而非 XGBoost:
  - MOF 性质由局部配位环境决定, ALIGNN 直接学习原子图+线图的局部结构
  - 小样本 (~数百 GCMC 标签) 下 GNN+预训练迁移优于手工特征
  - NIST ALIGNN 在吸附/扩散类性质上 SOTA
安装: pip install alignn (torch>=2.0, 需 GPU)
"""
import os, json, argparse
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from alignn.data import get_train_val_loaders
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.train import train_dgl  # alignn 训练入口

def pmg_to_jarvis(structure):
    return Atoms(lattice_mat=structure.lattice.matrix,
                 coords=structure.frac_coords,
                 elements=[str(s.specie) for s in structure],
                 cartesian=False)

def make_dataset(features_parquet, labels_csv, out_dir, target="loading_mmol_g"):
    feat = pd.read_parquet(features_parquet)
    lab  = pd.read_csv(labels_csv)
    df = feat.merge(lab[["material_id", target]], on="material_id").dropna(subset=[target])
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for _, r in df.iterrows():
        atoms = pmg_to_jarvis(r["structure"])
        entries.append({"id": r.material_id, "atoms": atoms.to_dict(),
                        target: r[target]})
    json.dump(entries, open(os.path.join(out_dir, f"id_{target}.json"), "w"))
    return len(entries)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="loading_mmol_g")
    ap.add_argument("--data_dir", default="data/alignn_loading")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    n = make_dataset("data/features.parquet", "data/labels.csv",
                     args.data_dir, args.target)
    print(f"dataset size: {n}")
    config = ALIGNNConfig(
        name="alignn", epochs=args.epochs, batch_size=16,
        learning_rate=1e-3, target=args.target,
        n_conv=4, atom_input_features=92,
        output_dir=f"./alignn_out_{args.target}",
    )
    # alignn 命令行训练（推荐）:
    #   train_alignn.py --root_dir data/alignn_loading --config config.json --epochs 300
    # 此处留 programmatic 入口; 若版本差异请参考 alignn 文档
    print("Run: train_alignn.py --root_dir", args.data_dir,
          "--target", args.target, "--epochs", args.epochs,
          "--output_dir", config.output_dir)

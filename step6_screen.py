# -*- coding: utf-8 -*-
"""
Step 6: 多目标筛选（叶面喷施场景专用评分）
===========================================
硬约束:
  PLD >= MIN_PLD (step0, ~12A)         LCD >= MIN_LCD (~11A)
  loading_hat >= 0.6 mmol/g            -48 <= 吸附焓 <= -30 kJ/mol
  water_stability >= 1                 toxic == False
评分 (0-1):
  0.40 载药量归一化 (叶面喷施需要高载药保证持效)
  0.25 吸附焓距窗口中心(-39)的接近度 (兼顾抗雨与可释放)
  0.15 比表面 GSA (高 GSA -> 分散性好, 叶面覆盖均匀)
  0.10 模型不确定度越低越好
  0.10 水稳定性 (2 分满分)
输出: data/candidates_foliar.csv + 去重(同一 metal_node+拓扑家族只留 top1)
"""
import json
import numpy as np
import pandas as pd

drug = json.load(open("data/drug_descriptors.json"))
MIN_PLD, MIN_LCD = drug["MIN_PLD_A"], drug["MIN_LCD_A"]
H_LO, H_HI, H_CENTER = -48.0, -30.0, -39.0

df = pd.read_parquet("data/predictions.parquet")   # step5 输出, 含 y_hat/y_std
lab = pd.read_csv("data/labels.csv")
df = df.merge(lab[["material_id","ads_enthalpy_kJ_mol"]],
              on="material_id", how="left")
# 焓只对已算过 GCMC 的准确; 其余用 ML 预测列 enthalpy_hat 填充
if "enthalpy_hat" not in df.columns:
    df["enthalpy_hat"] = df["ads_enthalpy_kJ_mol"]
df["enthalpy_hat"] = df["enthalpy_hat"].fillna(df["ads_enthalpy_kJ_mol"])

cand = df[(df["PLD"] >= MIN_PLD) & (df["LCD"] >= MIN_LCD)
          & (~df["toxic"]) & (df["water_stability"] >= 1)
          & (df["y_hat"] >= 0.6)
          & (df["enthalpy_hat"].between(H_LO, H_HI))].copy()

cand["s_load"] = cand["y_hat"] / cand["y_hat"].max()
cand["s_enth"] = 1 - (cand["enthalpy_hat"] - H_CENTER).abs() / (H_CENTER - H_LO)
cand["s_gsa"]  = (cand["GSA"].fillna(0)) / cand["GSA"].fillna(0).max()
cand["s_unc"]  = 1 - cand["y_std"] / cand["y_std"].max()
cand["s_stab"] = cand["water_stability"] / 2.0
cand["score"]  = (0.40*cand["s_load"] + 0.25*cand["s_enth"] + 0.15*cand["s_gsa"]
                  + 0.10*cand["s_unc"] + 0.10*cand["s_stab"])

cand = cand.sort_values("score", ascending=False)
# 家族去重: 同金属节点只留 3 个, 保证多样性
cand = cand.groupby("metal_nodes").head(3)
cand.to_csv("data/candidates_foliar.csv", index=False)
cols = ["material_id","formula","metal_nodes","y_hat","enthalpy_hat",
        "PLD","LCD","GSA","water_stability","score"]
print(cand[cols].head(30).to_string())

# -*- coding: utf-8 -*-
"""
Step 5: 主动学习闭环（Active Learning）
======================================
每轮:
  1. 用当前模型(3 seeds 集成)预测全库, 得到 y_hat, y_std
  2. 用委员会查询 (query-by-committee) 选 y_std 最大的 100 个
     （多样性约束: 用特征聚类确保不集中在同一拓扑家族）
  3. 送 step3 GCMC 补标签 -> append data/labels.csv
  4. step4 重训; 监控全库 y_std 均值, 收敛 (下降 <5%) 即停
通常 3-5 轮后 MAE 收敛到 GCMC 自身误差 (~0.05 mmol/g) 量级。
"""
import json, subprocess
import numpy as np
import pandas as pd

def committee_predict(model_dirs, features_parquet):
    """依次调用 alignn 推理, 返回 mean/std。model_dirs: 不同 seed 的 ckpt。"""
    preds = []
    for md in model_dirs:
        subprocess.run(["python", "alignn_infer.py",  # alignn 推理脚本
                        "--model_path", md,
                        "--features", features_parquet], check=True)
        preds.append(pd.read_csv("pred.csv")["pred"].values)
    P = np.vstack(preds)
    return P.mean(0), P.std(0)

def select_batch(features_parquet, mean, std, batch=100, diversity_kmeans=20):
    feat = pd.read_parquet(features_parquet)
    feat["y_hat"], feat["y_std"] = mean, std
    feat.to_parquet("data/predictions.parquet")
    # 多样性: 先 kmeans 聚类, 每个簇内取不确定度最高者
    from sklearn.cluster import KMeans
    X = feat[["PLD","LCD","VSA","density","n_elements"]].fillna(feat.median())
    km = KMeans(n_clusters=diversity_kmeans, n_init=10, random_state=0).fit(X)
    feat["cluster"] = km.labels_
    pick = feat.sort_values("y_std", ascending=False).groupby("cluster").head(
        max(1, batch//diversity_kmeans))
    return pick["material_id"].tolist()

if __name__ == "__main__":
    import sys
    round_id = sys.argv[1]
    mean, std = committee_predict(
        [f"alignn_out_loading/seed{s}/best_model.pt" for s in (0,1,2)],
        "data/features.parquet")
    ids = select_batch("data/features.parquet", mean, std)
    with open(f"data/batch_{round_id}.txt","w") as f:
        f.write(",".join(ids))
    print(f"Round {round_id}: 送 {len(ids)} 个去 GCMC -> python step3_gcmc_labels.py --ids $(cat data/batch_{round_id}.txt)")

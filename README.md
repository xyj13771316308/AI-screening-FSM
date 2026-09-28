# MOF 载药筛选工程：氟砜草胺（Flusulfinam）叶面喷施载体
================================================================

## 分子档案（step0 已确认）
- 名称: 氟砜草胺 / Flusulfinam, CAS 2428458-82-4
- 分子式: C14H13F4N3O3S, MW 379.33
- 结构: 2-氟-N-(5-甲基-1,3,4-噁二唑-2-基)-3-丙基亚砜基-4-三氟甲基苯甲酰胺
- SMILES: CCCS(=O)c1c(F)c(ccc1C(F)(F)F)C(=O)Nc2oc(C)nn2
- 注意: 分子中亚砜硫为手性中心，商品药为 R-rich (80-100% R)；本流程按消旋/外消旋处理，
  若只做 R 体，step0 加手性标记即可（对吸附性质影响通常 <2 kJ/mol）
- 关键尺寸（估算）: vdW 体积 ~397 A^3, 等效球径 ~9.1 A, 动力学直径 ~8.7 A

## 场景定义：叶面喷施（foliar spray）决定的目标函数
与土壤缓释完全不同，叶面场景要求：
  1. 抗雨水冲刷 (rainfastness)  -> 结合偏强: 吸附焓窗口 -30 ~ -48 kJ/mol
  2. 露水/湿度触发可释放       -> 不能太強: 上限 -48（弱于化学吸附阈值 ~-60）
  3. 高载药量                  -> loading >= 0.6 mmol/g (~23 wt%)
  4. 叶面润湿与粘附            -> 粒度 0.2-2 um（step6 后处理，DB 不含量）
  5. 水稳定性                  -> 金属节点 Zr/Fe/Al/Cr(III)/Ti 优先, Zn 次之
  6. 光保护                    -> MOF 骨架遮蔽 UV（后续验证项）

## 目录结构
  step0_drug.py          分子描述符 + 3D 构象 + RESP 电荷准备
  step1_collect.py       三大 MOF 数据库下载/清洗 (MP-MOF / QMOF / CoRE MOF)
  step2_featurize.py     几何特征 (zeo++) + 化学特征 (matminer) + 水稳定性标签
  step3_gcmc_labels.py   RASPA GCMC 批量标签生成 (loading + 吸附焓)
  step4_train_alignn.py  ALIGNN 图神经网络训练 (GPU)
  step5_active_learning.py 主动学习闭环 (uncertainty sampling)
  step6_screen.py        多目标筛选 + 评分 + 候选清单

## 硬件/软件
  GPU: >=8GB 显存 (ALIGNN), RASPA/zeo++ 用 CPU 多核
  conda env: python=3.10, torch, alignn, pymatgen, matminer, rdkit, xgboost, ase
  外部二进制: zeo++ (https://www.zeoplusplus.org), RASPA2 (https://github.com/iRASPA/RASPA2)

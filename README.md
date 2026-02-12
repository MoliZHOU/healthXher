# RA 预测 Pipeline — 快速上手指南

## 目录结构

```
ra_pipeline/
├── ra_pipeline.py        # 核心代码（全部逻辑在一个文件中）
├── README.md             # 本文档
└── data/
    ├── The_final_data_after_screening.csv   # 主训练数据
    └── The_primary_data_of_this_research_.csv
```

---

## 生物医学同学：如何修改 DOM 权重

打开 `ra_pipeline.py`，找到顶部的 `DOM_FEATURE_WEIGHTS` 字典：

```python
DOM_FEATURE_WEIGHTS: dict[str, float] = {
    "BRI":           4.5,   # ← 修改这里的数值
    "SmokingStatus": 3.5,
    "Age":           3.0,
    ...
}
```

**权重规则：**
- `1.0` = 与 RA 相关性低
- `3.0` = 中等临床证据
- `5.0` = 强临床相关性（如：吸烟、BRI 等）

修改完数字后，重新运行 `python3 ra_pipeline.py` 即可，无需改动其他代码。

---

## 运行训练

```bash
# 使用默认数据
python3 ra_pipeline.py

# 使用自定义数据路径
python3 ra_pipeline.py /path/to/your_data.csv
```

---

## 接入新数据集

新数据只要包含相同的特征列（列名相同），直接传入即可：

```python
from ra_pipeline import run_pipeline, predict_new_patients

# 训练
predictor, models, evaluator = run_pipeline()

# 对新数据预测
results = predict_new_patients(predictor, "new_data.csv")
print(results[['RA_probability', 'RA_risk_level']])
```

如果新数据的**列名不同**，传入映射字典：

```python
results = predict_new_patients(
    predictor,
    "hospital_data.csv",
    column_map={
        "age_years": "Age",
        "sex":       "Gender",
        "bmi_val":   "BMI",
    }
)
```

---

## 训练集划分说明

| 集合 | 比例 | 用途 |
|------|------|------|
| 训练集 | 70% | 模型学习 |
| 验证集 | 15% | 调参、模型选择 |
| 测试集 | 15% | 最终评估（只跑一次） |

所有划分均使用**分层抽样**，保证每个子集中 RA 阳性比例一致（约 6.3%）。

---

## 当前运行结果（供参考）

```
模型                       AUROC      F1  Recall
LogisticRegression      0.7649   0.196   0.824
RandomForest            0.7639   0.209   0.752
GradientBoosting        0.7575   0.216   0.712
```

**阈值设为 0.35**（而非默认 0.5），目的是提高 Recall（减少漏诊），
代价是会有更多误报（FPR ~44%），这在医疗筛查场景中是合理权衡。

当专家补充更多生物标志物特征（如 DII 炎症指数、血细胞指标）后，预计 AUROC 可提升至 0.85+。

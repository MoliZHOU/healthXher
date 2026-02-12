"""
=============================================================================
  类风湿关节炎 (RA) 预测 Pipeline
  Rheumatoid Arthritis Prediction Pipeline
=============================================================================
  架构说明:
    1. config.py       — 特征定义 & DOM 专家权重配置（唯一需要手动修改的地方）
    2. data_loader.py  — 数据加载 & 新数据接入
    3. preprocessor.py — 特征工程 & 编码
    4. weighting.py    — DOM 权重 & NHANES 采样权重
    5. splitter.py     — 分层切分 (训练/验证/测试)
    6. models.py       — 模型定义 & 训练
    7. evaluator.py    — 评估指标
    8. pipeline.py     — 主入口 (本文件)
=============================================================================
"""

# ── 依赖 ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings, json, os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                              precision_score, classification_report,
                              confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

warnings.filterwarnings('ignore')
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════
#  SECTION 0 — CONFIG（专家修改区）
# ══════════════════════════════════════════════════════════════════════════
"""
DOM 权重说明
-----------
数值范围：1.0（低相关性）→ 5.0（强临床相关性）
由具有生物医学背景的领域专家（Domain Expert, DOM）填写。
其他同学**不需要修改**此处，只需更新下方字典的数值即可。
"""

DOM_FEATURE_WEIGHTS: dict[str, float] = {
    # ── 炎症 / 体脂分布指标（直接与RA病理相关）──────────────────────────
    "BRI":                    4.5,   # Body Roundness Index，内脏脂肪代理指标
    "BRI_Trend":              3.5,   # BRI 趋势，动态变化信息

    # ── 人口学（固定危险因素）────────────────────────────────────────────
    "Age":                    3.0,   # 年龄：RA高发于中老年
    "Gender":                 3.5,   # 性别：女性风险更高（约2-3倍）
    "Race":                   2.0,   # 种族：部分种族遗传易感性差异

    # ── 代谢 / 合并症──────────────────────────────────────────────────────
    "BMI":                    3.0,   # BMI：肥胖加重系统性炎症
    "Hypertension":           2.5,   # 高血压：自身免疫共病
    "Diabetes":               2.5,   # 糖尿病：代谢炎症重叠
    "Hyperlipidemia":         2.0,   # 血脂异常

    # ── 生活方式─────────────────────────────────────────────────────────
    "SmokingStatus":          3.5,   # 吸烟：RA最强可改变危险因素之一
    "PhysicalActivity":       2.0,   # 体力活动：保护性因素
    "DrinkingStatus":         1.5,   # 饮酒：证据较弱

    # ── 社会经济─────────────────────────────────────────────────────────
    "EducationLevel":         1.5,
    "MaritalStatus":          1.0,
    "FamilyIncome":           1.5,

    # ── 饮食摄入─────────────────────────────────────────────────────────
    "CalorieConsumption":     1.5,
    "ProteinConsumption":     2.0,   # 蛋白质：免疫功能底物
    "CarbohydrateConsumption":1.5,
    "FatConsumption":         1.5,
    "CaffeineConsumption":    1.0,
    "FiberConsumption":       2.0,   # 纤维：肠道菌群，影响免疫
}

# ── 模型超参（可按需调整）─────────────────────────────────────────────────
MODEL_CONFIG = {
    "test_size":        0.15,   # 测试集比例
    "val_size":         0.15,   # 验证集比例（从训练集再切）
    "random_state":     42,
    "cv_folds":         5,      # 交叉验证折数
    "decision_threshold": 0.35, # 决策阈值（<0.5 提高召回，减少漏诊）

    # 类别不平衡处理: "oversample" | "class_weight" | "none"
    "imbalance_strategy": "oversample",

    # 数据文件路径（相对于本脚本）
    "screened_data_path": "./data/The_final_data_after_screening.csv",
    "primary_data_path":  "./data/The_primary_data_of_this_research_.csv",
}


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADER
# ══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    加载数据集。支持两种输入:
      A) 已筛选数据 (final_screened) — 列名已经可读，直接使用
      B) 原始数据   (primary_raw)    — 需要列映射 + 标签解码

    新增数据集: 只需提供符合下方 COLUMN_MAP 格式的 CSV，
    调用 load_new_data(path) 即可无缝接入。
    """

    # 原始数据列名 → 标准列名映射
    COLUMN_MAP = {
        "MCQ160A":  "RheumatoidArthritis_raw",
        "DII":      "DII",
        "BMXWAIST": "WaistCircumference",
        "RIDAGEYR": "Age",
        "RIAGENDR": "Gender_raw",
        "RIDRETH1": "Race_raw",
        "DMDEDUC2": "EducationLevel_raw",
        "DMDMARTL": "MaritalStatus_raw",
        "INDFMPIR": "FamilyIncome_raw",
        "BPQ020":   "Hypertension_raw",
        "DIQ010":   "Diabetes_raw",
        "LBXTC":    "TotalCholesterol",
        "LBXTR":    "Triglycerides",
        "LBDLDL":   "LDL",
        "LBDHDD":   "HDL",
        "PAQ605":   "VigorousActivity_raw",
        "PAQ620":   "ModerateActivity_raw",
        "PAD680":   "SedentaryMinutes",
        "SMQ020":   "EverSmoked_raw",
        "SMQ040":   "CurrentSmoking_raw",
        "LBDNENO":  "Neutrophils",
        "LBDLYMNO": "Lymphocytes",
        "LBXPLTSI": "Platelets",
        "Drinking":  "DrinkingStatus_raw",
        "DR1TKCAL": "CalorieConsumption",
        "DR1TPROT": "ProteinConsumption",
        "DR1TCARB": "CarbohydrateConsumption",
        "DR1TTFAT": "FatConsumption",
        "DR1TCAFF": "CaffeineConsumption",
        "DR1TFIBE": "FiberConsumption",
    }

    @staticmethod
    def load_screened(path: str) -> pd.DataFrame:
        """加载已筛选数据（推荐主训练数据集）"""
        df = pd.read_csv(path)
        print(f"[DataLoader] 已筛选数据: {df.shape[0]} 行, {df.shape[1]} 列")
        print(f"  RA阳性: {df['RheumatoidArthritis'].sum()}  "
              f"RA阴性: {(df['RheumatoidArthritis']==0).sum()}")
        return df

    @staticmethod
    def load_primary(path: str) -> pd.DataFrame:
        """加载原始NHANES数据并做初步标签转换"""
        df = pd.read_csv(path, na_values=['NA', 'na', '', ' '])
        df = df.rename(columns=DataLoader.COLUMN_MAP)

        # 标签转换: 1→RA阳性=1, 2→阴性=0, 其他→丢弃
        valid_mask = df['RheumatoidArthritis_raw'].isin([1, 2])
        df = df[valid_mask].copy()
        df['RheumatoidArthritis'] = (df['RheumatoidArthritis_raw'] == 1).astype(int)

        print(f"[DataLoader] 原始数据(过滤后): {df.shape[0]} 行")
        print(f"  RA阳性: {df['RheumatoidArthritis'].sum()}  "
              f"RA阴性: {(df['RheumatoidArthritis']==0).sum()}")
        return df

    @staticmethod
    def load_new_data(path: str, label_col: str = None,
                      column_map: dict = None) -> pd.DataFrame:
        """
        接入新数据集的通用接口
        -------------------------------------------------------
        path        : CSV 文件路径
        label_col   : 标签列名（若已有则直接使用；若为 None 则视为待预测数据）
        column_map  : 列名映射字典（若列名与标准不同时传入）

        示例:
          new_df = DataLoader.load_new_data(
              "new_hospital_data.csv",
              label_col="RA_diagnosis",
              column_map={"age_years": "Age", "sex": "Gender"}
          )
        """
        df = pd.read_csv(path, na_values=['NA', 'na', '', ' '])
        if column_map:
            df = df.rename(columns=column_map)
        if label_col and label_col != 'RheumatoidArthritis':
            df = df.rename(columns={label_col: 'RheumatoidArthritis'})
        print(f"[DataLoader] 新数据: {df.shape[0]} 行, {df.shape[1]} 列")
        return df


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════

class Preprocessor:
    """
    特征工程:
      - 自动识别数值型 / 类别型特征
      - 类别编码 (Label Encoding)
      - 数值缺失值填充 (中位数)
      - 标准化 (StandardScaler，仅数值特征)
    """

    # 已筛选数据中的特征列（不含目标、ID、权重、调查辅助列）
    SCREENED_FEATURES = [
        "BRI", "BRI_Trend",
        "Gender", "Age", "Race", "EducationLevel", "MaritalStatus",
        "FamilyIncome", "PhysicalActivity", "SmokingStatus", "BMI",
        "DrinkingStatus", "Hypertension", "Diabetes", "Hyperlipidemia",
        "CalorieConsumption", "ProteinConsumption", "CarbohydrateConsumption",
        "FatConsumption", "CaffeineConsumption", "FiberConsumption",
    ]

    CATEGORICAL_FEATURES = [
        "Gender", "Race", "EducationLevel", "MaritalStatus",
        "FamilyIncome", "PhysicalActivity", "SmokingStatus",
        "DrinkingStatus", "Hypertension", "Diabetes", "Hyperlipidemia",
    ]

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.num_cols: list[str] = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame,
                      feature_cols: list[str] = None) -> pd.DataFrame:
        self.feature_cols = feature_cols or self.SCREENED_FEATURES
        # 只保留存在于 df 中的列
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]

        X = df[self.feature_cols].copy()

        # 类别编码
        self.cat_cols = [c for c in self.CATEGORICAL_FEATURES
                         if c in self.feature_cols]
        self.num_cols = [c for c in self.feature_cols
                         if c not in self.cat_cols]

        for col in self.cat_cols:
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown').astype(str)
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # 数值填充
        for col in self.num_cols:
            median = pd.to_numeric(X[col], errors='coerce').median()
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(median)

        # 标准化（数值列）
        X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])
        self.is_fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """用已拟合的编码器转换新数据（推理时使用）"""
        assert self.is_fitted, "请先调用 fit_transform"
        X = df[self.feature_cols].copy()

        for col in self.cat_cols:
            le = self.label_encoders[col]
            X[col] = X[col].fillna('Unknown').astype(str)
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v: v if v in known else 'Unknown')
            if 'Unknown' not in known:
                le.classes_ = np.append(le.classes_, 'Unknown')
            X[col] = le.transform(X[col])

        for col in self.num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(
                pd.to_numeric(X[col], errors='coerce').median())

        X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        return X


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DOM WEIGHTING
# ══════════════════════════════════════════════════════════════════════════

class DOMWeighter:
    """
    DOM（领域专家）权重处理器
    -------------------------------------------------------
    功能:
      1. 将 DOM_FEATURE_WEIGHTS 归一化为乘数向量
      2. 对特征矩阵按列加权（放大高权重特征的影响）
      3. 结合 NHANES 采样权重生成最终 sample_weight

    如何更新权重:
      只需修改文件顶部 DOM_FEATURE_WEIGHTS 字典的值，
      无需改动任何其他代码。
    """

    def __init__(self, feature_weights: dict[str, float] = None):
        self.raw_weights = feature_weights or DOM_FEATURE_WEIGHTS

    def get_feature_multipliers(self,
                                feature_cols: list[str]) -> np.ndarray:
        """
        返回与 feature_cols 对齐的权重向量（已归一化到均值=1）
        未在 DOM_FEATURE_WEIGHTS 中定义的特征默认权重 = 1.0
        """
        weights = np.array([
            self.raw_weights.get(col, 1.0) for col in feature_cols
        ])
        # 归一化：让均值保持为1，避免整体数值膨胀
        weights = weights / weights.mean()
        return weights

    def apply_to_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """对特征矩阵逐列乘以 DOM 权重"""
        multipliers = self.get_feature_multipliers(list(X.columns))
        X_weighted = X.copy()
        X_weighted = X_weighted * multipliers
        print(f"[DOMWeighter] 已对 {len(X.columns)} 个特征应用专家权重")
        self._print_top_weights(list(X.columns))
        return X_weighted

    def compute_sample_weights(self,
                               nhanes_weights: pd.Series,
                               normalize: bool = True) -> np.ndarray:
        """
        将 NHANES 调查权重转换为模型可用的 sample_weight。
        归一化到 [0, 1] 以避免数值过大影响训练稳定性。
        """
        w = nhanes_weights.values.astype(float)
        w = np.where(np.isnan(w) | (w <= 0), w[w > 0].min(), w)
        if normalize:
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            w = w + 0.1   # 最小值不为0，避免某些样本完全无权重
        return w

    def _print_top_weights(self, feature_cols: list[str], top_n: int = 8):
        weights = {col: self.raw_weights.get(col, 1.0) for col in feature_cols}
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        print("  Top特征权重:")
        for col, w in sorted_w[:top_n]:
            bar = "█" * int(w)
            print(f"    {col:<30} {bar} ({w:.1f})")


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SPLITTER
# ══════════════════════════════════════════════════════════════════════════

class DataSplitter:
    """
    分层数据集划分
    -------------------------------------------------------
    策略: 先切出测试集（永久封存），再从剩余中切出验证集。
    全程保持 RA 正负样本比例一致（stratify=y）。

    返回:
      train_idx, val_idx, test_idx  （DataFrame 行索引）
    """

    def __init__(self, test_size: float = 0.15,
                 val_size: float = 0.15,
                 random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame,
              label_col: str = 'RheumatoidArthritis'):
        from sklearn.model_selection import train_test_split

        y = df[label_col].values
        idx = np.arange(len(df))

        # Step 1: 切出测试集
        idx_trainval, idx_test = train_test_split(
            idx, test_size=self.test_size,
            stratify=y, random_state=self.random_state)

        # Step 2: 从 trainval 中切出验证集
        val_ratio = self.val_size / (1 - self.test_size)
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=val_ratio,
            stratify=y[idx_trainval],
            random_state=self.random_state)

        self._report(y, idx_train, idx_val, idx_test)
        return idx_train, idx_val, idx_test

    def _report(self, y, idx_train, idx_val, idx_test):
        total = len(y)
        print("\n[DataSplitter] 数据集划分完成")
        print(f"{'集合':<8} {'样本数':>8} {'比例':>7} {'RA阳性':>8} {'阳性率':>8}")
        print("-" * 44)
        for name, idx in [("训练集", idx_train),
                          ("验证集", idx_val),
                          ("测试集", idx_test)]:
            n = len(idx)
            pos = y[idx].sum()
            print(f"{name:<8} {n:>8} {n/total:>7.1%} {pos:>8} {pos/n:>8.1%}")
        print()


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — IMBALANCE HANDLER (替代 SMOTE)
# ══════════════════════════════════════════════════════════════════════════

def handle_imbalance(X: np.ndarray, y: np.ndarray,
                     sample_weights: np.ndarray,
                     strategy: str = "oversample") -> tuple:
    """
    类别不平衡处理（在训练集上应用，测试集不处理）

    strategy:
      "oversample"   — 少数类随机过采样（带权重复制）
      "class_weight" — 不修改数据，返回调整后的权重（供模型使用）
      "none"         — 不做处理
    """
    if strategy == "none":
        return X, y, sample_weights

    pos_mask = (y == 1)
    neg_mask = (y == 0)
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if strategy == "oversample":
        # 将少数类（RA阳性）过采样到与多数类相同数量
        X_pos = X[pos_mask];  y_pos = y[pos_mask];  w_pos = sample_weights[pos_mask]
        X_neg = X[neg_mask];  y_neg = y[neg_mask];  w_neg = sample_weights[neg_mask]

        X_pos_rs, y_pos_rs, w_pos_rs = resample(
            X_pos, y_pos, w_pos,
            replace=True, n_samples=n_neg, random_state=42)

        X_out = np.vstack([X_neg, X_pos_rs])
        y_out = np.concatenate([y_neg, y_pos_rs])
        w_out = np.concatenate([w_neg, w_pos_rs])
        print(f"[ImbalanceHandler] oversample: {n_pos}→{n_neg} 阳性样本")
        return X_out, y_out, w_out

    elif strategy == "class_weight":
        # 增加少数类样本权重
        ratio = n_neg / n_pos
        adjusted = sample_weights.copy()
        adjusted[pos_mask] *= ratio
        print(f"[ImbalanceHandler] class_weight: 阳性权重 ×{ratio:.1f}")
        return X, y, adjusted

    return X, y, sample_weights


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MODELS
# ══════════════════════════════════════════════════════════════════════════

def build_models(class_weight_mode: bool = False) -> dict:
    """
    返回待评估的模型字典
    如需添加新模型: 直接在此字典中增加一项即可
    """
    cw = 'balanced' if class_weight_mode else None
    return {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000,
            class_weight=cw, random_state=42),

        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_leaf=5,
            class_weight=cw, random_state=42, n_jobs=-1),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            random_state=42),
    }


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 7 — EVALUATOR
# ══════════════════════════════════════════════════════════════════════════

class Evaluator:
    """评估模型性能，重点关注医疗场景的 Recall（减少漏诊）"""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self.results: dict = {}

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray,
                 model_name: str = "Model") -> dict:
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= self.threshold).astype(int)

        metrics = {
            "AUROC":     round(roc_auc_score(y_test, proba), 4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        }
        self.results[model_name] = metrics

        print(f"\n[Evaluator] {model_name} (阈值={self.threshold})")
        print(f"  AUROC={metrics['AUROC']:.4f}  F1={metrics['F1']:.4f}  "
              f"Recall={metrics['Recall']:.4f}  Precision={metrics['Precision']:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"  混淆矩阵: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"  漏诊率(FNR)={fn/(fn+tp):.2%}  误诊率(FPR)={fp/(fp+tn):.2%}")
        return metrics

    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                              model_name: str = "Model", cv: int = 5):
        """K折交叉验证（在训练集上执行）"""
        cv_results = cross_validate(
            model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True,
                                             random_state=42),
            scoring=['roc_auc', 'f1', 'recall'],
            return_train_score=False)
        print(f"\n[CV] {model_name} {cv}折交叉验证:")
        print(f"  AUROC={cv_results['test_roc_auc'].mean():.4f}±"
              f"{cv_results['test_roc_auc'].std():.4f}")
        print(f"  F1   ={cv_results['test_f1'].mean():.4f}±"
              f"{cv_results['test_f1'].std():.4f}")
        print(f"  Recall={cv_results['test_recall'].mean():.4f}±"
              f"{cv_results['test_recall'].std():.4f}")

    def summary(self):
        """打印所有模型对比表"""
        print("\n" + "="*60)
        print("  模型性能汇总")
        print("="*60)
        print(f"{'模型':<22} {'AUROC':>7} {'F1':>7} {'Recall':>7} {'Precision':>10}")
        print("-"*60)
        for name, m in self.results.items():
            print(f"{name:<22} {m['AUROC']:>7} {m['F1']:>7} "
                  f"{m['Recall']:>7} {m['Precision']:>10}")
        print()
        best = max(self.results, key=lambda k: self.results[k]['AUROC'])
        print(f"  ★ 推荐模型（最高AUROC）: {best}")
        return best


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PREDICTOR（推理接口）
# ══════════════════════════════════════════════════════════════════════════

class RAPredictor:
    """
    训练完成后的推理接口
    -------------------------------------------------------
    用法:
      predictor = RAPredictor(model, preprocessor, weighter, threshold)
      result = predictor.predict(new_df)
    """

    def __init__(self, model, preprocessor: Preprocessor,
                 weighter: DOMWeighter, threshold: float = 0.35):
        self.model = model
        self.preprocessor = preprocessor
        self.weighter = weighter
        self.threshold = threshold

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输入: 包含特征列的 DataFrame（格式与训练数据相同）
        输出: 原始 DataFrame + RA_probability + RA_prediction 两列
        """
        X = self.preprocessor.transform(df)
        X_w = self.weighter.apply_to_features(X)
        proba = self.model.predict_proba(X_w.values)[:, 1]
        pred = (proba >= self.threshold).astype(int)

        result = df.copy()
        result['RA_probability'] = np.round(proba, 4)
        result['RA_prediction']  = pred
        result['RA_risk_level']  = pd.cut(
            proba,
            bins=[0, 0.2, 0.4, 0.6, 1.0],
            labels=['低风险', '中等风险', '高风险', '极高风险'])
        return result


# ══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_pipeline(data_path: str = None, save_model: bool = False):
    """
    完整训练流程入口
    -------------------------------------------------------
    data_path: 可传入新的 CSV 路径（符合已筛选数据格式），
               None 则使用 MODEL_CONFIG 中的默认路径
    """
    print("\n" + "★"*60)
    print("  RA 预测 Pipeline 启动")
    print("★"*60 + "\n")

    # ── 1. 加载数据 ───────────────────────────────────────────────────────
    path = data_path or MODEL_CONFIG["screened_data_path"]
    # 支持直接传绝对路径
    if not Path(path).exists():
        # 尝试在脚本目录下找
        alt = Path(__file__).parent / path
        if alt.exists():
            path = str(alt)

    df = DataLoader.load_screened(path)

    # ── 2. 预处理 ─────────────────────────────────────────────────────────
    preprocessor = Preprocessor()
    X_processed = preprocessor.fit_transform(df)

    # ── 3. DOM 加权 ───────────────────────────────────────────────────────
    weighter = DOMWeighter(DOM_FEATURE_WEIGHTS)
    X_weighted = weighter.apply_to_features(X_processed)

    # NHANES 采样权重
    nhanes_w = df['Weight'] if 'Weight' in df.columns else pd.Series(
        np.ones(len(df)))
    sample_weights = weighter.compute_sample_weights(nhanes_w)

    # ── 4. 数据集划分 ─────────────────────────────────────────────────────
    splitter = DataSplitter(
        test_size=MODEL_CONFIG["test_size"],
        val_size=MODEL_CONFIG["val_size"],
        random_state=MODEL_CONFIG["random_state"])

    y = df['RheumatoidArthritis'].values
    idx_train, idx_val, idx_test = splitter.split(df)

    X_arr = X_weighted.values
    X_train_raw = X_arr[idx_train]; y_train = y[idx_train]
    X_val        = X_arr[idx_val];   y_val   = y[idx_val]
    X_test        = X_arr[idx_test];  y_test  = y[idx_test]
    w_train      = sample_weights[idx_train]

    # ── 5. 不平衡处理（仅训练集）──────────────────────────────────────────
    X_train, y_train_bal, w_train_bal = handle_imbalance(
        X_train_raw, y_train, w_train,
        strategy=MODEL_CONFIG["imbalance_strategy"])

    # ── 6. 模型训练 & 交叉验证 ────────────────────────────────────────────
    models = build_models()
    evaluator = Evaluator(threshold=MODEL_CONFIG["decision_threshold"])
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  训练: {name}")

        # 交叉验证（在原始训练集上，不过采样）
        evaluator.cross_validate_model(
            model, X_train_raw, y_train,
            model_name=name, cv=MODEL_CONFIG["cv_folds"])

        # 正式训练（用过采样后的训练集 + 样本权重）
        fit_kwargs = {}
        if hasattr(model, 'fit'):
            import inspect
            sig = inspect.signature(model.fit)
            if 'sample_weight' in sig.parameters:
                fit_kwargs['sample_weight'] = w_train_bal

        model.fit(X_train, y_train_bal, **fit_kwargs)
        trained_models[name] = model

        # 验证集评估
        evaluator.evaluate(model, X_val, y_val, model_name=f"{name}[Val]")

    # ── 7. 最终测试集评估 ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  测试集最终评估（只运行一次）")
    print("="*60)

    final_evaluator = Evaluator(threshold=MODEL_CONFIG["decision_threshold"])
    for name, model in trained_models.items():
        final_evaluator.evaluate(model, X_test, y_test, model_name=name)

    best_name = final_evaluator.summary()
    best_model = trained_models[best_name]

    # ── 8. 特征重要性（RandomForest / GradientBoosting）──────────────────
    for name in ["RandomForest", "GradientBoosting"]:
        if name in trained_models:
            _print_feature_importance(
                trained_models[name], preprocessor.feature_cols,
                weighter, model_name=name)
            break

    # ── 9. 打包推理接口 ───────────────────────────────────────────────────
    predictor = RAPredictor(
        model=best_model,
        preprocessor=preprocessor,
        weighter=weighter,
        threshold=MODEL_CONFIG["decision_threshold"])

    print(f"\n[Pipeline] 完成！推荐模型: {best_name}")
    print("  使用方式: predictor.predict(new_dataframe)")
    return predictor, trained_models, final_evaluator


def _print_feature_importance(model, feature_cols: list,
                               weighter: DOMWeighter,
                               model_name: str = "Model", top_n: int = 12):
    """打印特征重要性与 DOM 权重对比"""
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    dom_w = weighter.get_feature_multipliers(feature_cols)

    fi_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': importances,
        'dom_weight': [weighter.raw_weights.get(c, 1.0) for c in feature_cols],
    }).sort_values('importance', ascending=False)

    print(f"\n[FeatureImportance] {model_name} Top-{top_n}")
    print(f"{'特征':<30} {'模型重要性':>10} {'DOM权重':>10}")
    print("-"*54)
    for _, row in fi_df.head(top_n).iterrows():
        bar = "▓" * int(row['importance'] * 200)
        print(f"{row['feature']:<30} {row['importance']:>10.4f} "
              f"{row['dom_weight']:>10.1f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════
#  便捷函数：接入新数据
# ══════════════════════════════════════════════════════════════════════════

def predict_new_patients(predictor: RAPredictor, csv_path: str,
                         column_map: dict = None) -> pd.DataFrame:
    """
    对新患者数据进行 RA 风险预测
    -------------------------------------------------------
    csv_path   : 新数据 CSV 路径
    column_map : 若列名与标准不同，传入映射字典

    示例:
      results = predict_new_patients(predictor, "hospital_q1_2025.csv")
      print(results[['SEQN','RA_probability','RA_risk_level']])
    """
    new_df = DataLoader.load_new_data(csv_path, column_map=column_map)
    result = predictor.predict(new_df)
    print(f"\n[Predict] 完成预测 {len(result)} 条记录")
    print(result[['RA_probability', 'RA_prediction', 'RA_risk_level']]
          .value_counts('RA_risk_level').to_string())
    return result


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # 接受可选的数据路径参数
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    predictor, models, evaluator = run_pipeline(data_path)

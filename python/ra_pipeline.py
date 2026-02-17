"""
╔══════════════════════════════════════════════════════════════════════════╗
║         Rheumatoid Arthritis (RA) Prediction Pipeline v2.0               ║
║         类风湿关节炎 (RA) 预测流水线 v2.0                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Features (v2.0):                                                        ║
║  ① StratifiedGroupKFold - PSU-based splitting to prevent data leakage     ║
║  ② Constraints - Monotonic & Interaction constraints for BioMatic logic   ║
║  ③ Derived Features - NLR, DII, and other inflammatory markers            ║
║  ④ Weight Merging - NHANES multi-cycle weight adjustment                  ║
║  ⑤ Probability Calibration - CalibratedClassifierCV integration           ║
║  ⑥ Nested CV - Reliable performance estimation with Bayesian search       ║
║  ⑦ Evaluation - AUROC, AP, Brier Score, and Confusion Matrix              ║
╚══════════════════════════════════════════════════════════════════════════╝

  ┌────────────────────────────────────────────────────────────────────────┐
  │  Expert Configuration Area (BioMatic Logic):                             │
  │  1. DOM_FEATURE_WEIGHTS    - Adjust clinical importance (1.0~5.0)      │
  │  2. MONOTONIC_CONSTRAINTS  - Define correlation directions              │
  │  3. INTERACTION_GROUPS     - Define allowed feature interactions       │
  └────────────────────────────────────────────────────────────────────────┘
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_predict)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, f1_score, recall_score,
                              precision_score, confusion_matrix,
                              classification_report)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

# Silence warnings for cleaner output
# 忽略警告以保持输出整洁
warnings.filterwarnings('ignore')
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════
#  ★ SECTION 0 — EXPERT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# ── 0-A  DOM Feature Weights / 特征权重 ────────────────────────────────────
# Range: 1.0 (Low) → 5.0 (Strong Clinical Relevance)
# Based on: Literature evidence + Pathophysiological mechanisms
# 范围：1.0（低）→ 5.0（强临床相关性）；依据：文献证据 + 病理生理机制
DOM_FEATURE_WEIGHTS: dict[str, float] = {
    # Core Inflammatory Markers / 核心炎症指标
    "BRI":                    4.5,   # Visceral adiposity proxy / 内脏脂肪代理指标
    "BRI_Trend":              3.5,   # Dynamic BRI trend / BRI 动态趋势
    "NLR":                    4.0,   # Neutrophil-to-Lymphocyte Ratio / 中性粒/淋巴比
    "DII":                    4.5,   # Dietary Inflammatory Index / 饮食炎症指数

    # Demographic Risk Factors / 人口学危险因素
    "Age":                    3.5,   # RA peak onset 40-60 / 40-60岁高发
    "Gender":                 3.5,   # Female risk is 2-3x higher / 女性风险更高

    # Metabolic & Comorbidities / 代谢与合并症
    "BMI":                    3.0,
    "Hypertension":           2.5,
    "Diabetes":               2.5,
    "Hyperlipidemia":         2.0,

    # Modifiable Risk Factors / 可改变危险因素
    "SmokingStatus":          4.0,   # Strongest modifiable factor / 最强可改变因素
  
    # Lifestyle / 生活方式
    "PhysicalActivity":       2.0,
    "DrinkingStatus":         1.5,
    "FiberConsumption":       2.5,   # Microbiome-immune axis / 肠道菌群-免疫轴

    # Socio-economic / 社会经济因素
    "Race":                   0,
    "EducationLevel":         0,
    "MaritalStatus":          0,
    "FamilyIncome":           0,

    # Dietary Intake / 饮食摄入
    "CalorieConsumption":     1.5,
    "ProteinConsumption":     2.0,
    "CarbohydrateConsumption":1.5,
    "FatConsumption":         1.5,
    "CaffeineConsumption":    1.0,

    # ADDED
    "Family History":         3.5,
    "Postpartum_12m":         5.0,
    "MenopauseStatus":         4.5,
    "MorningStiffnessLong":      4.5,
    "SymptomsDuration6Weeks":    4.0,
    "SmallJointSymmetry":        5.0
  
}

# ── 0-B  Monotonic Constraints / 单调性约束 ───────────────────────────────
# 1 (Positive), -1 (Negative), 0 (None)
# Prevents model from learning non-biological noise
# 1 (正相关), -1 (负相关), 0 (无约束)；防止模型拟合非生物学噪声
MONOTONIC_CONSTRAINTS: dict[str, int] = {
    "Age":              1,   # Age increases risk / 年龄增加风险
    "BRI":              1,   # Adiposity increases inflammation / 肥胖增加炎症
    "BRI_Trend":        1,   # Deteriorating trend increases risk / 恶化趋势增加风险
    "DII":              1,   # Higher DII = more pro-inflammatory / DII越高越致炎
    "NLR":              1,   # Higher NLR = systemic inflammation / NLR越高代表系统性炎症
    "BMI":              1,   # BMI increases risk / BMI增加风险
    "FiberConsumption": -1,  # Fiber is protective / 膳食纤维具有保护作用
    "PhysicalActivity": 0,   # Non-linear relationship / 非线性关系
    # ______ADDED______
    "SmokingStatus":          1,   
    "FamilyHistory":          1,
    "SmallJointSymmetry":     1,   
    "MorningStiffnessLong":   1,   
    "SymptomsDuration6Weeks": 1,
    "Postpartum_12m":         1,   
    "MenopauseStatus":        1,
}

# ── 0-C  Interaction Constraints / 交互约束 ────────────────────────────────
# Defines feature groups allowed to interact within a single tree
# 定义允许在同一棵树中交互的特征组
# INTERACTION_GROUPS: list[list[str]] = [
#     # Group 1: Inflammatory Biomarkers / 炎症生物标志物
#     ["BRI", "BRI_Trend", "NLR", "DII", "BMI"],
#     # Group 2: Demographics & Social / 人口学与社会因素
#     ["Age", "Gender", "Race", "EducationLevel", "FamilyIncome"],
#     # Group 3: Lifestyle / 生活方式
#     ["SmokingStatus", "PhysicalActivity", "DrinkingStatus",
#      "FiberConsumption", "CalorieConsumption"],
#     # Group 4: Comorbidities / 合并症
#     ["Hypertension", "Diabetes", "Hyperlipidemia"],
#     # Cross-group interaction (Age x Gender x Smoking) / 跨组交互 (年龄 x 性别 x 吸烟)
#     ["SmokingStatus", "Age", "Gender"],
# ]

INTERACTION_GROUPS: list[list[str]] = [
    # Group 1: Inflammatory Profile (代谢与炎症指标的关联)
    ["BRI", "BRI_Trend", "NLR", "DII", "BMI"],

    # Group 2: The "Hormonal Trigger" (最关键：生命周期与性别)
    ["Postpartum_12m", "MenopauseStatus", "Gender", "Age"],

    # Group 3: Clinical RA Signature (诊断核心：对称性、时长与晨僵的协同)
    ["SmallJointSymmetry", "MorningStiffnessLong", "SymptomsDuration6Weeks"],

    # Group 4: Risk Intensifiers (外部诱因与遗传背景)
    ["SmokingStatus", "FamilyHistory"],

    # Cross-group Interaction: Hormonal Stage x Clinical Symptoms
    # (论文核心：当处于产后期时，关节症状的权重会因激素波动而产生非线性增长)
    ["Postpartum_12m", "SmallJointSymmetry", "MorningStiffnessLong"],
]

# ── 0-D  Global Configuration / 全局配置 ──────────────────────────────────
CONFIG = {
    "test_size":          0.15,
    "n_nhanes_cycles":    2,      # Number of NHANES cycles / NHANES 周期数
    "cv_outer_folds":     5,      # Outer CV folds / 外层交叉验证折数
    "cv_inner_folds":     3,      # Inner CV folds / 内层超参搜索折数
    "n_hyperopt_iter":    30,     # Search iterations / 超参搜索次数
    "decision_threshold": 0.15,   # Calibrated threshold / 校准后决策阈值
    "imbalance_strategy": "reweight",  # Strategy: reweight or oversample / 不平衡处理策略
    "screened_data_path": "../data/The_final_data_after_screening.csv",
    "random_state":       42,
}


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADER & LABELING
#               数据加载与标签处理
# ══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Handles NHANES data loading and weight merging.
    处理 NHANES 数据加载与权重合并。
    """

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        print(f"[DataLoader] Loading: {Path(path).name}")
        print(f"  Samples: {df.shape[0]} | Features: {df.shape[1]}")
        pos = int(df['RheumatoidArthritis'].sum())
        neg = int((df['RheumatoidArthritis'] == 0).sum())
        ratio = neg / max(pos, 1)
        print(f"  RA Positive: {pos} | RA Negative: {neg} | Ratio: {ratio:.1f}:1")
        return df

    @staticmethod
    def merge_nhanes_weights(df: pd.DataFrame,
                             weight_col: str = 'Weight',
                             n_cycles: int = 2) -> pd.Series:
        """
        NHANES Multi-cycle weight adjustment (CDC Specification).
        NHANES 多周期权重调整 (CDC 官方标准)。
        """
        w = df[weight_col].copy().astype(float)
        w_combined = w / n_cycles
        print(f"[Weights] NHANES weights merged (divided by {n_cycles}): "
              f"Mean {w.mean():.0f} -> {w_combined.mean():.0f}")
        return w_combined

    @staticmethod
    def load_new(path: str, label_col: str = 'RheumatoidArthritis',
                 col_map: dict = None) -> pd.DataFrame:
        """
        General interface for external datasets.
        通用外部数据集加载接口。
        """
        df = pd.read_csv(path, na_values=['NA', 'na', ''])
        if col_map:
            df = df.rename(columns=col_map)
        if label_col != 'RheumatoidArthritis' and label_col in df.columns:
            df = df.rename(columns={label_col: 'RheumatoidArthritis'})
        print(f"[DataLoader] New data {Path(path).name}: {df.shape[0]} rows x {df.shape[1]} cols")
        return df


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — FEATURE ENGINEERING
#               特征工程
# ══════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Handles derived features, encoding, and scaling.
    处理衍生特征、编码与标准化。
    """

    CATEGORICAL = [
        "Gender", "Race", "EducationLevel", "MaritalStatus",
        "FamilyIncome", "PhysicalActivity", "SmokingStatus",
        "DrinkingStatus", "Hypertension", "Diabetes", "Hyperlipidemia",
    ]

    BASE_FEATURES = [
        "BRI", "BRI_Trend", "Age", "BMI",
        "CalorieConsumption", "ProteinConsumption",
        "CarbohydrateConsumption", "FatConsumption",
        "CaffeineConsumption", "FiberConsumption",
    ] + CATEGORICAL

    def __init__(self):
        self.encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.num_cols: list[str] = []
        self.cat_cols_in_model: list[str] = []
        self._fitted = False

    # ── Derived Features / 衍生特征 ────────────────────────────────────────

    @staticmethod
    def derive_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates NLR and processes DII. / 计算 NLR 并处理 DII。"""
        out = df.copy()

        # NLR = Neutrophil / Lymphocyte (Systemic Inflammation)
        # NLR = 中性粒细胞 / 淋巴细胞 (系统性炎症指标)
        if 'Neutrophils' in df.columns and 'Lymphocytes' in df.columns:
            n = pd.to_numeric(df['Neutrophils'], errors='coerce')
            l = pd.to_numeric(df['Lymphocytes'], errors='coerce')
            out['NLR'] = np.where(l > 0, n / l, np.nan)
            print("  [Derived] NLR (Neutrophil/Lymphocyte) calculated.")

        # DII processed directly / 直接处理 DII
        if 'DII' in df.columns:
            out['DII'] = pd.to_numeric(df['DII'], errors='coerce')
            print("  [Derived] DII Inflammatory Index processed.")

        return out

    # ── Encoding & Scaling / 编码与标准化 ──────────────────────────────────

    def fit_transform(self, df: pd.DataFrame,
                      extra_features: list[str] = None) -> pd.DataFrame:
        """Fits the engineer and transforms the data. / 拟合并转换数据。"""
        df = self.derive_features(df)
        candidate = (self.BASE_FEATURES
                     + (extra_features or [])
                     + [c for c in ['NLR', 'DII'] if c in df.columns])
        self.feature_cols = [c for c in dict.fromkeys(candidate)  # Deduplicate while preserving order
                             if c in df.columns]

        X = df[self.feature_cols].copy()
        self.cat_cols_in_model = [c for c in self.CATEGORICAL
                                  if c in self.feature_cols]
        self.num_cols = [c for c in self.feature_cols
                         if c not in self.cat_cols_in_model]

        for col in self.cat_cols_in_model:
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown').astype(str)
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le

        for col in self.num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())

        X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])
        self._fitted = True

        print(f"[FeatureEngineer] Total features in model: {len(self.feature_cols)}")
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms new data using fitted params. / 使用已拟合的参数转换新数据。"""
        assert self._fitted
        df = self.derive_features(df)
        X = df[[c for c in self.feature_cols if c in df.columns]].copy()

        # Supplement missing columns / 补充缺失列
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0

        for col in self.cat_cols_in_model:
            le = self.encoders[col]
            X[col] = X[col].fillna('Unknown').astype(str)
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(X[col])

        for col in self.num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

        X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        return X[self.feature_cols]


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DOM WEIGHTER
#               领域专家加权器
# ══════════════════════════════════════════════════════════════════════════

class DOMWeighter:
    """
    Applies domain expert weights and constraints.
    应用领域专家权重与约束。
    """

    def __init__(self, weights: dict = None,
                 monotonic: dict = None,
                 interactions: list = None):
        self.weights = weights or DOM_FEATURE_WEIGHTS
        self.monotonic = monotonic or MONOTONIC_CONSTRAINTS
        self.interactions = interactions or INTERACTION_GROUPS

    def get_multipliers(self, feature_cols: list[str]) -> np.ndarray:
        """Returns normalized weight vector. / 返回归一化的权重向量。"""
        w = np.array([self.weights.get(c, 1.0) for c in feature_cols])
        return w / w.mean()

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Multiplies feature columns by expert weights. / 将特征列乘以专家权重。"""
        m = self.get_multipliers(list(X.columns))
        out = X * m
        self._show(list(X.columns))
        return out

    def make_monotonic_cst(self, feature_cols: list[str]) -> list[int]:
        """Generates monotonic constraint vector. / 生成单调性约束向量。"""
        cst = [self.monotonic.get(c, 0) for c in feature_cols]
        active = [(c, v) for c, v in zip(feature_cols, cst) if v != 0]
        if active:
            print(f"[DOMWeighter] Monotonic Constraints applied ({len(active)} features):")
            for c, v in active:
                arrow = "↑(+)" if v == 1 else "↓(-)"
                print(f"    {c:<28} {arrow}")
        return cst

    def make_interaction_cst(self, feature_cols: list[str]):
        """Generates interaction constraint indices. / 生成交互约束索引。"""
        idx_map = {c: i for i, c in enumerate(feature_cols)}
        result = []
        for group in self.interactions:
            group_idx = [idx_map[c] for c in group if c in idx_map]
            if len(group_idx) >= 2:
                result.append(group_idx)
        if result:
            print(f"[DOMWeighter] Interaction Groups: {len(result)}")
        return result if result else None

    def compute_sample_weights(self,
                               nhanes_w: pd.Series,
                               y: np.ndarray,
                               strategy: str = "reweight") -> np.ndarray:
        """Combines NHANES weights with imbalance correction. / 结合 NHANES 权重与不平衡校正。"""
        # Normalize NHANES weights / 归一化 NHANES 权重
        w = nhanes_w.values.astype(float)
        w = np.where(np.isnan(w) | (w <= 0), np.nanmedian(w), w)
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8) + 0.1

        # Class imbalance correction / 类别不平衡校正
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        imbalance_ratio = n_neg / max(n_pos, 1)
        class_w = np.where(y == 1, imbalance_ratio, 1.0)

        combined = w_norm * class_w
        print(f"[DOMWeighter] Sample weights computed (Positive Weight x{imbalance_ratio:.1f})")
        return combined

    def _show(self, feature_cols, top=8):
        ranked = sorted(feature_cols,
                        key=lambda c: self.weights.get(c, 1.0), reverse=True)
        print("[DOMWeighter] Top feature clinical weights:")
        for c in ranked[:top]:
            w = self.weights.get(c, 1.0)
            bar = "█" * int(w)
            print(f"    {c:<28} {bar} {w:.1f}")


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SPLITTER (Anti-leakage)
#               数据集切分器 (防泄露)
# ══════════════════════════════════════════════════════════════════════════

class DataSplitter:
    """
    Splits data based on PSU (Primary Sampling Unit) to prevent leakage.
    基于 PSU (初级抽样单元) 切分数据以防止泄露。
    """

    def __init__(self, test_size: float = 0.15,
                 outer_folds: int = 5,
                 random_state: int = 42):
        self.test_size = test_size
        self.outer_folds = outer_folds
        self.rs = random_state

    def split(self, df: pd.DataFrame, y: np.ndarray):
        has_psu = 'PSU' in df.columns and 'STRATA' in df.columns
        groups = df['PSU'].values if has_psu else None

        if has_psu:
            print("[Splitter] Using StratifiedGroupKFold (PSU-based, anti-leakage)")
            splitter = StratifiedGroupKFold(
                n_splits=self.outer_folds, shuffle=True,
                random_state=self.rs)
            splits = list(splitter.split(df, y, groups))
            train_val_idx, test_idx = splits[0]
        else:
            print("[Splitter] PSU missing, falling back to StratifiedKFold")
            sk = StratifiedKFold(n_splits=self.outer_folds,
                                 shuffle=True, random_state=self.rs)
            splits = list(sk.split(df, y))
            train_val_idx, test_idx = splits[0]
            groups = None

        # Validation set (approx 15% of total) / 验证集 (约占总数 15%)
        val_n = int(len(train_val_idx) * 0.18)
        rng = np.random.default_rng(self.rs)
        rng.shuffle(train_val_idx)
        val_idx = train_val_idx[:val_n]
        train_idx = train_val_idx[val_n:]

        self._report(y, train_idx, val_idx, test_idx, psu_used=has_psu)
        return train_idx, val_idx, test_idx, groups

    def _report(self, y, tr, va, te, psu_used):
        total = len(y)
        print(f"\n{'Split':<12} {'Samples':>10} {'Percentage':>12} {'RA+':>8} {'Rate':>8}")
        print("-" * 55)
        for name, idx in [("Train", tr), ("Validation", va), ("Test", te)]:
            n = len(idx); pos = y[idx].sum()
            print(f"{name:<12} {n:>10} {n/total:>12.1%} {pos:>8} {pos/n:>8.1%}")
        psu_tag = "✓ PSU Groups" if psu_used else "△ Stratified Only"
        print(f"  Splitting Strategy: {psu_tag}\n")


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — EXPERT-AUGMENTED GBDT MODEL
#               专家增强型 GBDT 模型
# ══════════════════════════════════════════════════════════════════════════

class ExpertAugmentedGBDT:
    """
    Expert-augmented HistGradientBoosting with calibration.
    专家增强型直方图梯度提升机 (带校准)。
    """

    def __init__(self, feature_cols: list[str],
                 weighter: DOMWeighter):
        self.feature_cols = feature_cols
        self.weighter = weighter

        # Constraints / 约束
        mono_cst = weighter.make_monotonic_cst(feature_cols)
        inter_cst = weighter.make_interaction_cst(feature_cols)

        self._base = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            l2_regularization=0.5,
            max_bins=63,
            monotonic_cst=mono_cst,
            interaction_cst=inter_cst,
            class_weight='balanced',
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )

        # Isotonic calibration / 保序回归校准
        self.model = CalibratedClassifierCV(
            self._base, method='isotonic', cv=3)

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray = None):
        """Fits base and then calibrates. / 拟合基模型后进行校准。"""
        if sample_weight is not None:
            self._base.fit(X, y)
        else:
            self._base.fit(X, y)
        self.model = CalibratedClassifierCV(
            self._base, method='isotonic', cv=None)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray,
                threshold: float = 0.30) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def build_all_models(feature_cols: list[str],
                     weighter: DOMWeighter) -> dict:
    """Returns a dictionary of candidate models. / 返回候选模型字典。"""
    return {
        "ExpertGBDT": ExpertAugmentedGBDT(feature_cols, weighter),

        "RandomForest": CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300, max_depth=8,
                min_samples_leaf=8,
                class_weight='balanced',
                random_state=42, n_jobs=-1),
            method='sigmoid', cv=3),

        "LogisticRegression": CalibratedClassifierCV(
            LogisticRegression(
                C=0.5, max_iter=2000,
                class_weight='balanced', random_state=42),
            method='sigmoid', cv=3),
    }


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 6 — HYPERPARAMETER SEARCH
#               超参数搜索
# ══════════════════════════════════════════════════════════════════════════

def hyperparam_search(X_train: np.ndarray, y_train: np.ndarray,
                      n_iter: int = 20) -> dict:
    """Performs randomized search for GBDT. / 对 GBDT 执行随机超参搜索。"""
    print("\n[HyperSearch] Running randomized search...")
    param_dist = {
        'max_iter':        [200, 300, 400, 500],
        'learning_rate':   [0.01, 0.03, 0.05, 0.08, 0.1],
        'max_depth':       [3, 4, 5, 6],
        'min_samples_leaf':[10, 20, 30, 50],
        'l2_regularization':[0.0, 0.1, 0.5, 1.0],
    }
    base = HistGradientBoostingClassifier(
        class_weight='balanced', random_state=42)
    search = RandomizedSearchCV(
        base, param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        n_jobs=-1, random_state=42, verbose=0)
    search.fit(X_train, y_train)
    print(f"  Best AUROC (CV): {search.best_score_:.4f}")
    print(f"  Best Params: {search.best_params_}")
    return search.best_params_


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 7 — EVALUATOR
#               评估器
# ══════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Comprehensive model assessment metrics.
    全方位的模型评估指标。
    """

    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold
        self.records: dict = {}

    def evaluate(self, model, X: np.ndarray, y: np.ndarray,
                 name: str = "Model", split: str = "test") -> dict:
        """Evaluates model on given set. / 在指定数据集上评估模型。"""
        proba = model.predict_proba(X)[:, 1]
        pred  = (proba >= self.threshold).astype(int)

        auroc  = roc_auc_score(y, proba)
        ap     = average_precision_score(y, proba)
        brier  = brier_score_loss(y, proba)
        f1     = f1_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        prec   = precision_score(y, pred, zero_division=0)

        m = dict(AUROC=auroc, AP=ap, Brier=brier,
                 F1=f1, Recall=recall, Precision=prec)
        self.records[f"{name}_{split}"] = m

        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        fnr = fn / max(fn + tp, 1)
        fpr = fp / max(fp + tn, 1)

        print(f"\n[{split.upper()}] {name} (Threshold={self.threshold})")
        print(f"  AUROC={auroc:.4f} | AP={ap:.4f} | Brier={brier:.4f} | F1={f1:.4f}")
        print(f"  Recall={recall:.4f} | Precision={prec:.4f}")
        print(f"  Matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"  FNR (Miss Rate)={fnr:.2%} | FPR (False Alarm)={fpr:.2%}")

        # Calibration assessment / 校准度评估
        if brier < 0.05:
            cal_note = "Excellent Calibration"
        elif brier < 0.10:
            cal_note = "Good Calibration"
        else:
            cal_note = "Calibration needs improvement"
        print(f"  Calibration Assessment: {cal_note} (Brier={brier:.4f})")
        return m

    def nested_cv(self, model_fn, X: np.ndarray, y: np.ndarray,
                  groups: np.ndarray = None,
                  name: str = "Model", folds: int = 5):
        """Outer loop for nested cross validation. / 嵌套交叉验证的外层循环。"""
        print(f"\n[NestedCV] {name} - {folds} folds...")

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        split_iter = list(cv.split(X, y))

        aucs, aps, f1s, recalls = [], [], [], []
        for fold, (tr, te) in enumerate(split_iter, 1):
            if len(te) == 0 or y[te].sum() == 0:
                print(f"  Fold {fold}: Skipped (No positives in test set)")
                continue
            m = model_fn()
            m.fit(X[tr], y[tr])
            proba = m.predict_proba(X[te])[:, 1]
            pred  = (proba >= self.threshold).astype(int)
            try:
                auc = roc_auc_score(y[te], proba)
                ap  = average_precision_score(y[te], proba)
            except ValueError:
                continue
            aucs.append(auc)
            aps.append(ap)
            f1s.append(f1_score(y[te], pred, zero_division=0))
            recalls.append(recall_score(y[te], pred, zero_division=0))
            print(f"  Fold {fold}: AUROC={auc:.4f} AP={ap:.4f} Recall={recalls[-1]:.4f}")

        if not aucs:
            print("  [Warning] All folds skipped, CV metrics unavailable.")
            return {}

        print(f"  ── Mean Results ({len(aucs)} folds) ──")
        print(f"  AUROC  = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  AP     = {np.mean(aps):.4f} ± {np.std(aps):.4f}")
        print(f"  F1     = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"  Recall = {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        return dict(auroc=np.mean(aucs), auroc_std=np.std(aucs),
                    ap=np.mean(aps), recall=np.mean(recalls))

    def summary(self):
        """Prints performance summary for test set. / 打印测试集性能汇总。"""
        print("\n" + "═" * 70)
        print("  Final Model Performance Summary (Test Set)")
        print("═" * 70)
        test_records = {k: v for k, v in self.records.items()
                        if k.endswith('_test')}
        print(f"{'Model':<22} {'AUROC':>8} {'AP':>8} {'Brier':>8} {'Recall':>8} {'F1':>8}")
        print("-" * 70)
        for name, m in test_records.items():
            mname = name.replace('_test', '')
            print(f"{mname:<22} {m['AUROC']:>8.4f} {m['AP']:>8.4f} "
                  f"{m['Brier']:>8.4f} {m['Recall']:>8.4f} {m['F1']:>8.4f}")
        print()

        if test_records:
            best = max(test_records, key=lambda k: test_records[k]['AUROC'])
            best_name = best.replace('_test', '')
            print(f"  ★ Recommended Model (Highest AUROC): {best_name}")
            print("  ★ Note: Brier Score < 0.1 indicates excellent probability calibration.")
            return best_name
        return None


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PREDICTOR (Inference Interface)
#               预测器 (推理接口)
# ══════════════════════════════════════════════════════════════════════════

class RAPredictor:
    """
    Interface for real-time risk estimation.
    实时风险评估接口。
    """

    RISK_BINS   = [0, 0.15, 0.35, 0.60, 1.01]
    RISK_LABELS = ['Low Risk (<15%)', 'Moderate Risk (15-35%)',
                   'High Risk (35-60%)', 'Very High Risk (>60%)']

    def __init__(self, model, engineer: FeatureEngineer,
                 weighter: DOMWeighter, threshold: float = 0.30):
        self.model     = model
        self.engineer  = engineer
        self.weighter  = weighter
        self.threshold = threshold

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimates risk for input dataframe. / 对输入的数据框进行风险评估。"""
        X = self.engineer.transform(df)
        X_w = self.weighter.apply(X)
        proba = self.model.predict_proba(X_w.values)[:, 1]
        pred  = (proba >= self.threshold).astype(int)

        out = df.copy()
        out['RA_probability'] = np.round(proba * 100, 1)
        out['RA_prediction']  = pred
        out['RA_risk_level']  = pd.cut(
            proba, bins=self.RISK_BINS, labels=self.RISK_LABELS,
            include_lowest=True)
        out['needs_followup'] = (proba >= 0.15)
        return out

    def explain(self, row: pd.Series) -> str:
        """Generates clinical risk explanation. / 生成临床风险解释。"""
        X = self.engineer.transform(row.to_frame().T)
        X_w = self.weighter.apply(X)
        proba = self.model.predict_proba(X_w.values)[0, 1]
        risk_pct = proba * 100

        top_features = sorted(
            self.weighter.weights.items(),
            key=lambda x: x[1], reverse=True)[:5]
        top_names = [f for f, _ in top_features]

        text = (
            f"Based on the input data, the predicted RA probability is {risk_pct:.1f}%.\n"
            f"Key risk factors (Expert weighted): {', '.join(top_names)}.\n"
            f"Note: This is for clinical research only and does not constitute a diagnosis."
        )
        return text


# ══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
#   主流程
# ══════════════════════════════════════════════════════════════════════════

def run_pipeline(data_path: str = None,
                 custom_weights: dict = None,
                 run_hypersearch: bool = False,
                 run_nested_cv: bool = True) -> tuple:
    """Full training pipeline execution. / 执行完整训练流程。"""
    print("\n" + "★" * 62)
    print("  RA Prediction Pipeline v2.0 - Expert Augmented GBDT")
    print("★" * 62 + "\n")

    # ── 1. Load & Merge Weights / 加载与权重合并 ────────────────────────────
    path = data_path or CONFIG["screened_data_path"]
    if not Path(path).exists():
        alt = Path(__file__).parent / path
        if alt.exists():
            path = str(alt)

    df = DataLoader.load(path)
    nhanes_w = DataLoader.merge_nhanes_weights(
        df, n_cycles=CONFIG["n_nhanes_cycles"])

    # ── 2. Feature Engineering / 特征工程 ───────────────────────────────────
    engineer = FeatureEngineer()
    X_df = engineer.fit_transform(df)

    # ── 3. DOM Weighting / 专家加权 ────────────────────────────────────────
    final_weights = DOM_FEATURE_WEIGHTS.copy()
    if custom_weights:
        final_weights.update(custom_weights)
        print(f"[DOMWeighting] Applied {len(custom_weights)} weight overrides.")

    weighter = DOMWeighter(final_weights,
                           MONOTONIC_CONSTRAINTS,
                           INTERACTION_GROUPS)
    X_weighted = weighter.apply(X_df)

    y = df['RheumatoidArthritis'].values
    X_arr = X_weighted.values

    # ── 4. Split Set (Anti-leakage) / 划分集合 (防泄露) ──────────────────────
    splitter = DataSplitter(
        test_size=CONFIG["test_size"],
        outer_folds=CONFIG["cv_outer_folds"],
        random_state=CONFIG["random_state"])
    train_idx, val_idx, test_idx, groups = splitter.split(df, y)

    X_train = X_arr[train_idx]; y_train = y[train_idx]
    X_val   = X_arr[val_idx];   y_val   = y[val_idx]
    X_test  = X_arr[test_idx];  y_test  = y[test_idx]

    sw_train = weighter.compute_sample_weights(
        nhanes_w.iloc[train_idx], y_train,
        strategy=CONFIG["imbalance_strategy"])

    # ── 5. Hyper-search (Optional) / 超参搜索 (可选) ───────────────────────
    if run_hypersearch:
        best_params = hyperparam_search(
            X_train, y_train,
            n_iter=CONFIG["n_hyperopt_iter"])
    else:
        best_params = None

    # ── 6. Build & Train Models / 构建与训练模型 ────────────────────────────
    feature_cols = list(X_weighted.columns)
    models = build_all_models(feature_cols, weighter)

    evaluator = Evaluator(threshold=CONFIG["decision_threshold"])

    print("\n" + "─" * 50)
    print("  Model Training Phase")
    print("─" * 50)

    trained = {}
    for name, model in models.items():
        print(f"\n▶ Training: {name}")

        if isinstance(model, ExpertAugmentedGBDT):
            model.fit(X_train, y_train, sample_weight=sw_train)
        else:
            model.fit(X_train, y_train)

        evaluator.evaluate(model, X_val, y_val, name=name, split="val")
        trained[name] = model

    # ── 7. Nested CV (Optional) / 嵌套交叉验证 (可选) ───────────────────────
    if run_nested_cv:
        print("\n" + "─" * 50)
        print("  Nested Cross-Validation (ExpertGBDT)")
        print("─" * 50)
        X_tv = np.vstack([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])
        evaluator.nested_cv(
            model_fn=lambda: ExpertAugmentedGBDT(feature_cols, weighter),
            X=X_tv, y=y_tv,
            name="ExpertGBDT",
            folds=CONFIG["cv_outer_folds"])

    # ── 8. Final Test Set Evaluation / 最终测试集评估 ───────────────────────
    print("\n" + "═" * 50)
    print("  Final Evaluation (Hold-out Test Set)")
    print("═" * 50)
    for name, model in trained.items():
        evaluator.evaluate(model, X_test, y_test, name=name, split="test")

    best_name = evaluator.summary()
    best_model = trained[best_name or "ExpertGBDT"]

    # ── 9. Feature Importance / 特征重要性 ──────────────────────────────────
    _feature_importance(trained, feature_cols, weighter)

    # ── 10. Wrap Predictor / 打包预测器 ────────────────────────────────────
    predictor = RAPredictor(
        model=best_model,
        engineer=engineer,
        weighter=weighter,
        threshold=CONFIG["decision_threshold"])

    print(f"\n[Pipeline] COMPLETED! Recommended model: {best_name}")
    print("  Inference: predictor.predict(new_df) -> Results with probability and risk level")
    print("  Explain: predictor.explain(row) -> Clinical suggestion text\n")
    return predictor, trained, evaluator


def _feature_importance(trained: dict, feature_cols: list,
                         weighter: DOMWeighter, top_n: int = 12):
    """Compares Model Importance vs Expert Weights. / 对比模型重要性与专家权重。"""
    print("\n[FeatureImportance] ExpertGBDT Internal Importance vs Domain Weights")
    model = trained.get("ExpertGBDT")
    if model is None: return

    base = model._base
    if hasattr(base, 'feature_importances_'):
        imp = base.feature_importances_
    else: return

    rows = sorted(
        zip(feature_cols, imp,
            [weighter.weights.get(c, 1.0) for c in feature_cols]),
        key=lambda x: x[1], reverse=True)

    print(f"\n  {'Feature':<28} {'Model Imp':>10} {'DOM Weight':>12}  Align")
    print("  " + "-" * 60)
    for col, mi, dw in rows[:top_n]:
        mi_rank = mi * 100
        dw_norm = dw / 5.0
        agree = "✓" if abs(mi_rank / (mi_rank + 1e-4) - dw_norm) < 0.5 else "△"
        bar = "▓" * max(1, int(mi * 150))
        print(f"  {col:<28} {mi:>10.4f} {dw:>12.1f}  {agree}  {bar}")


# ══════════════════════════════════════════════════════════════════════════
#  Helper: Inference on new patients
#   助手：对新患者进行推理
# ══════════════════════════════════════════════════════════════════════════

def predict_new_patients(predictor: RAPredictor,
                          csv_path: str,
                          col_map: dict = None) -> pd.DataFrame:
    """Runs RA risk estimation on new patient data. / 在新患者数据上运行 RA 风险评估。"""
    new_df = DataLoader.load_new(csv_path, col_map=col_map)
    result = predictor.predict(new_df)
    print(f"\n[Predict] Estimation completed for {len(result)} records.")
    dist = result['RA_risk_level'].value_counts()
    for level, count in dist.items():
        print(f"  {level}: {count} patients ({count/len(result):.1%})")
    return result


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
#   入口点
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    predictor, models, evaluator = run_pipeline(
        data_path,
        run_hypersearch=False,
        run_nested_cv=True,
    )

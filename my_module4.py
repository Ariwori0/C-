import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# ==========================================
# 🛠️ 1. 共通前処理クラス (Signal Processors)
# ==========================================
class SubjectAwareSmoother(BaseEstimator, TransformerMixin):
    """移動平均による平滑化 (Subject IDを考慮)"""
    def __init__(self, window=15,method='ewm'):
        self.window = window
        self.method = method

    def transform(self, X):
        if 'subject_id' in X.columns:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            cols_to_smooth = [c for c in numeric_cols if c != 'subject_id']
            X_num = X[cols_to_smooth]
            
            rolled = X_num.groupby(X['subject_id']).rolling(self.window, min_periods=1).mean()
            rolled = rolled.reset_index(level=0, drop=True).sort_index()
            
            X_out = X.copy()
            X_out[cols_to_smooth] = rolled
            return X_out
        else:
            return X.rolling(self.window, min_periods=1).mean()

    def fit(self, X, y=None): return self

class SubjectAwareInitialBiasSubtractor(BaseEstimator, TransformerMixin):
    """
    初期値引き (Bias Subtraction)
    Args:
        n_samples (int): 初期平均を計算するサンプル数
        skip_keywords (list): 初期値引きを行わない列名に含まれるキーワードのリスト
                              (Noneの場合はデフォルト設定を使用)
    """
    def __init__(self, n_samples=50, skip_keywords=None):
        self.n_samples = n_samples
        # デフォルトの除外キーワード
        if skip_keywords is None:
            self.skip_keywords = ['subject_id', '_init', '_vel', '_hysteresis', '_CLR', '_Power']
        else:
            self.skip_keywords = skip_keywords

    def transform(self, X):
        X_out = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # ★ ここで除外判定を行う（キーワードが含まれていない列だけを対象にする）
        cols = [c for c in numeric_cols if not any(k in c for k in self.skip_keywords)]
        
        if 'subject_id' in X_out.columns:
            for uid in X_out['subject_id'].unique():
                mask = X_out['subject_id'] == uid
                subset = X_out.loc[mask, cols]
                if len(subset) > 0:
                    bias = subset.iloc[:self.n_samples].mean()
                    X_out.loc[mask, cols] = subset - bias
        else:
            bias = X_out[cols].iloc[:self.n_samples].mean()
            X_out[cols] = X_out[cols] - bias
        return X_out

    def fit(self, X, y=None): return self

def generate_static_features(X, n_samples=50):
    """時系列データから静的特徴量（初期値）を生成して結合"""
    X_out = X.copy()
    if 'subject_id' not in X.columns:
        return X_out
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    cols = [c for c in numeric_cols if c != 'subject_id']
    init_means = X.groupby('subject_id')[cols].apply(lambda x: x.iloc[:n_samples].mean())
    init_means.columns = [f"{c}_init" for c in init_means.columns]
    X_out = pd.merge(X_out, init_means, on='subject_id', how='left')
    return X_out

# ==========================================
# 🧠 2. コアモデル (Hierarchical PLS)
# ==========================================
class HierarchicalPLSModel:
    """階層的混合モデル (Imputer付き)"""
    def __init__(self, n_components=15, use_gain=True, use_offset=True):
        self.n_components = n_components
        self.use_gain = use_gain
        self.use_offset = use_offset
        self.global_pls = None
        self.subject_gains = {}
        self.subject_offsets = {}
        self.default_gain = 1.0
        self.default_offset = 0.0
        # self.imputer = SimpleImputer(strategy='mean')
        
    def fit(self, X, y, subjects):
        if len(y.shape) > 1: y = y.flatten()
        # X = self.imputer.fit_transform(X)
        
        # Stage 1: Global PLS
        self.global_pls = PLSRegression(n_components=self.n_components, scale=False)
        self.global_pls.fit(X, y)
        y_global = self.global_pls.predict(X).flatten()
        
        # Stage 2: Subject Correction
        unique_subjects = np.unique(subjects)
        all_gains, all_offsets = [], []
        
        for subj in unique_subjects:
            mask = (subjects == subj)
            y_true_s = y[mask]
            y_glob_s = y_global[mask]
            
            if len(y_true_s) < 2 or np.std(y_glob_s) < 1e-6:
                gain, offset = 1.0, 0.0
            else:
                if self.use_gain and self.use_offset:
                    A = np.column_stack([y_glob_s, np.ones(len(y_glob_s))])
                    params = np.linalg.lstsq(A, y_true_s, rcond=None)[0]
                    gain, offset = params[0], params[1]
                elif self.use_gain:
                    gain = np.sum(y_true_s * y_glob_s) / (np.sum(y_glob_s**2) + 1e-8)
                    offset = 0.0
                elif self.use_offset:
                    gain = 1.0
                    offset = np.mean(y_true_s - y_glob_s)
                else:
                    gain, offset = 1.0, 0.0
            
            self.subject_gains[subj] = gain
            self.subject_offsets[subj] = offset
            all_gains.append(gain)
            all_offsets.append(offset)
        
        self.default_gain = np.median(all_gains) if all_gains else 1.0
        self.default_offset = np.median(all_offsets) if all_offsets else 0.0
        return self
    
    def predict(self, X, subjects):
        # X = self.imputer.transform(X)
        y_global = self.global_pls.predict(X).flatten()
        y_corrected = np.zeros_like(y_global)
        
        if np.isscalar(subjects): subjects = np.full(len(X), subjects)
            
        for subj in np.unique(subjects):
            mask = (subjects == subj)
            # 未知被験者は学習時の全被験者のgain/offsetの中央値 (median) 
            gain = self.subject_gains.get(subj, self.default_gain) 
            offset = self.subject_offsets.get(subj, self.default_offset)
            y_corrected[mask] = gain * y_global[mask] + offset
        
        return y_corrected
    
    def get_global_prediction(self, X):
        """グローバルPLSのみの予測（診断用）"""
        return self.global_pls.predict(X).flatten()

# ==========================================
# ♟️ 3. 戦略クラス (Strategy Pattern)
# ==========================================
class BaseModelStrategy(ABC):
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.enable_scale_x = config.get('scale_x', True)
        self.n_components_setting = config.get('n_components', 15) # Default 15
        self.scaler_x = StandardScaler()
        # self.imputer = SimpleImputer(strategy='mean')
        self.best_n_components_ = None # 自動探索結果格納用

    def _preprocess_internal(self, X, is_training=True):
        X_proc = X
        if self.enable_scale_x:
            if is_training:
                X_proc = self.scaler_x.fit_transform(X_proc)
            else:
                X_proc = self.scaler_x.transform(X_proc)
        return X_proc

    def _optimize_n_components(self, model_class, X, y, subjects=None, max_n=20):
        """成分数の自動探索 (CV)"""
        n_features = X.shape[1]
        limit = min(n_features, max_n)
        candidates = range(1, limit + 1)
        
        best_score = float('inf')
        best_n = 1
        
        if subjects is not None:
            cv = GroupKFold(n_splits=5)
            groups = subjects
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            groups = None
            
        print(f"   🔍 Auto-tuning n_components (1-{limit})...")
        
        for n in candidates:
            if model_class == HierarchicalPLSModel:
                # 簡易的にStandardPLSで探索 (高速化)
                scorer = PLSRegression(n_components=n, scale=False) 
            else:
                model = PLSRegression(n_components=n, scale=False)
                scorer = model

            scores = cross_val_score(scorer, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error')
            mse = -np.mean(scores)
            
            if mse < best_score:
                best_score = mse
                best_n = n
        
        print(f"   ✅ Best n_components: {best_n} (MSE: {best_score:.4f})")
        return best_n

    @abstractmethod
    def fit(self, X, y, subjects): pass

    @abstractmethod
    def predict(self, X, subjects): pass

# ------------------------------------------
# Strategy 1: Standard PLS
# ------------------------------------------
class StandardPLSStrategy(BaseModelStrategy):
    def __init__(self, config):
        super().__init__("Standard PLS", config)
        self.model = None

    def fit(self, X, y, subjects=None):
        X_proc = self._preprocess_internal(X, is_training=True)
        
        if self.n_components_setting == 'auto':
            best_n = self._optimize_n_components(PLSRegression, X_proc, y, subjects)
            self.best_n_components_ = best_n
            n_comp = best_n
        else:
            n_comp = int(self.n_components_setting)
            
        self.model = PLSRegression(n_components=n_comp, scale=False)
        self.model.fit(X_proc, y)
        return self

    def predict(self, X, subjects=None):
        X_proc = self._preprocess_internal(X, is_training=False)
        return self.model.predict(X_proc).flatten()

class HierarchicalPLSModelZeroShot(HierarchicalPLSModel):
    """Zero-shot 対応版 Hierarchical PLS"""
    
    def __init__(self, n_components=15, use_gain=True, use_offset=True, 
                 ridge_alpha=1.0):
        super().__init__(n_components, use_gain, use_offset)
        self.ridge_alpha = ridge_alpha
        self.gain_predictor = None
        self.offset_predictor = None
        self.static_feature_names = None
    
    def fit(self, X, y, subjects, static_features, feature_names):
        """
        学習（通常のfit + gain/offset予測モデルの学習）
        
        Args:
            X: 動的特徴量（前処理済み）
            y: 目的変数
            subjects: 被験者ID
            static_features: 静的特徴量の配列 (shape: [n_samples, n_static_features])
            feature_names: 静的特徴量の名前リスト
        """
        # Stage 1 & 2: 既存のfit()を実行
        super().fit(X, y, subjects)
        
        # Stage 3: 静的特徴量から gain/offset を予測するモデルを学習
        self.static_feature_names = feature_names
        self._fit_gain_offset_predictors(static_features, subjects)
        
        return self
    
    def _fit_gain_offset_predictors(self, static_features, subjects):
        """静的特徴量 → gain/offset の関係を学習"""
        
        # 被験者ごとの静的特徴量と gain/offset を収集
        unique_subjects = np.unique(subjects)
        X_static_list = []
        y_gain_list = []
        y_offset_list = []
        
        for subj in unique_subjects:
            # この被験者のデータマスク
            mask = (subjects == subj)
            
            # 静的特徴量の平均を取得（同一被験者内では一定値のはず）
            static_feat = static_features[mask].mean(axis=0)
            
            # この被験者の gain/offset
            gain = self.subject_gains.get(subj, self.default_gain)
            offset = self.subject_offsets.get(subj, self.default_offset)
            
            X_static_list.append(static_feat)
            y_gain_list.append(gain)
            y_offset_list.append(offset)
        
        X_static = np.array(X_static_list)
        y_gain = np.array(y_gain_list)
        y_offset = np.array(y_offset_list)
        
        # Ridge回帰で学習
        self.gain_predictor = Ridge(alpha=self.ridge_alpha)
        self.offset_predictor = Ridge(alpha=self.ridge_alpha)
        
        self.gain_predictor.fit(X_static, y_gain)
        self.offset_predictor.fit(X_static, y_offset)
        
        # 学習精度の評価
        gain_pred = self.gain_predictor.predict(X_static)
        offset_pred = self.offset_predictor.predict(X_static)
        
        r2_gain = r2_score(y_gain, gain_pred)
        r2_offset = r2_score(y_offset, offset_pred)
        
        print(f"\n✅ Gain/Offset Predictors Trained:")
        print(f"   - Gain R²: {r2_gain:.3f}")
        print(f"   - Offset R²: {r2_offset:.3f}")
        
        # 重要度分析
        self._analyze_feature_importance(X_static, y_gain, y_offset)
    
    def _analyze_feature_importance(self, X_static, y_gain, y_offset):
        """どの静的特徴量が重要か分析"""
        gain_coef = self.gain_predictor.coef_
        offset_coef = self.offset_predictor.coef_
        
        print(f"\n📊 Feature Importance (Coefficients):")
        print(f"{'Feature':<20} {'Gain':>10} {'Offset':>10}")
        print("-" * 42)
        
        for i, name in enumerate(self.static_feature_names):
            print(f"{name:<20} {gain_coef[i]:>10.4f} {offset_coef[i]:>10.4f}")
    
    def predict_new_subject(self, X, subject_id, static_features_new):
        """
        新規被験者の推論（Zero-shot）
        
        Args:
            X: 動的特徴量（前処理済み）
            subject_id: 新規被験者ID
            static_features_new: 新規被験者の静的特徴量ベクトル
        
        Returns:
            y_pred: 予測値
        """
        if self.gain_predictor is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # 静的特徴量から gain/offset を推定
        static_features_new = np.array(static_features_new).reshape(1, -1)
        predicted_gain = self.gain_predictor.predict(static_features_new)[0]
        predicted_offset = self.offset_predictor.predict(static_features_new)[0]
        
        # 新規被験者として登録
        self.subject_gains[subject_id] = predicted_gain
        self.subject_offsets[subject_id] = predicted_offset
        
        print(f"\n✅ Zero-shot Correction for Subject {subject_id}:")
        print(f"   - Predicted Gain: {predicted_gain:.4f}")
        print(f"   - Predicted Offset: {predicted_offset:.4f}")
        
        # 通常の predict() を呼び出し
        return self.predict(X, subjects=[subject_id] * len(X))
# ------------------------------------------
# Strategy 2: 階層的モデル Hierarchical PLS
# ------------------------------------------
class HierarchicalPLSStrategy(BaseModelStrategy):
    def __init__(self, config):
        super().__init__("Hierarchical PLS", config)
        self.model = None

    def fit(self, X, y, subjects):
        X_proc = self._preprocess_internal(X, is_training=True)
        
        if self.n_components_setting == 'auto':
            best_n = self._optimize_n_components(PLSRegression, X_proc, y, subjects)
            self.best_n_components_ = best_n
            n_comp = best_n
        else:
            n_comp = int(self.n_components_setting)
            
        self.model = HierarchicalPLSModel(n_components=n_comp, use_gain=True, use_offset=True)
        self.model.fit(X_proc, y, subjects)
        return self

    def predict(self, X, subjects):
        X_proc = self._preprocess_internal(X, is_training=False)
        return self.model.predict(X_proc, subjects)

# ------------------------------------------
# Strategy 3: Pure Stacking (PLS -> Ridge)
# ------------------------------------------
class StackingPLS2RidgeStrategy(BaseModelStrategy):
    def __init__(self, config, ridge_alpha=1.0, static_cols=None):
        super().__init__("Stacking (PLS->Ridge)", config)
        self.model_pls = None
        self.model_ridge = Ridge(alpha=ridge_alpha)
        self.static_cols = static_cols
        
        # ★ Stage 2用の設定とスケーラー
        self.enable_scale_stage2 = config.get('scale_stage2', False) # Default False
        self.scaler_stage2 = StandardScaler()

    def _split_features(self, X_df):
        if self.static_cols is None: return X_df, np.zeros((len(X_df), 0))
        available_static = [c for c in self.static_cols if c in X_df.columns]
        X_static = X_df[available_static].values
        drop_cols = available_static + ['subject_id'] if 'subject_id' in X_df.columns else available_static
        X_dynamic = X_df.drop(columns=drop_cols, errors='ignore').values
        return X_dynamic, X_static

    def fit(self, X, y, subjects=None):
        X_dyn, X_stat = self._split_features(X)
        X_dyn = self._preprocess_internal(X_dyn, is_training=True)
        
        if self.n_components_setting == 'auto':
            best_n = self._optimize_n_components(PLSRegression, X_dyn, y, subjects)
            self.best_n_components_ = best_n
            n_comp = best_n
        else:
            n_comp = int(self.n_components_setting)
            
        self.model_pls = PLSRegression(n_components=n_comp, scale=False)
        self.y_mean_ = np.mean(y)
        self.model_pls.fit(X_dyn, y - self.y_mean_)
        
        y_dyn_hat = self.model_pls.predict(X_dyn).reshape(-1, 1)
        X_stage2 = np.hstack([y_dyn_hat, X_stat])
        
        # ★ Stage 2 Scaling
        if self.enable_scale_stage2:
            X_stage2 = self.scaler_stage2.fit_transform(X_stage2)
            
        self.model_ridge.fit(X_stage2, y)
        return self

    def predict(self, X, subjects=None):
        X_dyn, X_stat = self._split_features(X)
        X_dyn = self._preprocess_internal(X_dyn, is_training=False)
        
        y_dyn_hat = self.model_pls.predict(X_dyn).reshape(-1, 1)
        X_stage2 = np.hstack([y_dyn_hat, X_stat])
        
        # ★ Stage 2 Scaling
        if self.enable_scale_stage2:
            X_stage2 = self.scaler_stage2.transform(X_stage2)
            
        return self.model_ridge.predict(X_stage2).flatten()

# -------------------------------------------------------
# Strategy 4: Hierarchical + Ridge (Hybrid)
# Hierarchicalの結果をさらにRidgeで補正 (最強構成？)
# -------------------------------------------------------
class HierarchicalRidgeHybridStrategy(BaseModelStrategy):
    def __init__(self, config, ridge_alpha=1.0, static_cols=None):
        super().__init__("4. Hierarchical + Ridge", config)
        self.static_cols = static_cols
        self.model_hpls = None
        self.model_ridge = Ridge(alpha=ridge_alpha)
        self.imputer_stage2 = SimpleImputer(strategy='mean')
        
        # ★ Stage 2用の設定とスケーラー
        self.enable_scale_stage2 = config.get('scale_stage2', False)
        self.scaler_stage2 = StandardScaler()
    
    def _split_features(self, X_df):
        if self.static_cols is None: return X_df, np.zeros((len(X_df), 0))
        available_static = [c for c in self.static_cols if c in X_df.columns]
        X_static = X_df[available_static].values
        drop_cols = available_static + ['subject_id'] if 'subject_id' in X_df.columns else available_static
        X_dynamic = X_df.drop(columns=drop_cols, errors='ignore').values
        return X_dynamic, X_static
    
    def fit(self, X, y, subjects):
        X_dyn, X_stat = self._split_features(X)
        X_dyn_proc = self._preprocess_internal(X_dyn, is_training=True)
        
        if self.n_components_setting == 'auto':
            best_n = self._optimize_n_components(PLSRegression, X_dyn_proc, y, subjects)
            self.best_n_components_ = best_n
            n_comp = best_n
        else:
            n_comp = int(self.n_components_setting)
            
        self.model_hpls = HierarchicalPLSModel(n_components=n_comp, use_gain=True, use_offset=True)
        self.model_hpls.fit(X_dyn_proc, y, subjects)
        y_hpls = self.model_hpls.predict(X_dyn_proc, subjects).reshape(-1, 1)
        
        X_stage2 = np.hstack([y_hpls, X_stat])
        X_stage2 = self.imputer_stage2.fit_transform(X_stage2)
        
        # ★ Stage 2 Scaling
        if self.enable_scale_stage2:
            X_stage2 = self.scaler_stage2.fit_transform(X_stage2)
        
        self.model_ridge.fit(X_stage2, y)
        return self

    def predict(self, X, subjects):
        X_dyn, X_stat = self._split_features(X)
        X_dyn_proc = self._preprocess_internal(X_dyn, is_training=False)
        
        y_hpls = self.model_hpls.predict(X_dyn_proc, subjects).reshape(-1, 1)
        
        X_stage2 = np.hstack([y_hpls, X_stat])
        X_stage2 = self.imputer_stage2.transform(X_stage2)
        
        # ★ Stage 2 Scaling
        if self.enable_scale_stage2:
            X_stage2 = self.scaler_stage2.transform(X_stage2)
        
        return self.model_ridge.predict(X_stage2).flatten()


# ==========================================
# 📊 4. 比較実行エンジン (Evaluator)
# ==========================================
class ModelComparator:
    def __init__(self, strategies, smooth_y=False, skip_bias_cols=None):
        """
        Args:
            strategies: 戦略リスト
            smooth_y (bool): 目的変数yに対しても平滑化を行うか
            skip_bias_cols (list): 初期値引きを除外するカラム名の一部（キーワード）のリスト
        """
        self.strategies = strategies
        self.smooth_y = smooth_y 
        self.smoother = SubjectAwareSmoother(window=10)
        # ★ 外部からリストを受け取ってインスタンス化
        self.bias_subtractor = SubjectAwareInitialBiasSubtractor(n_samples=60, skip_keywords=skip_bias_cols)

    def preprocess_common(self, X_raw):
        """共通の前処理（Static生成 -> Smooth -> BiasSub）"""
        print("   Running Common Preprocessing (X)...")
        X = X_raw.copy()
        X = generate_static_features(X, n_samples=50)
        X = self.smoother.transform(X)
        X = self.bias_subtractor.transform(X)
        return X.fillna(0) 

    def _get_y_target(self, y_raw, X_raw, target_col):
        """目的変数の取得（設定に応じて平滑化を行う）"""
        if self.smooth_y:
            print(f"   ✨ Smoothing Target Variable: {target_col}")
            temp_df = pd.DataFrame({
                'subject_id': X_raw['subject_id'].values,
                target_col: y_raw[target_col].values
            }, index=y_raw.index)
            temp_smooth = self.smoother.transform(temp_df)
            return temp_smooth[target_col].values.ravel()
        else:
            return y_raw[target_col].values.ravel()

    def run_comparison(self, X_raw, y_raw, target_col, holdout_subject_id):
        
        # --- リスト対応 (Multi-target Loop) ---
        if isinstance(target_col, list):
            if len(target_col) == 1:
                target_col = target_col[0]
            else:
                print(f"🔄 Multi-target batch processing: {target_col}")
                all_results = []
                for t_name in target_col:
                    res_df = self.run_comparison(X_raw, y_raw, t_name, holdout_subject_id)
                    res_df.insert(0, 'Target', t_name)
                    all_results.append(res_df)
                return pd.concat(all_results, ignore_index=True)

        # --- 単一ターゲット処理 ---
        print(f"\n🥊 Comparison Start: Target = {target_col}")
        
        X_proc = self.preprocess_common(X_raw)
        y_target = self._get_y_target(y_raw, X_raw, target_col)
        subjects = X_raw['subject_id'].values
        
        mask_test = subjects == holdout_subject_id
        mask_train = ~mask_test
        
        X_train, X_test = X_proc[mask_train], X_proc[mask_test]
        y_train, y_test = y_target[mask_train], y_target[mask_test]
        sub_train, sub_test = subjects[mask_train], subjects[mask_test]
        
        print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples (Subject {holdout_subject_id})")

        # 戦略ループ
        results_list = []
        for strategy in self.strategies:
            print(f"   🏃 Running: {strategy.name}...")
            
            strategy.fit(X_train, y_train, sub_train)
            y_pred = strategy.predict(X_test, sub_test)
            
            df_eval = pd.DataFrame({
                'True': y_test.flatten(), 
                'Pred': y_pred.flatten()
            }).dropna()
            
            if len(df_eval) > 0:
                r2 = r2_score(df_eval['True'], df_eval['Pred'])
                rmse = np.sqrt(mean_squared_error(df_eval['True'], df_eval['Pred']))
            else:
                r2, rmse = np.nan, np.nan
            
            best_n = getattr(strategy, 'best_n_components_', '-')
            
            results_list.append({
                'Model': strategy.name,
                'Best_N': best_n,
                'R2': r2,
                'RMSE': rmse
            })
            print(f"      -> R2: {r2:.4f} (N={best_n})")

        return pd.DataFrame(results_list)

# ==========================================
# 📊 5. 可視化関数群
# ==========================================
def plot_diagnosis_scatter(strategies, X_raw, y_raw, target_col, holdout_subject_id):
    """関数①: 散布図プロット (Train/Test)
    
    """
    print(f"📊 Generating Scatter Plots for {target_col}...")
    
    comparator = ModelComparator(strategies)
    X_proc = comparator.preprocess_common(X_raw)
    y_target = y_raw[target_col].values.ravel()
    subjects = X_raw['subject_id'].values
    
    mask_test = subjects == holdout_subject_id
    mask_train = ~mask_test
    
    X_train, X_test = X_proc[mask_train], X_proc[mask_test]
    y_train, y_test = y_target[mask_train], y_target[mask_test]
    sub_train = subjects[mask_train]
    
    for strategy in strategies:
        strategy.fit(X_train, y_train, sub_train)
        y_pred_train = strategy.predict(X_train, sub_train)
        y_pred_test = strategy.predict(X_test, subjects[mask_test])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Train
        df_train_plot = pd.DataFrame({'True': y_train, 'Pred': y_pred_train, 'Subject': sub_train})
        sns.scatterplot(data=df_train_plot, x='True', y='Pred', hue='Subject', palette='tab20', alpha=0.6, s=15, ax=axes[0], legend='full')
        min_tr, max_tr = y_train.min(), y_train.max()
        axes[0].plot([min_tr, max_tr], [min_tr, max_tr], 'r--', lw=2)
        axes[0].set_title(f"Train (R2={r2_score(y_train, y_pred_train):.3f})")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Subject ID")
        axes[0].grid(alpha=0.3)
        
        # Right: Test
        axes[1].scatter(y_test, y_pred_test, alpha=0.6, s=20, c='orange', label=f'Subj {holdout_subject_id}')
        min_te, max_te = y_test.min(), y_test.max()
        axes[1].plot([min_te, max_te], [min_te, max_te], 'r--', lw=2)
        axes[1].set_title(f"Test (Target: {holdout_subject_id}) | R2={r2_score(y_test, y_pred_test):.3f}")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        
        plt.suptitle(f"Strategy: {strategy.name}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

def plot_diagnosis_timeseries_grid(strategies, X_raw, y_raw, target_col, holdout_subject_id):
    """関数②: 全被験者時系列グリッド"""
    print(f"📊 Generating Time Series Grid for {target_col}...")
    
    comparator = ModelComparator(strategies)
    X_proc = comparator.preprocess_common(X_raw)
    y_target = y_raw[target_col].values.ravel()
    subjects = X_raw['subject_id'].values
    unique_subjects = np.sort(np.unique(subjects))
    
    mask_test = subjects == holdout_subject_id
    mask_train = ~mask_test
    X_train = X_proc[mask_train]
    y_train = y_target[mask_train]
    sub_train = subjects[mask_train]
    
    all_preds = {}
    for strategy in strategies:
        strategy.fit(X_train, y_train, sub_train)
        all_preds[strategy.name] = strategy.predict(X_proc, subjects)

    n_cols = 2
    n_rows = (len(unique_subjects) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharey=True)
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['--', '-.', ':', '--', '-.']

    for i, subj_id in enumerate(unique_subjects):
        if i >= len(axes): break
        ax = axes[i]
        mask_subj = subjects == subj_id
        y_true_s = y_target[mask_subj]
        ax.plot(y_true_s, label='True', color='black', linewidth=2.5, alpha=0.4)
        
        for j, (strat_name, full_pred) in enumerate(all_preds.items()):
            y_pred_s = full_pred[mask_subj]
            ax.plot(y_pred_s, label=strat_name, color=colors[j % len(colors)], linestyle=line_styles[j % len(line_styles)], linewidth=1.2, alpha=0.8)
        
        status = "【TEST】" if subj_id == holdout_subject_id else "Train"
        ax.set_title(f"Subject {subj_id} ({status})", fontsize=12, fontweight='bold')
        if i == 0: ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)
    
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.show()

def plot_model_comparison_bar(results_df, title="Model Comparison by Target Variable", ylim=(0, 1.05), figsize=(15, 6)):
    """モデル比較結果の棒グラフ (目的変数ごと)"""
    plt.rcParams['font.family'] = 'MS Gothic'
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=results_df, x='Target', y='R2', hue='Model', palette='viridis', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.ylabel("R2 Score", fontsize=14); plt.xlabel("Target Variable", fontsize=14)
    plt.ylim(ylim); plt.grid(axis='y', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Model Strategy")
    plt.tight_layout()
    plt.show()




def plot_diagnosis_timeseries_grid2(strategies, X_raw, y_raw, target_col, holdout_subject_id, feature_col='pedal'):
    """
    関数②: 全被験者時系列グリッド (Twin軸で特徴量表示付き)
    
    Args:
        strategies: モデル戦略のリスト
        X_raw: 特徴量DataFrame
        y_raw: ターゲットDataFrame
        target_col: 予測対象のカラム名
        holdout_subject_id: テスト対象の被験者ID
        feature_col: 第2軸(右軸)に表示したい特徴量カラム名 (デフォルト: 'pedal')
    """
    print(f"📊 Generating Time Series Grid for {target_col} with {feature_col}...")
    
    # --- 前処理 & 予測 ---
    comparator = ModelComparator(strategies)
    X_proc = comparator.preprocess_common(X_raw)
    y_target = y_raw[target_col].values.ravel()
    subjects = X_raw['subject_id'].values
    unique_subjects = np.sort(np.unique(subjects))
    
    mask_test = subjects == holdout_subject_id
    mask_train = ~mask_test
    X_train = X_proc[mask_train]
    y_train = y_target[mask_train]
    sub_train = subjects[mask_train]
    
    all_preds = {}
    for strategy in strategies:
        strategy.fit(X_train, y_train, sub_train)
        all_preds[strategy.name] = strategy.predict(X_proc, subjects)

    # --- プロット設定 ---
    n_cols = 2
    n_rows = (len(unique_subjects) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharey=False) # Twinxを使うためshareyはFalse推奨、または手動調整
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['--', '-.', ':', '--', '-.']

    # --- ループ処理 ---
    for i, subj_id in enumerate(unique_subjects):
        if i >= len(axes): break
        ax = axes[i]
        
        # マスク作成
        mask_subj = subjects == subj_id
        
        # 1. 右軸 (Feature: Pedal) のプロット ※メインが見やすいよう背景的に薄く描画
        # X_rawがDataFrameであることを想定
        if feature_col in X_raw.columns:
            ax2 = ax.twinx()
            feat_vals = X_raw.loc[mask_subj, feature_col].values
            # 視認性を邪魔しないよう、灰色で塗りつぶし または 薄い線にする
            ax2.fill_between(range(len(feat_vals)), feat_vals, color='gray', alpha=0.15, label=feature_col)
            # 線だけが良い場合は以下を使用:
            # ax2.plot(feat_vals, color='gray', alpha=0.3, linewidth=1, label=feature_col)
            ax2.set_ylim([0,14])
            
            ax2.set_ylabel(feature_col, color='gray', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='gray')
        
        # 2. 左軸 (Ground Truth)
        y_true_s = y_target[mask_subj]
        # Z-orderを高めにしてPedalの上に描画
        ax.plot(y_true_s, label='True', color='black', linewidth=2.5, alpha=0.6, zorder=10)
        
        # 3. 左軸 (Predictions)
        for j, (strat_name, full_pred) in enumerate(all_preds.items()):
            y_pred_s = full_pred[mask_subj]
            ax.plot(y_pred_s, label=strat_name, 
                    color=colors[j % len(colors)], 
                    linestyle=line_styles[j % len(line_styles)], 
                    linewidth=1.5, alpha=0.9, zorder=11)
        
        # タイトル等の設定
        status = "【TEST】" if subj_id == holdout_subject_id else "Train"
        ax.set_title(f"Subject {subj_id} ({status})", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, zorder=0)
        
        # 凡例の統合 (左軸と右軸のラベルをまとめる)
        if i == 0:
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=8, framealpha=0.9)

    # 余ったサブプロットを非表示
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()



# =======================================================
# 戦略⑤: Hybrid + 非線形オフセット残差補正 (最強版)
# =======================================================
from sklearn.ensemble import RandomForestRegressor

class HierarchicalRidgeHybridPlusOffset(BaseModelStrategy):
    """
    戦略④ (Hybrid) の結果に対し、さらにRandomForestで
    「取り切れなかったオフセット誤差」を予測して補正するモデル
    """
    def __init__(self, config, ridge_alpha=1.0, static_cols=None):
        super().__init__("5. Hybrid + RF Offset", config)
        self.static_cols = static_cols
        
        # ベースモデル (戦略4と同じ構成)
        self.base_strategy = HierarchicalRidgeHybridStrategy(
            config, ridge_alpha, static_cols
        )
        
        # 残差補正用モデル (非線形)
        # オフセットは複雑な要因で決まるため、表現力の高いRFを採用
        self.offset_corrector = RandomForestRegressor(
            n_estimators=100, 
            max_depth=5,       # 過学習抑制のため浅めに
            random_state=42
        )
        self.scaler_static = StandardScaler() # RF用

    def fit(self, X, y, subjects):
        print(f"   🏃 Training {self.name}...")
        
        # 1. ベースモデル (Hybrid) を普通に学習
        self.base_strategy.fit(X, y, subjects)
        
        # 2. ベースモデルで学習データを予測してみる
        y_pred_base = self.base_strategy.predict(X, subjects)
        
        # 3. 「取り切れなかったオフセット誤差」を計算
        # 各被験者ごとに (正解 - 予測) の平均値を計算
        residuals = y - y_pred_base
        unique_subjects = np.unique(subjects)
        
        X_stat_list = []
        y_resid_offset_list = []
        
        # 静的特徴量の取得用
        if self.static_cols is not None:
            available_static = [c for c in self.static_cols if c in X.columns]
            X_static_all = X[available_static].values
        else:
            raise ValueError("static_cols required")

        for subj in unique_subjects:
            mask = (subjects == subj)
            
            # この被験者の平均的なズレ (Offset残差)
            mean_residual = np.mean(residuals[mask])
            
            # この被験者の静的特徴量 (代表値)
            stat_feat = np.mean(X_static_all[mask], axis=0)
            
            X_stat_list.append(stat_feat)
            y_resid_offset_list.append(mean_residual)
            
        # 4. 静的特徴量 -> 残留オフセット を予測するRFを学習
        X_stat_train = np.array(X_stat_list)
        y_resid_train = np.array(y_resid_offset_list)
        
        # RF用にスケーリング（必須ではないが良い習慣）
        X_stat_train = self.scaler_static.fit_transform(X_stat_train)
        
        self.offset_corrector.fit(X_stat_train, y_resid_train)
        
        # 精度確認
        r2 = self.offset_corrector.score(X_stat_train, y_resid_train)
        print(f"      ✅ Offset Residual Corrector Trained (R2={r2:.3f})")
        
        return self

    def predict(self, X, subjects):
        # 1. ベースモデルで予測
        y_base = self.base_strategy.predict(X, subjects)
        
        # 2. 静的特徴量を抽出
        available_static = [c for c in self.static_cols if c in X.columns]
        X_static = X[available_static].values
        
        # 3. 被験者ごとにオフセット補正量を予測して加算
        # (Zero-shot対応: 未知の被験者IDが来ても、その場のX_staticから推論する)
        
        # 行ごとに予測するのは非効率なので、Uniqueな静的特徴量パターンごとに計算したいが、
        # 実装を簡単にするため全行に対してpredictをかける (RFは高速なのでOK)
        X_static_scaled = self.scaler_static.transform(X_static)
        offset_correction = self.offset_corrector.predict(X_static_scaled)
        
        # 4. 最終予測
        return y_base + offset_correction



# =======================================================
# 戦略⑥: アンサンブル (Hybrid + Hybrid_RF の平均)
# =======================================================
class EnsembleHybridStrategy(BaseModelStrategy):
    """
    「線形補正 (Ridge)」と「非線形補正 (RF)」の予測値を平均する戦略。
    RFが過剰適合して暴れた場合のリスクを、堅実なRidgeが緩和する。
    """
    def __init__(self, config, ridge_alpha=1.0, static_cols=None):
        super().__init__("6. Ensemble (Linear + RF)", config)
        
        # 2つのモデルを内部に持つ
        # 1. 赤線: 線形補正 (安定型)
        self.model_linear = HierarchicalRidgeHybridStrategy(
            config, ridge_alpha, static_cols
        )
        
        # 2. 紫線: 非線形残差補正 (特化型)
        self.model_rf = HierarchicalRidgeHybridPlusOffset(
            config, ridge_alpha, static_cols
        )

    def fit(self, X, y, subjects):
        print(f"   🏃 Training {self.name}...")
        print("      └─ [1/2] Training Linear Base...")
        self.model_linear.fit(X, y, subjects)
        
        print("      └─ [2/2] Training RF Offset Corrector...")
        self.model_rf.fit(X, y, subjects)
        return self

    def predict(self, X, subjects):
        # 両方の予測を出して平均をとる
        pred_linear = self.model_linear.predict(X, subjects)
        pred_rf = self.model_rf.predict(X, subjects)
        
        # アンサンブル (単純平均)
        return (pred_linear + pred_rf) / 2


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from abc import ABC, abstractmethod
import my_module as mm_base  # 既存のmy_moduleを継承

# ==========================================
# 🧠 Zero-shot 対応 Hierarchical PLS モデル (Core)
# ==========================================

class HierarchicalPLSModelZeroShot:
    """
    Zero-shot 対応版 Hierarchical PLS
    未知の被験者に対して、静的特徴量(X_static)からGain/Offsetを推定して適用する。
    """
    def __init__(self, n_components=15, use_gain=True, use_offset=True, ridge_alpha=1.0):
        self.n_components = n_components
        self.use_gain = use_gain
        self.use_offset = use_offset
        self.ridge_alpha = ridge_alpha
        
        # 内部モデル
        self.global_pls = None
        self.gain_predictor = None   # X_static -> Gain
        self.offset_predictor = None # X_static -> Offset
        
        # 記憶用パラメータ
        self.subject_gains = {}
        self.subject_offsets = {}
        self.default_gain = 1.0
        self.default_offset = 0.0
        
        # Imputer
        self.imputer = SimpleImputer(strategy='mean')

    def fit(self, X_dynamic, y, subjects, X_static, feature_names=None):
        """
        Args:
            X_dynamic: 動的特徴量 (PLSへの入力)
            y: 目的変数
            subjects: 被験者ID配列
            X_static: 静的特徴量配列 (Gain/Offset予測用)
            feature_names: 特徴量名リスト（分析用）
        """
        if len(y.shape) > 1: y = y.flatten()
        X_dynamic = self.imputer.fit_transform(X_dynamic)
        
        # --------------------------------------------
        # 1. Global PLS (ベースモデル) の学習
        # --------------------------------------------
        self.global_pls = PLSRegression(n_components=self.n_components, scale=False)
        self.global_pls.fit(X_dynamic, y)
        y_global = self.global_pls.predict(X_dynamic).flatten()
        
        # --------------------------------------------
        # 2. 各被験者の「正解 Gain / Offset」を算出
        # --------------------------------------------
        unique_subjects = np.unique(subjects)
        X_static_train = []
        y_gain_train = []
        y_offset_train = []
        
        all_gains = []
        all_offsets = []

        for subj in unique_subjects:
            mask = (subjects == subj)
            y_true_s = y[mask]
            y_glob_s = y_global[mask]
            
            # Gain/Offset計算 (最小二乗法)
            if len(y_true_s) < 5 or np.std(y_glob_s) < 1e-6:
                gain, offset = 1.0, 0.0
            else:
                if self.use_gain and self.use_offset:
                    # y_true = gain * y_global + offset
                    A = np.column_stack([y_glob_s, np.ones(len(y_glob_s))])
                    params = np.linalg.lstsq(A, y_true_s, rcond=None)[0]
                    gain, offset = params[0], params[1]
                elif self.use_gain:
                    gain = np.sum(y_true_s * y_glob_s) / (np.sum(y_glob_s**2) + 1e-8)
                    offset = 0.0
                elif self.use_offset:
                    gain = 1.0
                    offset = np.mean(y_true_s - y_glob_s)
                else:
                    gain, offset = 1.0, 0.0
            
            # 登録
            self.subject_gains[subj] = gain
            self.subject_offsets[subj] = offset
            all_gains.append(gain)
            all_offsets.append(offset)
            
            # メタモデル学習用データ作成
            if X_static is not None:
                subj_static_vec = X_static[mask].mean(axis=0)
                X_static_train.append(subj_static_vec)
                y_gain_train.append(gain)
                y_offset_train.append(offset)

        # デフォルト値（中央値）
        self.default_gain = np.median(all_gains) if all_gains else 1.0
        self.default_offset = np.median(all_offsets) if all_offsets else 0.0
        
        # --------------------------------------------
        # 3. メタモデル (Gain/Offset Predictor) の学習
        # --------------------------------------------
        if X_static is not None and len(X_static_train) > 0:
            X_static_train = np.array(X_static_train) 
            y_gain_train = np.array(y_gain_train)
            y_offset_train = np.array(y_offset_train)
            
            self._fit_predictors(X_static_train, y_gain_train, y_offset_train)
            
        return self

    def _fit_predictors(self, X_train, y_gain, y_offset):
        """設定フラグに基づいてGain/Offset予測モデルを学習"""
        
        # Gain Predictor (use_gain=True の場合のみ学習)
        if self.use_gain:
            print("   🌲 Training Random Forest for Gain prediction...")
            self.gain_predictor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            self.gain_predictor.fit(X_train, y_gain)
        else:
            self.gain_predictor = None # 学習しない

        # Offset Predictor (use_offset=True の場合のみ学習)
        if self.use_offset:
            print("   🌲 Training Random Forest for Offset prediction...")
            self.offset_predictor = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            self.offset_predictor.fit(X_train, y_offset)
        else:
            self.offset_predictor = None

    def predict(self, X_dynamic, subjects, X_static):
        """推論フェーズ"""
        X_dynamic = self.imputer.transform(X_dynamic)
        
        # 1. Global Prediction (ベースライン)
        y_global = self.global_pls.predict(X_dynamic).flatten()
        y_corrected = np.zeros_like(y_global)
        
        if np.isscalar(subjects):
            subjects = np.full(len(X_dynamic), subjects)
            
        unique_test_subjects = np.unique(subjects)
        
        for subj in unique_test_subjects:
            mask = (subjects == subj)
            
            # --- 補正係数の決定 ---
            if subj in self.subject_gains:
                # A. 既知の被験者 -> 登録値を使用
                gain = self.subject_gains[subj]
                offset = self.subject_offsets[subj]
            else:
                # B. 未知の被験者 -> Zero-shot 推定
                if X_static is not None:
                    subj_static_data = X_static[mask]
                    
                    if len(subj_static_data) > 0:
                        # 全サンプルの平均を使って予測
                        subj_static_vec = subj_static_data.mean(axis=0).reshape(1, -1)
                        
                        # ★ Gain予測 (モデルがあれば予測、なければ1.0)
                        if self.gain_predictor is not None:
                            gain = self.gain_predictor.predict(subj_static_vec)[0]
                        else:
                            gain = 1.0
                        
                        # ★ Offset予測 (モデルがあれば予測、なければ0.0)
                        if self.offset_predictor is not None:
                            offset = self.offset_predictor.predict(subj_static_vec)[0]
                        else:
                            offset = 0.0
                    else:
                        gain, offset = self.default_gain, self.default_offset
                else:
                    gain, offset = self.default_gain, self.default_offset
            
            # --- 補正適用 ---
            y_corrected[mask] = gain * y_global[mask] + offset
            
        return y_corrected



# =======================================================
# 戦略⑧: Global PLS + RF Offset 残差補正
# Globalモデル(Standard)をベースにし、残差をRFで埋める
# =======================================================
class GlobalPLSPlusOffsetStrategy(BaseModelStrategy):
    """
    Standard PLS の予測結果に対し、RandomForestで
    「オフセット誤差（残差）」を予測して補正するモデル。
    戦略5 (Hybrid+Offset) のベースモデルを Global PLS に置き換えたもの。
    """
    def __init__(self, config, static_cols=None):
        super().__init__("8. Global PLS + RF Offset", config)
        self.static_cols = static_cols
        
        # ベースモデル: Standard PLS (Global)
        # ※ 内部で PLSRegression を持つ
        self.model_global = None
        
        # 残差補正用モデル (非線形)
        self.offset_corrector = RandomForestRegressor(
            n_estimators=100, 
            max_depth=5,
            random_state=42
        )
        self.scaler_static = StandardScaler()

    def _split_features(self, X_df):
        """静的・動的特徴量の分離"""
        if self.static_cols is None: 
            return X_df, np.zeros((len(X_df), 0))
            
        available_static = [c for c in self.static_cols if c in X_df.columns]
        X_static = X_df[available_static].values
        
        drop_cols = available_static + ['subject_id'] if 'subject_id' in X_df.columns else available_static
        X_dynamic = X_df.drop(columns=drop_cols, errors='ignore').values
        
        return X_dynamic, X_static

    def fit(self, X, y, subjects):
        print(f"   🏃 Training {self.name}...")
        
        # 1. 特徴量分離
        X_dyn, X_stat = self._split_features(X)
        
        # 2. 前処理 (Scaling)
        X_dyn_proc = self._preprocess_internal(X_dyn, is_training=True)
        
        # 3. Global PLS 学習
        if self.n_components_setting == 'auto':
            n_comp = self._optimize_n_components(PLSRegression, X_dyn_proc, y, subjects)
            self.best_n_components_ = n_comp
        else:
            n_comp = int(self.n_components_setting)
            self.best_n_components_ = n_comp
            
        self.model_global = PLSRegression(n_components=n_comp, scale=False)
        self.model_global.fit(X_dyn_proc, y)
        
        # 4. ベース予測と残差の計算
        y_pred_global = self.model_global.predict(X_dyn_proc).flatten()
        residuals = y - y_pred_global
        
        # 5. 被験者ごとの平均残差と静的特徴量を紐付け
        unique_subjects = np.unique(subjects)
        X_stat_train_list = []
        y_resid_train_list = []
        
        for subj in unique_subjects:
            mask = (subjects == subj)
            
            # この被験者の平均的なズレ (Offset残差)
            # 例: Globalだと平均5mm低く出る -> residualは +5mm
            mean_residual = np.mean(residuals[mask])
            
            # 静的特徴量の代表値
            stat_feat = np.mean(X_stat[mask], axis=0)
            
            X_stat_train_list.append(stat_feat)
            y_resid_train_list.append(mean_residual)
            
        # 6. RF学習 (静的特徴量 -> 残差)
        X_stat_train = np.array(X_stat_train_list)
        y_resid_train = np.array(y_resid_train_list)
        
        X_stat_train = self.scaler_static.fit_transform(X_stat_train)
        self.offset_corrector.fit(X_stat_train, y_resid_train)
        
        return self

    def predict(self, X, subjects):
        # 1. 特徴量分離
        X_dyn, X_stat = self._split_features(X)
        
        # 2. 前処理
        X_dyn_proc = self._preprocess_internal(X_dyn, is_training=False)
        
        # 3. Global PLS 予測
        y_global = self.model_global.predict(X_dyn_proc).flatten()
        
        # 4. 残差(Offset) 予測
        X_stat_scaled = self.scaler_static.transform(X_stat)
        offset_correction = self.offset_corrector.predict(X_stat_scaled)
        
        # 5. 加算して返す
        return y_global + offset_correction

# ==========================================
# ♟️ Strategy Wrapper (Comparator用)
# ==========================================

class HierarchicalPLSStrategyZeroShot(mm_base.BaseModelStrategy):
    """
    Standard Zero-shot (Gain + Offset 両方予測)
    """
    def __init__(self, name, config, static_features_cols, ridge_alpha=1.0):
        super().__init__(name, config) 
        self.static_features_cols = static_features_cols
        self.ridge_alpha = ridge_alpha
        self.model = None
        self.enable_scale_x = config.get('scale_x', False)
        self.n_components_setting = config.get('n_components', 15)

    def fit(self, X, y, subjects):
        print(f"   🏃 Training {self.name}...")
        
        available_static = [c for c in self.static_features_cols if c in X.columns]
        X_static = X[available_static].values
        
        drop_cols = available_static + ['subject_id']
        X_dynamic_df = X.drop(columns=drop_cols, errors='ignore')
        X_dynamic = X_dynamic_df.values
        
        X_dynamic_proc = self._preprocess_internal(X_dynamic, is_training=True)
        
        if self.n_components_setting == 'auto':
            best_n = self._optimize_n_components(PLSRegression, X_dynamic_proc, y, subjects)
            n_comp = best_n
        else:
            n_comp = int(self.n_components_setting)
            self.best_n_components_ = n_comp

        # GainとOffsetの両方を有効にする（デフォルト）
        self.model = HierarchicalPLSModelZeroShot(
            n_components=n_comp,
            use_gain=True, 
            use_offset=True,
            ridge_alpha=self.ridge_alpha
        )
        
        self.model.fit(X_dynamic_proc, y, subjects, X_static, feature_names=available_static)
        return self

    def predict(self, X, subjects):
        available_static = [c for c in self.static_features_cols if c in X.columns]
        X_static = X[available_static].values
        
        drop_cols = available_static + ['subject_id']
        X_dynamic_df = X.drop(columns=drop_cols, errors='ignore')
        X_dynamic = X_dynamic_df.values
        
        X_dynamic_proc = self._preprocess_internal(X_dynamic, is_training=False)
        
        return self.model.predict(X_dynamic_proc, subjects, X_static)


# =======================================================
# 戦略⑦: Zero-shot (Offset Only) - 安全版
# =======================================================
class HierarchicalPLSStrategyZeroShotOffsetOnly(HierarchicalPLSStrategyZeroShot):
    """
    Gain予測は行わず(1.0固定)、Offsetのみを予測する安全版戦略
    """
    def fit(self, X, y, subjects):
        print(f"   🏃 Training {self.name} (Offset Only)...")
        
        available_static = [c for c in self.static_features_cols if c in X.columns]
        X_static = X[available_static].values
        
        drop_cols = available_static + ['subject_id']
        X_dynamic_df = X.drop(columns=drop_cols, errors='ignore')
        X_dynamic = X_dynamic_df.values
        
        X_dynamic_proc = self._preprocess_internal(X_dynamic, is_training=True)
        
        if self.n_components_setting == 'auto':
            n_comp = self._optimize_n_components(PLSRegression, X_dynamic_proc, y, subjects)
            self.best_n_components_ = n_comp
        else:
            n_comp = int(self.n_components_setting)
            self.best_n_components_ = n_comp

        # ★ここが重要: use_gain=False を渡す
        self.model = HierarchicalPLSModelZeroShot(
            n_components=n_comp,
            use_gain=False,   # <--- Gain予測無効 (1.0固定)
            use_offset=True,  # <--- Offset予測有効
            ridge_alpha=self.ridge_alpha
        )
        
        self.model.fit(X_dynamic_proc, y, subjects, X_static, feature_names=available_static)
        return self



# class HierarchicalPLSStrategyZeroShot:
#     """
#     Zero-shot対応版 Hierarchical PLS Strategy
#     """
#     def __init__(self, name, config, static_features_cols, ridge_alpha=1.0):
#         self.name = name
#         self.config = config
#         self.static_features_cols = static_features_cols
#         self.ridge_alpha = ridge_alpha
        
#         self.enable_scale_x = config.get('scale_x', False)
#         self.n_components_setting = config.get('n_components', 15)
        
#         from sklearn.preprocessing import StandardScaler
#         self.scaler_x = StandardScaler()
#         self.model = None
#         self.best_n_components_ = None
    
#     def _preprocess_internal(self, X, is_training=True):
#         """前処理（既存コードと同じ）"""
#         X_proc = X
#         if self.enable_scale_x:
#             if is_training:
#                 X_proc = self.scaler_x.fit_transform(X_proc)
#             else:
#                 X_proc = self.scaler_x.transform(X_proc)
#         return X_proc
    
#     def fit(self, X, y, subjects):
#         """学習（Zero-shot predictorも含む）"""
#         print(f"   🏃 Running: {self.name}...")
        
#         # 🔧 利用可能な静的特徴量のみを抽出
#         available_static = [col for col in self.static_features_cols if col in X.columns]
        
#         if len(available_static) == 0:
#             raise ValueError("No static features available for Zero-shot!")
        
#         if len(available_static) < len(self.static_features_cols):
#             missing = set(self.static_features_cols) - set(available_static)
#             print(f"      ⚠️  Warning: Missing static features: {missing}")
#             print(f"      → Using {len(available_static)} available features")
        
#         # 静的特徴量の抽出
#         static_features = X[available_static].values
        
#         # 動的特徴量（静的特徴量とsubject_idを除く）
#         drop_cols = available_static + ['subject_id']
#         X_dynamic = X.drop(columns=drop_cols, errors='ignore')
        
#         print(f"      Dynamic features: {X_dynamic.shape[1]} columns")
#         print(f"      Static features: {len(available_static)} columns")
        
#         # 前処理（スケーリング等）
#         X_proc = self._preprocess_internal(X_dynamic.values, is_training=True)
        
#         # 🔧 n_components の調整（特徴量数に応じて）
#         n_features = X_proc.shape[1]
#         n_samples = X_proc.shape[0]
#         max_components = min(n_features, n_samples)
        
#         if self.n_components_setting > max_components:
#             print(f"      ⚠️  n_components={self.n_components_setting} → {max_components} (auto-adjusted)")
#             n_comp = max_components
#         else:
#             n_comp = self.n_components_setting
        
#         # ★ my_module4.py の HierarchicalPLSModelZeroShot を使用
#         import my_module4 as mm  # または import sys.modules[__name__]
        
#         self.model = mm.HierarchicalPLSModelZeroShot(
#             n_components=n_comp,
#             ridge_alpha=self.ridge_alpha
#         )
        
#         self.model.fit(
#             X_proc, 
#             y, 
#             subjects, 
#             static_features, 
#             feature_names=available_static
#         )
        
#         self.best_n_components_ = n_comp
        
#         return self
    
#     def predict(self, X, subjects):
#         """推論（未知被験者を自動検出してZero-shot適用）"""
        
#         # 利用可能な静的特徴量のみを抽出
#         available_static = [col for col in self.static_features_cols if col in X.columns]
#         static_features = X[available_static].values
        
#         # 動的特徴量
#         drop_cols = available_static + ['subject_id']
#         X_dynamic = X.drop(columns=drop_cols, errors='ignore')
#         X_proc = self._preprocess_internal(X_dynamic.values, is_training=False)
        
#         # 未知被験者のチェック
#         unique_test_subjects = np.unique(subjects)
        
#         for subj in unique_test_subjects:
#             if subj not in self.model.subject_gains:
#                 # 未知被験者を検出 → Zero-shot 適用
#                 print(f"      🆕 New subject {subj} → Zero-shot applied")
                
#                 mask = (subjects == subj)
#                 n_available = np.sum(mask)
#                 n_use = min(50, n_available)
                
#                 if n_use < 10:
#                     print(f"         ⚠️  Only {n_use} samples. Using default correction.")
#                     self.model.subject_gains[subj] = self.model.default_gain
#                     self.model.subject_offsets[subj] = self.model.default_offset
#                     continue
                
#                 # 最初のn_useサンプルから静的特徴量の平均
#                 static_feat_subset = static_features[mask][:n_use]
#                 static_feat_mean = static_feat_subset.mean(axis=0)
                
#                 # gain/offset を推定して登録
#                 static_feat_reshaped = static_feat_mean.reshape(1, -1)
#                 pred_gain = 1 # self.model.gain_predictor.predict(static_feat_reshaped)[0]
#                 pred_offset = self.model.offset_predictor.predict(static_feat_reshaped)[0]
                
#                 self.model.subject_gains[subj] = pred_gain
#                 self.model.subject_offsets[subj] = pred_offset
                
#                 print(f"         Gain: {pred_gain:.4f}, Offset: {pred_offset:.4f}")
        
#         # 通常の推論
#         return self.model.predict(X_proc, subjects)


# import numpy as np
# import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from abc import ABC, abstractmethod
# import my_module as mm_base  # 既存のmy_moduleを継承する場合

# # ==========================================
# # 🧠 Zero-shot 対応 Hierarchical PLS モデル
# # ==========================================

# class HierarchicalPLSModelZeroShot:
#     """
#     Zero-shot 対応版 Hierarchical PLS
#     未知の被験者に対して、静的特徴量(X_static)からGain/Offsetを推定して適用する。
#     """
#     def __init__(self, n_components=15, use_gain=True, use_offset=True, ridge_alpha=1.0):
#         self.n_components = n_components
#         self.use_gain = use_gain
#         self.use_offset = use_offset
#         self.ridge_alpha = ridge_alpha
        
#         # 内部モデル
#         self.global_pls = None
#         self.gain_predictor = None   # X_static -> Gain
#         self.offset_predictor = None # X_static -> Offset
        
#         # 記憶用パラメータ
#         self.subject_gains = {}
#         self.subject_offsets = {}
#         self.default_gain = 1.0
#         self.default_offset = 0.0
        
#         # Imputer
#         self.imputer = SimpleImputer(strategy='mean')

#     def fit(self, X_dynamic, y, subjects, X_static):
#         """
#         Args:
#             X_dynamic: 動的特徴量 (PLSへの入力)
#             y: 目的変数
#             subjects: 被験者ID配列
#             X_static: 静的特徴量配列 (Gain/Offset予測用)
#         """
#         if len(y.shape) > 1: y = y.flatten()
#         X_dynamic = self.imputer.fit_transform(X_dynamic)
        
#         # --------------------------------------------
#         # 1. Global PLS (ベースモデル) の学習
#         # --------------------------------------------
#         self.global_pls = PLSRegression(n_components=self.n_components, scale=False)
#         self.global_pls.fit(X_dynamic, y)
#         y_global = self.global_pls.predict(X_dynamic).flatten()
        
#         # --------------------------------------------
#         # 2. 各被験者の「正解 Gain / Offset」を算出
#         # --------------------------------------------
#         unique_subjects = np.unique(subjects)
#         X_static_train = []
#         y_gain_train = []
#         y_offset_train = []
        
#         all_gains = []
#         all_offsets = []

#         for subj in unique_subjects:
#             mask = (subjects == subj)
#             y_true_s = y[mask]
#             y_glob_s = y_global[mask]
            
#             # Gain/Offset計算 (最小二乗法)
#             if len(y_true_s) < 5 or np.std(y_glob_s) < 1e-6:
#                 gain, offset = 1.0, 0.0
#             else:
#                 if self.use_gain and self.use_offset:
#                     # y_true = gain * y_global + offset
#                     A = np.column_stack([y_glob_s, np.ones(len(y_glob_s))])
#                     params = np.linalg.lstsq(A, y_true_s, rcond=None)[0]
#                     gain, offset = params[0], params[1]
#                 elif self.use_gain:
#                     gain = np.sum(y_true_s * y_glob_s) / (np.sum(y_glob_s**2) + 1e-8)
#                     offset = 0.0
#                 elif self.use_offset:
#                     gain = 1.0
#                     offset = np.mean(y_true_s - y_glob_s)
#                 else:
#                     gain, offset = 1.0, 0.0
            
#             # 登録
#             self.subject_gains[subj] = gain
#             self.subject_offsets[subj] = offset
#             all_gains.append(gain)
#             all_offsets.append(offset)
            
#             # メタモデル学習用データ作成
#             # その被験者の静的特徴量（代表値）を取得
#             if X_static is not None:
#                 # この被験者のX_staticの平均ベクトル (1次元)
#                 subj_static_vec = X_static[mask].mean(axis=0)
#                 X_static_train.append(subj_static_vec)
#                 y_gain_train.append(gain)
#                 y_offset_train.append(offset)

#         # デフォルト値（中央値）
#         self.default_gain = np.median(all_gains) if all_gains else 1.0
#         self.default_offset = np.median(all_offsets) if all_offsets else 0.0
        
#         # --------------------------------------------
#         # 3. メタモデル (Gain/Offset Predictor) の学習
#         # --------------------------------------------
#         if X_static is not None and len(X_static_train) > 0:
#             X_static_train = np.array(X_static_train) # (n_subjects, n_static_feats)
#             y_gain_train = np.array(y_gain_train)
#             y_offset_train = np.array(y_offset_train)
            
#             self.gain_predictor = Ridge(alpha=self.ridge_alpha)
#             self.offset_predictor = Ridge(alpha=self.ridge_alpha)
            
#             self.gain_predictor.fit(X_static_train, y_gain_train)
#             self.offset_predictor.fit(X_static_train, y_offset_train)
            
#             print(f"   ✅ Meta-models trained on {len(unique_subjects)} subjects.")
            
#         return self

#     def predict(self, X_dynamic, subjects, X_static):
#         """
#         推論フェーズ
#         - 既知の被験者: 記憶しているGain/Offsetを使用
#         - 未知の被験者: X_static から Gain/Offset を予測して使用 (Zero-shot)
#         """
#         X_dynamic = self.imputer.transform(X_dynamic)
        
#         # 1. Global Prediction (ベースライン)
#         y_global = self.global_pls.predict(X_dynamic).flatten()
#         y_corrected = np.zeros_like(y_global)
        
#         if np.isscalar(subjects):
#             subjects = np.full(len(X_dynamic), subjects)
            
#         unique_test_subjects = np.unique(subjects)
        
#         for subj in unique_test_subjects:
#             mask = (subjects == subj)
            
#             # --- 補正係数の決定 ---
#             if subj in self.subject_gains:
#                 # A. 既知の被験者 -> 登録値を使用
#                 gain = self.subject_gains[subj]
#                 offset = self.subject_offsets[subj]
#             else:
#                 # B. 未知の被験者 -> Zero-shot 推定
#                 if self.gain_predictor is not None and X_static is not None:
#                     # その被験者の静的特徴量（最初の50サンプル程度から平均を取るのが安全）
#                     # ここでは入力された全期間の平均を使う（リアルタイムなら初期値を使う）
#                     subj_static_data = X_static[mask]
#                     if len(subj_static_data) > 0:
#                         # (n_features, )
#                         subj_static_vec = subj_static_data.mean(axis=0).reshape(1, -1)
                        
#                         gain = self.gain_predictor.predict(subj_static_vec)[0]
#                         offset = self.offset_predictor.predict(subj_static_vec)[0]
                        
#                         # ログ（任意）
#                         # print(f"   🆕 Zero-shot Subject {subj}: Gain={gain:.3f}, Offset={offset:.3f}")
#                     else:
#                         gain, offset = self.default_gain, self.default_offset
#                 else:
#                     # C. メタモデルがない場合 -> デフォルト値（中央値）
#                     gain, offset = self.default_gain, self.default_offset
            
#             # --- 補正適用 ---
#             y_corrected[mask] = gain * y_global[mask] + offset
            
#         return y_corrected


# # ==========================================
# # ♟️ Strategy Wrapper (Comparatorから呼ばれる部分)
# # ==========================================

# class HierarchicalPLSStrategyZeroShot(mm_base.BaseModelStrategy):
#     """
#     ModelComparatorで使えるようにラップしたクラス
#     """
#     def __init__(self, name, config, static_features_cols, ridge_alpha=1.0):
#         super().__init__(name, config) # Baseのinit
#         self.static_features_cols = static_features_cols
#         self.ridge_alpha = ridge_alpha
#         self.model = None # HierarchicalPLSModelZeroShot
        
#         # Baseクラスの設定読み込み
#         self.enable_scale_x = config.get('scale_x', False)
#         self.n_components_setting = config.get('n_components', 15)

#     def fit(self, X, y, subjects):
#         print(f"   🏃 Training {self.name} (Zero-shot enabled)...")
        
#         # 1. 静的特徴量の分離
#         available_static = [c for c in self.static_features_cols if c in X.columns]
#         X_static = X[available_static].values
        
#         # 2. 動的特徴量の分離 (静的列とIDを除く)
#         drop_cols = available_static + ['subject_id']
#         X_dynamic_df = X.drop(columns=drop_cols, errors='ignore')
#         X_dynamic = X_dynamic_df.values
        
#         # 3. 前処理 (Scalingなど)
#         X_dynamic_proc = self._preprocess_internal(X_dynamic, is_training=True)
        
#         # 4. 成分数決定 (Auto or Fixed)
#         if self.n_components_setting == 'auto':
#             # 簡易探索 (Global PLSで探索)
#             best_n = self._optimize_n_components(PLSRegression, X_dynamic_proc, y, subjects)
#             self.best_n_components_ = best_n
#             n_comp = best_n
#         else:
#             n_comp = int(self.n_components_setting)
#             self.best_n_components_ = n_comp

#         # 5. モデル本体の初期化と学習
#         self.model = HierarchicalPLSModelZeroShot(
#             n_components=n_comp,
#             ridge_alpha=self.ridge_alpha
#         )
        
#         self.model.fit(
#             X_dynamic=X_dynamic_proc,
#             y=y,
#             subjects=subjects,
#             X_static=X_static
#         )
#         return self

#     def predict(self, X, subjects):
#         # 1. 静的特徴量の分離
#         available_static = [c for c in self.static_features_cols if c in X.columns]
#         X_static = X[available_static].values
        
#         # 2. 動的特徴量の分離
#         drop_cols = available_static + ['subject_id']
#         X_dynamic_df = X.drop(columns=drop_cols, errors='ignore')
#         X_dynamic = X_dynamic_df.values
        
#         # 3. 前処理 (Scaling: Transformのみ)
#         X_dynamic_proc = self._preprocess_internal(X_dynamic, is_training=False)
        
#         # 4. 推論 (Zero-shotロジックは内部で自動分岐)
#         return self.model.predict(
#             X_dynamic=X_dynamic_proc,
#             subjects=subjects,
#             X_static=X_static
#         )
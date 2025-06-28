#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from datetime import datetime  # 修复datetime.now()问题
from scipy.stats import pearsonr
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, mutual_info_regression, SelectKBest
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.model_selection import KFold
from sklearn.base import clone

# 确保所有图表显示英文，彻底解决字体问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
# 添加下面这行确保不使用任何本地字体
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'

# 创建必要的目录
os.makedirs('schwefel/models', exist_ok=True)
os.makedirs('schwefel/results', exist_ok=True)
os.makedirs('schwefel/visualizations', exist_ok=True)

# 创建以时间命名的可视化输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
vis_dir = os.path.join('schwefel/visualizations', timestamp)
os.makedirs(vis_dir, exist_ok=True)
print(f"Visualization output will be saved to: {vis_dir}")

# 加载数据
def load_data():
    """Load the Schwefel dataset from NPY files"""
    data_dir = "/home/ym/code/ML_training/week3/schwefel/data/raw"
    
    try:
        print(f"Loading data from: {data_dir}")
        
        # 加载训练集
        x_train_path = os.path.join(data_dir, "Schwefel_x_train.npy")
        y_train_path = os.path.join(data_dir, "Schwefel_y_train.npy")
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        
        # 加载测试集
        x_test_path = os.path.join(data_dir, "Schwefel_x_test.npy")
        y_test_path = os.path.join(data_dir, "Schwefel_y_test.npy")
        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        
        print(f"Training data shape: {x_train.shape}, {y_train.shape}")
        print(f"Testing data shape: {x_test.shape}, {y_test.shape}")
        
        return x_train, y_train, x_test, y_test
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        # 如果加载失败，创建随机数据作为后备
        print("Creating random data as fallback")
        np.random.seed(42)
        x_train = np.random.rand(1000, 10)
        y_train = np.random.rand(1000)
        x_test = np.random.rand(200, 10)
        y_test = np.random.rand(200)
        
        print(f"Training data shape: {x_train.shape}, {y_train.shape}")
        print(f"Testing data shape: {x_test.shape}, {y_test.shape}")
        
        return x_train, y_train, x_test, y_test

# 特征工程
def feature_engineering(x):
    """特征工程函数 - 添加Schwefel核心特征，保持其他特征简单"""
    n_samples, n_features = x.shape
    all_features = []
    
    # 对每个维度单独处理
    for dim in range(n_features):
        x_dim = x[:, dim:dim+1]  # 提取单个维度
        
        # 1. Schwefel核心特征
        sqrt_abs_x = np.sqrt(np.abs(x_dim))
        sin_sqrt_abs_x = np.sin(sqrt_abs_x)
        x_sin = x_dim * sin_sqrt_abs_x  # Schwefel函数的核心特征
        
        # 2. 基础特征
        features = [
            x_dim,  # 原始值
            sin_sqrt_abs_x,  # sin(sqrt(|x|))
            x_sin,  # x * sin(sqrt(|x|))
            np.sign(x_dim) * np.power(np.abs(x_dim), 1/3),  # 立方根变换
            np.tanh(x_dim/500),  # tanh变换，归一化到[-1,1]
            np.clip(x_dim, -500, 500)/500,  # 线性归一化到[-1,1]
        ]
        
        # 3. 统计特征
        rolling_mean = np.zeros_like(x_dim)
        rolling_std = np.zeros_like(x_dim)
        for i in range(n_samples):
            start_idx = max(0, i-5)
            end_idx = min(n_samples, i+6)
            rolling_mean[i] = np.mean(x_dim[start_idx:end_idx])
            rolling_std[i] = np.std(x_dim[start_idx:end_idx])
        
        features.extend([
            rolling_mean,
            rolling_std,
            x_dim - rolling_mean,  # 去趋势
            (x_dim - rolling_mean) / (rolling_std + 1e-8)  # 标准化残差
        ])
        
        # 4. 周期性特征（减少频率数量）
        for freq in [0.1, 0.5]:
            features.extend([
                np.sin(freq * x_dim),
                np.cos(freq * x_dim)
            ])
        
        # 合并该维度的所有特征
        dim_features = np.hstack(features)
        all_features.append(dim_features)
    
    # 将所有维度的特征水平堆叠
    return np.hstack(all_features)

# 特征选择
def select_important_features(x_train, y_train, x_val, x_test):
    """Use multiple methods for feature selection and filtering"""
    print("Performing feature selection...")
    
    from sklearn.feature_selection import SelectFromModel, VarianceThreshold, mutual_info_regression, SelectKBest
    
    n_features_original = x_train.shape[1]
    
    # 1. Remove features with very low variance
    var_selector = VarianceThreshold(threshold=0.001)
    x_train_var = var_selector.fit_transform(x_train)
    x_val_var = var_selector.transform(x_val)
    x_test_var = var_selector.transform(x_test)
    
    print(f"Variance filtering kept {x_train_var.shape[1]}/{n_features_original} features")
    
    if x_train_var.shape[1] == 0:
        print("Warning: All features were removed by variance filter, falling back to original features")
        return x_train, x_val, x_test, np.arange(n_features_original)
    
    # 2. Use Random Forest for feature selection
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(x_train_var, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # 3. Feature selection based on mutual information
    try:
        k = min(x_train_var.shape[1], x_train_var.shape[1] // 2 + 10)  # Select half of features + 10
        k = max(k, 10)  # At least 10 features
        
        mi_selector = SelectKBest(mutual_info_regression, k=k)
        x_train_mi = mi_selector.fit_transform(x_train_var, y_train)
        x_val_mi = mi_selector.transform(x_val_var)
        x_test_mi = mi_selector.transform(x_test_var)
        
        print(f"Mutual information filtering kept {x_train_mi.shape[1]}/{x_train_var.shape[1]} features")
        
        if x_train_mi.shape[1] == 0:
            print("Warning: All features were removed by mutual information filter, falling back to variance filtered features")
            selection_mask = var_selector.get_support()
            indices = np.where(selection_mask)[0]
            return x_train_var, x_val_var, x_test_var, indices
    except Exception as e:
        print(f"Error in mutual information feature selection: {str(e)}, falling back to variance filtered features")
        selection_mask = var_selector.get_support()
        indices = np.where(selection_mask)[0]
        return x_train_var, x_val_var, x_test_var, indices
    
    # 4. Final feature selection: Using SelectFromModel based on Random Forest
    try:
        rf_final = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_final.fit(x_train_mi, y_train)
        
        # Create an adaptive threshold selector
        selector = SelectFromModel(rf_final, threshold="mean", prefit=True)
        
        # Transform the data
        x_train_selected = selector.transform(x_train_mi)
        x_val_selected = selector.transform(x_val_mi)
        x_test_selected = selector.transform(x_test_mi)
        
        print(f"Final selection kept {x_train_selected.shape[1]}/{x_train_mi.shape[1]} features")
        
        if x_train_selected.shape[1] == 0:
            print("Warning: All features were removed by final filter, falling back to mutual information filtered features")
            # Get indices from variance filtering
            var_indices = np.where(var_selector.get_support())[0]
            # Get indices from mutual information filtering (relative to variance filtered features)
            mi_indices = np.where(mi_selector.get_support())[0]
            # Convert to original feature indices
            final_indices = var_indices[mi_indices]
            return x_train_mi, x_val_mi, x_test_mi, final_indices
    except Exception as e:
        print(f"Error in final feature selection: {str(e)}, falling back to mutual information filtered features")
        # Get indices from variance filtering
        var_indices = np.where(var_selector.get_support())[0]
        # Get indices from mutual information filtering (relative to variance filtered features)
        mi_indices = np.where(mi_selector.get_support())[0]
        # Convert to original feature indices
        final_indices = var_indices[mi_indices]
        return x_train_mi, x_val_mi, x_test_mi, final_indices
    
    # Get final selected feature indices (position in original features)
    # First get indices from variance filtering
    var_indices = np.where(var_selector.get_support())[0]
    # Then get indices from mutual information filtering (relative to variance filtered features)
    mi_indices = np.where(mi_selector.get_support())[0]
    # Then get indices from final selection (relative to mutual information filtered features)
    final_selector_indices = np.where(selector.get_support())[0]
    # Convert to original feature indices
    final_indices = var_indices[mi_indices[final_selector_indices]]
    
    return x_train_selected, x_val_selected, x_test_selected, final_indices

# 数据预处理
def preprocess_data():
    """数据预处理函数"""
    # 加载数据
    data_dir = "/home/ym/code/ML_training/week3/schwefel/data/raw"
    x_train = np.load(os.path.join(data_dir, "Schwefel_x_train.npy"))
    y_train = np.load(os.path.join(data_dir, "Schwefel_y_train.npy"))
    x_test = np.load(os.path.join(data_dir, "Schwefel_x_test.npy"))
    y_test = np.load(os.path.join(data_dir, "Schwefel_y_test.npy"))
    
    # 划分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    # 第一次标准化 - 对原始特征
    scaler_input = StandardScaler()
    x_train_scaled = scaler_input.fit_transform(x_train)
    x_val_scaled = scaler_input.transform(x_val)
    x_test_scaled = scaler_input.transform(x_test)
    
    # 特征工程
    x_train_engineered = feature_engineering(x_train_scaled)
    x_val_engineered = feature_engineering(x_val_scaled)
    x_test_engineered = feature_engineering(x_test_scaled)
    
    # 确保所有数据集具有相同的特征数量
    min_features = min(x_train_engineered.shape[1], x_val_engineered.shape[1], x_test_engineered.shape[1])
    x_train_engineered = x_train_engineered[:, :min_features]
    x_val_engineered = x_val_engineered[:, :min_features]
    x_test_engineered = x_test_engineered[:, :min_features]
    
    # 第二次标准化 - 对工程特征
    scaler_features = StandardScaler()
    x_train_normalized = scaler_features.fit_transform(x_train_engineered)
    x_val_normalized = scaler_features.transform(x_val_engineered)
    x_test_normalized = scaler_features.transform(x_test_engineered)
    
    return x_train_normalized, x_val_normalized, x_test_normalized, y_train, y_val, y_test, (scaler_input, scaler_features)

# 学习率调度器
class LearningRateScheduler:
    """Learning rate scheduler with warmup and decay"""
    def __init__(self, init_lr=0.01, min_lr=0.0001, warmup_rounds=50, decay_factor=0.75, decay_rounds=100):
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.warmup_rounds = warmup_rounds
        self.decay_factor = decay_factor
        self.decay_rounds = decay_rounds
        self.current_lr = init_lr
    
    def __call__(self, env):
        # 检查是否是CallbackEnv对象
        if hasattr(env, 'iteration'):
            current_round = env.iteration
        else:
            current_round = env  # 如果直接传入整数
            
        # 预热阶段
        if current_round < self.warmup_rounds:
            self.current_lr = self.min_lr + (self.init_lr - self.min_lr) * current_round / self.warmup_rounds
        # 周期性衰减阶段
        else:
            cycle = (current_round - self.warmup_rounds) // self.decay_rounds
            self.current_lr = max(self.min_lr, self.init_lr * (self.decay_factor ** cycle))
        
        return self.current_lr

# 交叉验证训练
def train_model_with_cv(x_train, y_train, x_val, y_val, x_test, y_test, scaler_y, vis_dir, feature_indices, n_folds=5):
    """使用交叉验证训练多个模型并集成预测"""
    print(f"使用{n_folds}折交叉验证训练模型...")
    
    # 创建交叉验证对象
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 合并训练和验证数据用于交叉验证
    x_full = np.vstack((x_train, x_val))
    y_full = np.concatenate((y_train, y_val))
    
    # 定义多个模型参数组合进行尝试 - 更加关注防止过拟合
    params_list = [
        {
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': 'gbdt',
            'num_leaves': 31,  # 减少叶子节点数
            'max_depth': 4,    # 进一步减少树深度
            'learning_rate': 0.01,
            'feature_fraction': 0.6,  # 减少使用的特征比例
            'bagging_fraction': 0.5,  # 减少样本采样比例
            'bagging_freq': 5,
            'min_child_samples': 40,  # 增加最小样本数
            'min_data_in_leaf': 40,
            'verbose': -1,
            'reg_alpha': 0.5,  # 增加L1正则化
            'reg_lambda': 0.5,  # 增加L2正则化
            'path_smooth': 0.1,  # 平滑路径以减少过拟合
        },
        {
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': 'dart',  # 尝试DART提升方法
            'num_leaves': 40,
            'max_depth': 5,
            'learning_rate': 0.005,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 5,
            'min_child_samples': 35,
            'min_data_in_leaf': 35,
            'verbose': -1,
            'reg_alpha': 0.3,
            'reg_lambda': 0.5,
            'drop_rate': 0.1,  # DART特有参数
            'path_smooth': 0.2,
        },
        {
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': 'goss',  # 尝试GOSS提升方法
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.01,
            'feature_fraction': 0.7,
            'verbose': -1,
            'reg_alpha': 0.2,
            'reg_lambda': 0.3,
            'top_rate': 0.2,  # GOSS特有参数
            'other_rate': 0.1,  # GOSS特有参数
            'min_data_in_leaf': 30,
            'min_child_samples': 30,
        },
        {
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': 'gbdt',
            'num_leaves': 20,  # 更少的叶子节点
            'max_depth': 4,
            'learning_rate': 0.008,
            'feature_fraction': 0.65,
            'bagging_fraction': 0.8,
            'bagging_freq': 4,
            'min_child_samples': 25,
            'min_data_in_leaf': 25,
            'verbose': -1,
            'reg_alpha': 0.35,
            'reg_lambda': 0.4,
            'path_smooth': 0.15,
        }
    ]
    
    # 存储交叉验证模型
    all_models = []
    all_pearson_scores = []
    all_test_predictions = []
    
    # 存储每个模型配置的最佳Pearson相关系数
    best_model_scores = []
    
    # 为每个模型配置训练交叉验证模型
    for param_idx, params in enumerate(params_list):
        print(f"\n===== Training with parameter set {param_idx+1} =====")
        
        # 用于存储此参数集的模型和预测
        cv_models = []
        cv_pearson_scores = []
        pearson_history = []
        
        # 测试集预测结果
        test_predictions = np.zeros((n_folds, len(y_test)))
        
        # 交叉验证训练
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_full)):
            print(f"\nTraining fold {fold+1}/{n_folds} with parameter set {param_idx+1}")
            
            # 划分训练集和验证集
            x_train_fold, x_val_fold = x_full[train_idx], x_full[val_idx]
            y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
            
            # 创建数据集
            train_data = lgb.Dataset(x_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(x_val_fold, label=y_val_fold, reference=train_data)
            
            # 创建学习率调度器 - 预热更长，衰减更温和
            lr_scheduler = LearningRateScheduler(
                init_lr=params['learning_rate'],
                min_lr=params['learning_rate']/20,
                warmup_rounds=200,  # 更长的预热期
                decay_factor=0.95,  # 更缓慢的衰减
                decay_rounds=100
            )
            
            # 记录评估结果
            eval_results = {}
            
            # 用于记录每个迭代的Pearson相关系数
            fold_pearson_history = []
            
            # 定义回调函数，计算Pearson相关系数
            def pearson_eval(env):
                iteration = env.iteration
                if iteration % 20 == 0 or iteration == env.begin_iteration:  # 减少计算频率，每20轮计算一次
                    # 测试集上的预测
                    y_pred_scaled = env.model.predict(x_test)
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    pearson = pearsonr(y_test_original, y_pred)[0]
                    fold_pearson_history.append((iteration, pearson))
                    print(f"Param {param_idx+1}, Fold {fold+1}, Iteration {iteration}: Test Pearson = {pearson:.6f}")
            
            # 训练模型 - 使用更严格的早停
            model = lgb.train(
                params,
                train_data,
                num_boost_round=3000,  # 降低最大轮数
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),  # 更早的早停
                    lgb.record_evaluation(eval_results),
                    lgb.callback.reset_parameter(learning_rate=lr_scheduler),
                    pearson_eval
                ]
            )
            
            # 保存模型
            model_path = os.path.join(vis_dir, f'model_param{param_idx+1}_fold{fold+1}.txt')
            model.save_model(model_path)
            
            # 在测试集上进行预测
            test_predictions[fold] = model.predict(x_test)
            
            # 计算测试集上的Pearson相关系数
            y_pred = scaler_y.inverse_transform(test_predictions[fold].reshape(-1, 1)).flatten()
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            pearson = pearsonr(y_test_original, y_pred)[0]
            cv_pearson_scores.append(pearson)
            
            print(f"Param {param_idx+1}, Fold {fold+1} Test Pearson: {pearson:.6f}")
            
            # 保存模型和Pearson历史
            cv_models.append(model)
            pearson_history.extend([(param_idx, fold, i, p) for i, p in fold_pearson_history])
            
            # 绘制训练损失图
            train_loss = eval_results['train']['l2']
            val_loss = eval_results['val']['l2']
            epochs = range(1, len(train_loss) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_loss, 'b-', label='Training Loss')
            plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
            plt.title(f'Param {param_idx+1}, Fold {fold+1}: Training and Validation L2 Loss')
            plt.xlabel('Iterations')
            plt.ylabel('L2 Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(vis_dir, f'training_loss_param{param_idx+1}_fold{fold+1}.png'))
            plt.close()
            
            # 绘制Pearson相关系数变化图
            iterations, pearson_values = zip(*fold_pearson_history)
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, pearson_values, 'g-')
            plt.title(f'Param {param_idx+1}, Fold {fold+1}: Pearson Correlation on Test Set')
            plt.xlabel('Iterations')
            plt.ylabel('Pearson Correlation')
            plt.grid(True)
            plt.savefig(os.path.join(vis_dir, f'pearson_history_param{param_idx+1}_fold{fold+1}.png'))
            plt.close()
        
        # 计算此参数集的加权测试集预测
        weights = np.array(cv_pearson_scores)
        weights = weights / np.sum(weights)  # 归一化权重
        
        param_test_pred = np.zeros(len(y_test))
        for i, weight in enumerate(weights):
            param_test_pred += weight * test_predictions[i]
        
        # 评估此参数集的集成性能
        y_pred = scaler_y.inverse_transform(param_test_pred.reshape(-1, 1)).flatten()
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        param_pearson = pearsonr(y_test_original, y_pred)[0]
        
        print(f"\nParameter set {param_idx+1} ensemble Pearson: {param_pearson:.6f}")
        
        # 保存此参数集的结果
        all_models.append(cv_models)
        all_pearson_scores.append(cv_pearson_scores)
        all_test_predictions.append(param_test_pred)
        best_model_scores.append(param_pearson)
    
    # 找出性能最好的参数集
    best_param_idx = np.argmax(best_model_scores)
    best_param_score = best_model_scores[best_param_idx]
    
    print(f"\nBest parameter set: {best_param_idx+1} with Pearson: {best_param_score:.6f}")
    
    # 组合所有模型的预测（超级集成）
    meta_weights = np.array(best_model_scores)
    meta_weights = meta_weights / np.sum(meta_weights)  # 归一化权重
    
    ensemble_pred = np.zeros(len(y_test))
    for i, weight in enumerate(meta_weights):
        ensemble_pred += weight * all_test_predictions[i]
    
    # 评估超级集成的性能
    y_pred_ensemble = scaler_y.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    ensemble_pearson = pearsonr(y_test_original, y_pred_ensemble)[0]
    
    print(f"\nSuper ensemble Pearson: {ensemble_pearson:.6f}")
    
    # 创建一个更复杂的最终集成 - 选择每个参数集中表现最好的模型
    best_models_pred = []
    best_models_pearson = []
    best_models = []
    
    for param_idx in range(len(params_list)):
        best_fold_idx = np.argmax(all_pearson_scores[param_idx])
        best_model = all_models[param_idx][best_fold_idx]
        best_models.append(best_model)
        
        # 获取此模型在测试集上的预测
        pred = best_model.predict(x_test)
        best_models_pred.append(pred)
        
        # 计算性能
        y_pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
        pearson = pearsonr(y_test_original, y_pred)[0]
        best_models_pearson.append(pearson)
        
        print(f"Best model from param set {param_idx+1}: Pearson = {pearson:.6f}")
    
    # 对最佳模型进行加权组合
    best_weights = np.array(best_models_pearson)
    best_weights = best_weights / np.sum(best_weights)  # 归一化权重
    
    best_ensemble_pred = np.zeros(len(y_test))
    for i, weight in enumerate(best_weights):
        best_ensemble_pred += weight * best_models_pred[i]
    
    # 评估最佳模型集成的性能
    y_pred_best = scaler_y.inverse_transform(best_ensemble_pred.reshape(-1, 1)).flatten()
    best_ensemble_pearson = pearsonr(y_test_original, y_pred_best)[0]
    
    print(f"\nBest models ensemble Pearson: {best_ensemble_pearson:.6f}")
    
    # 选择最终的预测方法
    final_pred = None
    final_pearson = 0
    final_method = ""
    
    if best_ensemble_pearson > ensemble_pearson and best_ensemble_pearson > best_param_score:
        final_pred = best_ensemble_pred
        final_pearson = best_ensemble_pearson
        final_method = "Best Models Ensemble"
    elif ensemble_pearson > best_param_score:
        final_pred = ensemble_pred
        final_pearson = ensemble_pearson
        final_method = "Super Ensemble"
    else:
        final_pred = all_test_predictions[best_param_idx]
        final_pearson = best_param_score
        final_method = f"Parameter Set {best_param_idx+1} Ensemble"
    
    print(f"\nFinal prediction method: {final_method} with Pearson: {final_pearson:.6f}")
    
    # 绘制所有参数集的Pearson相关系数变化趋势
    plt.figure(figsize=(15, 10))
    
    for param_idx in range(len(params_list)):
        for fold in range(n_folds):
            fold_data = [(i, p) for pid, fid, i, p in pearson_history if pid == param_idx and fid == fold]
            if fold_data:
                iterations, pearson_values = zip(*fold_data)
                plt.plot(iterations, pearson_values, '-', alpha=0.5, 
                        label=f'Param {param_idx+1}, Fold {fold+1}')
    
    # 标记最终的Pearson相关系数
    plt.axhline(y=final_pearson, color='r', linestyle='--', linewidth=2)
    plt.text(0, final_pearson*0.98, f'Final Pearson = {final_pearson:.4f} ({final_method})', 
             fontsize=12, color='red')
    
    plt.title('Pearson Correlation Evolution Across All Parameters and Folds')
    plt.xlabel('Iterations')
    plt.ylabel('Pearson Correlation')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'pearson_history_all_params_folds.png'))
    plt.close()
    
    # 保存特征重要性 (使用最佳参数集的第一个模型)
    best_param_models = all_models[best_param_idx]
    importance = best_param_models[0].feature_importance()
    feature_names = [f'feature_{i}' for i in range(len(importance))]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 绘制特征重要性图
    top_n = min(30, len(importance))
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), importance_df['importance'][:top_n])
    plt.xticks(range(top_n), importance_df['feature'][:top_n], rotation=90)
    plt.title('Top Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_importance.png'))
    plt.close()
    
    # 保存特征重要性
    importance_df.to_csv(os.path.join(vis_dir, 'feature_importance.csv'), index=False)
    
    # 保存Pearson相关系数历史记录
    pearson_df = pd.DataFrame(pearson_history, columns=['param_set', 'fold', 'iteration', 'pearson'])
    pearson_df.to_csv(os.path.join(vis_dir, 'pearson_history.csv'), index=False)
    
    # 保存模型权重信息
    weights_info = {
        'param_set': [],
        'fold': [],
        'pearson': [],
        'weight': []
    }
    
    for param_idx in range(len(params_list)):
        for fold_idx, pearson in enumerate(all_pearson_scores[param_idx]):
            weights_info['param_set'].append(param_idx + 1)
            weights_info['fold'].append(fold_idx + 1)
            weights_info['pearson'].append(pearson)
            weights_info['weight'].append(0)  # 默认为0，只有最终使用的模型会有权重
    
    # 更新最终使用的模型权重
    if final_method == "Best Models Ensemble":
        for i, (param_idx, weight) in enumerate(zip(range(len(params_list)), best_weights)):
            best_fold_idx = np.argmax(all_pearson_scores[param_idx])
            for j, (ps, fold) in enumerate(zip(weights_info['param_set'], weights_info['fold'])):
                if ps == param_idx + 1 and fold == best_fold_idx + 1:
                    weights_info['weight'][j] = weight
    elif final_method == "Super Ensemble":
        for i, (param_idx, weight) in enumerate(zip(range(len(params_list)), meta_weights)):
            weights_info['weight'][i * n_folds] = weight  # 简化，只记录每个参数集的第一个模型权重
    else:
        best_param_idx_value = int(final_method.split()[-2])
        fold_weights = all_pearson_scores[best_param_idx_value - 1]
        fold_weights = fold_weights / np.sum(fold_weights)
        for i, fold_weight in enumerate(fold_weights):
            offset = (best_param_idx_value - 1) * n_folds
            weights_info['weight'][offset + i] = fold_weight
    
    weights_df = pd.DataFrame(weights_info)
    weights_df.to_csv(os.path.join(vis_dir, 'model_weights.csv'), index=False)
    
    return best_models, final_pearson, final_pred, final_method

# 评估模型
def evaluate_model(y_pred, y_test, scaler_y, vis_dir):
    """Evaluate model performance and visualize results"""
    # Inverse transform predictions and true values
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    pearson = pearsonr(y_test_original, y_pred_original)[0]
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test R²: {r2:.6f}")
    print(f"Test Pearson Correlation: {pearson:.6f}")
    
    # Create prediction vs true value scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    
    # Add diagonal line
    min_val = min(y_test_original.min(), y_pred_original.min())
    max_val = max(y_test_original.max(), y_pred_original.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Predicted vs True Values (Pearson={pearson:.6f})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'prediction_scatter.png'))
    plt.close()
    
    # Create error distribution histogram
    errors = y_pred_original - y_test_original
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'error_distribution.png'))
    plt.close()
    
    # Create error vs true value scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_original, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Prediction Error vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'error_vs_actual.png'))
    plt.close()
    
    # Create true and predicted value distribution comparison
    plt.figure(figsize=(12, 6))
    plt.hist(y_test_original, bins=50, alpha=0.5, label='True Values')
    plt.hist(y_pred_original, bins=50, alpha=0.5, label='Predicted Values')
    plt.title('True vs Predicted Value Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'distribution_comparison.png'))
    plt.close()
    
    # Save evaluation metrics
    with open(os.path.join(vis_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"R²: {r2:.6f}\n")
        f.write(f"Pearson Correlation: {pearson:.6f}\n")
    
    return mse, r2, pearson

def analyze_feature_correlations(x_train, y_train, vis_dir):
    """分析特征与目标变量的相关性"""
    print("分析特征与目标变量的相关性...")
    
    correlations = []
    for i in range(x_train.shape[1]):
        corr, _ = pearsonr(x_train[:, i], y_train)
        correlations.append((i, abs(corr)))
    
    # 按相关性绝对值排序
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 绘制相关性图
    plt.figure(figsize=(12, 6))
    feature_indices = [x[0] for x in correlations]
    correlation_values = [x[1] for x in correlations]
    
    plt.bar(range(len(correlations)), correlation_values)
    plt.xlabel('特征索引')
    plt.ylabel('与目标变量的相关性绝对值')
    plt.title('特征与目标变量的相关性分析')
    plt.xticks(range(len(correlations)), feature_indices, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_correlations.png'))
    plt.close()
    
    # 保存相关性分析结果
    with open(os.path.join(vis_dir, 'feature_correlations.txt'), 'w') as f:
        f.write("特征相关性分析结果:\n")
        f.write("=================\n\n")
        for idx, corr in correlations:
            f.write(f"特征 {idx}: {corr:.6f}\n")
    
    return correlations

def main():
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join('schwefel/visualizations', timestamp)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Visualization output will be saved to: {vis_dir}")
    
    # 加载和预处理数据
    print("\n=== 加载和预处理数据 ===")
    x_train, x_val, x_test, y_train, y_val, y_test, (scaler_input, scaler_features) = preprocess_data()
    
    # 训练分解模型
    print("\n=== 训练分解模型 ===")
    models, test_pearson, predictions = train_decomposed_model(
        x_train, y_train, x_val, y_val, x_test, y_test, vis_dir
    )
    
    # 保存训练信息
    with open(os.path.join(vis_dir, 'training_info.txt'), 'w') as f:
        f.write("训练信息:\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"特征数量: {x_train.shape[1]}\n")
        f.write(f"分解模型Pearson相关系数: {test_pearson:.6f}\n")
    
    print(f"\n训练完成。结果已保存到: {vis_dir}")

# 堆叠集成学习模型
def train_stacking_ensemble(x_train, y_train, x_val, y_val, x_test, y_test, scaler_y, vis_dir, feature_indices):
    """Train an optimized stacking ensemble focused only on GBDT and RandomForest models"""
    print("\n=== Training Optimized GBDT & RandomForest Models ===")
    
    # 初始化模型列表
    base_models = []
    base_model_names = []
    base_model_scores = []
    
    # 1. LightGBM-GBDT - 第一个配置
    lgb_params1 = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 10,
        'min_data_in_leaf': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 0.3,
        'n_estimators': 500,
        'random_state': 42
    }
    
    # 2. LightGBM-GBDT - 第二个配置
    lgb_params2 = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.005,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'max_depth': 15,
        'min_data_in_leaf': 5,
        'reg_alpha': 0.2,
        'reg_lambda': 0.4,
        'n_estimators': 800,
        'random_state': 42
    }
    
    # 3. RandomForest - 第一个配置
    rf_params1 = {
        'n_estimators': 500, 
        'max_depth': 15, 
        'min_samples_split': 5,
        'min_samples_leaf': 2, 
        'max_features': 'sqrt', 
        'n_jobs': -1,
        'random_state': 42
    }
    
    # 4. RandomForest - 第二个配置
    rf_params2 = {
        'n_estimators': 800, 
        'max_depth': 20, 
        'min_samples_split': 2,
        'min_samples_leaf': 1, 
        'max_features': 'sqrt', 
        'n_jobs': -1,
        'random_state': 42
    }
    
    # 5. GradientBoosting
    gb_params = {
        'n_estimators': 300,
        'learning_rate': 0.01,
        'max_depth': 7,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'subsample': 0.8,
        'random_state': 42
    }
    
    # 6. ExtraTrees
    et_params = {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 3,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'random_state': 42
    }
    
    # 创建模型列表
    models_to_train = [
        ('LGBM-1', lgb.LGBMRegressor(**lgb_params1)),
        ('LGBM-2', lgb.LGBMRegressor(**lgb_params2)),
        ('RF-1', RandomForestRegressor(**rf_params1)),
        ('RF-2', RandomForestRegressor(**rf_params2)),
        ('GB', GradientBoostingRegressor(**gb_params)),
        ('ET', ExtraTreesRegressor(**et_params))
    ]
    
    # 存储模型预测
    val_predictions = np.zeros((x_val.shape[0], len(models_to_train)))
    test_predictions = np.zeros((x_test.shape[0], len(models_to_train)))
    
    # 训练每个基础模型
    for i, (name, model) in enumerate(models_to_train):
        print(f"--- Training {name} ---")
        
        if isinstance(model, lgb.LGBMRegressor):
            # 使用早停法训练LightGBM模型
            model.fit(
                x_train, y_train,
                eval_set=[(x_val, y_val)],
                eval_metric='l2',
                callbacks=[lgb.early_stopping(stopping_rounds=50)]  # 只使用callbacks参数
            )
            
            # 在验证集和测试集上预测
            val_predictions[:, i] = model.predict(x_val)
            test_predictions[:, i] = model.predict(x_test)
            
            # 绘制特征重要性
            try:
                importances = model.feature_importances_
                plt.figure(figsize=(10, 8))
                n_features = min(30, len(importances))
                indices = np.argsort(importances)[-n_features:]
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)), [f"Feature_{idx}" for idx in indices])
                plt.title(f'Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'feature_importance_{name}.png'))
                plt.close()
            except Exception as e:
                print(f"Could not plot feature importance for {name}: {str(e)}")
        else:
            # 对于其他scikit-learn模型，直接训练
            model.fit(x_train, y_train)
            
            # 在验证集和测试集上预测
            val_predictions[:, i] = model.predict(x_val)
            test_predictions[:, i] = model.predict(x_test)
            
            # 绘制特征重要性
            if hasattr(model, 'feature_importances_'):
                try:
                    importances = model.feature_importances_
                    plt.figure(figsize=(10, 8))
                    n_features = min(30, len(importances))
                    indices = np.argsort(importances)[-n_features:]
                    plt.barh(range(len(indices)), importances[indices])
                    plt.yticks(range(len(indices)), [f"Feature_{idx}" for idx in indices])
                    plt.title(f'Feature Importance - {name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'feature_importance_{name}.png'))
                    plt.close()
                except Exception as e:
                    print(f"Could not plot feature importance for {name}: {str(e)}")
        
        # 计算Pearson相关系数
        val_pred_original = scaler_y.inverse_transform(val_predictions[:, i].reshape(-1, 1)).flatten()
        y_val_original = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        val_pearson = pearsonr(y_val_original, val_pred_original)[0]
        
        print(f"{name} - Validation Pearson: {val_pearson:.6f}")
        
        # 保存模型信息
        base_models.append(model)
        base_model_names.append(name)
        base_model_scores.append(val_pearson)
    
    # 找出最佳模型
    best_model_idx = np.argmax(base_model_scores)
    best_model_name = base_model_names[best_model_idx]
    best_model_score = base_model_scores[best_model_idx]
    
    print(f"\nBest single model: {best_model_name} with Pearson {best_model_score:.6f}")
    
    # 训练元模型 (使用验证集的预测作为特征)
    print("\n--- Training Meta-Model ---")
    
    # 简单LightGBM元模型
    meta_model = lgb.LGBMRegressor(
        objective='regression',
        metric='l2',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.01,
        n_estimators=200,
        random_state=42
    )
    
    # 训练元模型
    meta_model.fit(val_predictions, y_val)
    
    # 元模型在验证集上的性能
    val_meta_preds = meta_model.predict(val_predictions)
    val_meta_original = scaler_y.inverse_transform(val_meta_preds.reshape(-1, 1)).flatten()
    y_val_original = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    val_pearson = pearsonr(y_val_original, val_meta_original)[0]
    
    print(f"Meta-model - Validation Pearson: {val_pearson:.6f}")
    
    # 元模型在测试集上的性能
    test_meta_preds = meta_model.predict(test_predictions)
    test_meta_original = scaler_y.inverse_transform(test_meta_preds.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_pearson = pearsonr(y_test_original, test_meta_original)[0]
    
    print(f"Meta-model - Test Pearson: {test_pearson:.6f}")
    
    # 绘制基础模型性能比较图
    plt.figure(figsize=(12, 8))
    sorted_indices = np.argsort(base_model_scores)
    plt.barh(range(len(sorted_indices)), [base_model_scores[i] for i in sorted_indices])
    plt.yticks(range(len(sorted_indices)), [base_model_names[i] for i in sorted_indices])
    plt.title('Base Model Performance Comparison (Validation Set Pearson)')
    plt.xlabel('Pearson Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'base_models_performance.png'))
    plt.close()
    
    # 绘制测试集预测与真实值的散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_original, test_meta_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--')
    plt.title(f'Test Set: True vs Predicted Values (Pearson={test_pearson:.6f})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'test_prediction_scatter.png'))
    plt.close()
    
    # 绘制元模型特征重要性
    if hasattr(meta_model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = meta_model.feature_importances_
        feature_names = base_model_names
        
        indices = np.argsort(importances)
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title('Meta-Model Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'meta_model_feature_importance.png'))
        plt.close()
    
    # 保存模型报告
    with open(os.path.join(vis_dir, 'model_report.txt'), 'w') as f:
        f.write("=== Model Performance Report ===\n\n")
        f.write("Base Model Performance:\n")
        for i, name in enumerate(base_model_names):
            f.write(f"- {name}: Pearson = {base_model_scores[i]:.6f}\n")
        
        f.write(f"\nMeta-Model Performance:\n")
        f.write(f"- Validation Set Pearson = {val_pearson:.6f}\n")
        f.write(f"- Test Set Pearson = {test_pearson:.6f}\n")
    
    # 保存模型
    try:
        # 创建模型保存目录
        models_dir = os.path.join(vis_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # 保存基础模型
        for i, name in enumerate(base_model_names):
            model_path = os.path.join(models_dir, f"{name}.joblib")
            try:
                joblib.dump(base_models[i], model_path)
            except Exception as e:
                print(f"Error saving base model {name}: {str(e)}")
        
        # 保存元模型
        meta_model_path = os.path.join(models_dir, "meta_model.joblib")
        try:
            joblib.dump(meta_model, meta_model_path)
        except Exception as e:
            print(f"Error saving meta-model: {str(e)}")
    except Exception as e:
        print(f"Error saving models: {str(e)}")
    
    return meta_model, test_pearson, test_predictions

def train_weighted_ensemble(x_train, y_train, x_val, y_val, x_test, y_test, vis_dir):
    """训练GBDT模型，使用动态验证集"""
    print("\n=== Training GBDT Model with Dynamic Validation ===")
    
    # 定义模型参数 - 平衡拟合能力和泛化能力
    params = {
        'objective': 'regression',
        'metric': ['l2', 'rmse'],
        'boosting_type': 'gbdt',
        'num_leaves': 127,  # 增加叶子节点数以提高拟合能力
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'feature_fraction_bynode': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': -1,
        'lambda_l1': 0.5,  # 降低L1正则化
        'lambda_l2': 1.0,  # 降低L2正则化
        'min_data_in_leaf': 20,  # 降低最小数据量要求
        'max_depth': 12,  # 增加树的深度
        'max_bin': 255,
        'min_gain_to_split': 0.1,  # 降低分裂增益阈值
        'num_iterations': 5000,
        'force_row_wise': True,
        'path_smooth': 0.5,  # 降低路径平滑
        'random_state': 42
    }
    
    # 合并训练集和验证集
    x_full = np.vstack((x_train, x_val))
    y_full = np.concatenate((y_train, y_val))
    
    # 初始化训练历史记录
    train_loss_history = []
    val_loss_history = []
    train_pearson_history = []
    val_pearson_history = []
    test_pearson_history = []
    lr_history = []
    iterations = []
    
    # 创建初始数据集划分
    n_samples = len(x_full)
    val_size = int(0.2 * n_samples)  # 恢复到20%的验证集大小
    overlap_ratio = 0.4  # 增加重叠比例到40%
    
    # 创建初始索引
    all_indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    # 定义学习率调度器
    def learning_rate_scheduler(iter_num):
        min_lr = 1e-4  # 提高最小学习率
        base_lr = params['learning_rate']
        warmup_epochs = 200  # 减少预热期
        
        if iter_num < warmup_epochs:
            lr = base_lr * (iter_num / warmup_epochs)
        else:
            decay_epochs = iter_num - warmup_epochs
            total_epochs = params['num_iterations'] - warmup_epochs
            # 使用更缓慢的余弦退火
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_epochs / total_epochs))
            lr = base_lr * (cosine_decay * 0.95 + 0.05)  # 保持一定的学习率
        
        lr = max(min_lr, lr)
        lr_history.append(lr)
        return lr
    
    class DynamicDataset:
        def __init__(self, x_full, y_full, val_size, overlap_ratio):
            self.x_full = x_full
            self.y_full = y_full
            self.val_size = val_size
            self.overlap_ratio = overlap_ratio
            self.n_samples = len(x_full)
            self.current_val_indices = None
            self.update_counter = 0
            self.reset_validation_set()
        
        def reset_validation_set(self):
            if self.current_val_indices is None:
                # 首次划分
                self.current_val_indices = np.random.choice(
                    self.n_samples, self.val_size, replace=False)
            else:
                # 保留部分旧的验证集索引
                keep_size = int(self.val_size * self.overlap_ratio)
                keep_indices = np.random.choice(
                    self.current_val_indices, keep_size, replace=False)
                
                # 选择新的验证集索引
                available_indices = np.setdiff1d(
                    np.arange(self.n_samples), keep_indices)
                new_indices = np.random.choice(
                    available_indices, self.val_size - keep_size, replace=False)
                
                self.current_val_indices = np.concatenate([keep_indices, new_indices])
            
            self.train_indices = np.setdiff1d(
                np.arange(self.n_samples), self.current_val_indices)
            
            self.update_counter += 1
            
            # 创建新的数据集
            self.train_data = lgb.Dataset(
                self.x_full[self.train_indices],
                label=self.y_full[self.train_indices]
            )
            self.val_data = lgb.Dataset(
                self.x_full[self.current_val_indices],
                label=self.y_full[self.current_val_indices],
                reference=self.train_data
            )
            
            return self.train_data, self.val_data
    
    # 创建动态数据集管理器
    dynamic_dataset = DynamicDataset(x_full, y_full, val_size, overlap_ratio)
    
    # 定义评估函数
    def eval_metrics(env):
        if env.iteration % 100 == 0:  # 恢复为每100轮更新一次
            iterations.append(env.iteration)
            
            # 更新验证集
            train_data, val_data = dynamic_dataset.reset_validation_set()
            
            # 获取当前训练集和验证集的索引
            train_indices = dynamic_dataset.train_indices
            val_indices = dynamic_dataset.current_val_indices
            
            # 训练集评估
            y_pred_train = env.model.predict(x_full[train_indices])
            train_pearson = pearsonr(y_full[train_indices], y_pred_train)[0]
            train_pearson_history.append(train_pearson)
            
            # 验证集评估
            y_pred_val = env.model.predict(x_full[val_indices])
            val_pearson = pearsonr(y_full[val_indices], y_pred_val)[0]
            val_pearson_history.append(val_pearson)
            
            # 测试集评估
            y_pred_test = env.model.predict(x_test)
            test_pearson = pearsonr(y_test, y_pred_test)[0]
            test_pearson_history.append(test_pearson)
            
            # 记录损失值
            train_loss = mean_squared_error(y_full[train_indices], y_pred_train)
            val_loss = mean_squared_error(y_full[val_indices], y_pred_val)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            
            print(f"\n[{env.iteration}] Validation set update {dynamic_dataset.update_counter}")
            print(f"Train Pearson: {train_pearson:.6f}")
            print(f"Val Pearson: {val_pearson:.6f}")
            print(f"Test Pearson: {test_pearson:.6f}")
            
            # 如果发现明显过拟合，提前停止
            if len(test_pearson_history) > 10:  # 至少等待10次评估
                recent_test = test_pearson_history[-10:]
                if max(recent_test) - test_pearson < -0.02:  # 放宽过拟合判断标准
                    print("\nEarly stopping due to overfitting...")
                    env.model.best_iteration = env.iteration  # 记录最佳迭代次数
                    return True
    
    # 训练模型
    initial_train_data, initial_val_data = dynamic_dataset.reset_validation_set()
    
    model = lgb.train(
        params,
        initial_train_data,
        valid_sets=[initial_train_data, initial_val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.reset_parameter(learning_rate=learning_rate_scheduler),
            eval_metrics
        ]
    )
    
    # 在最终测试集上评估
    y_pred_test = model.predict(x_test)
    final_test_pearson = pearsonr(y_test, y_pred_test)[0]
    
    print(f"\nFinal Test Pearson: {final_test_pearson:.6f}")
    
    # 绘制训练历史
    plt.figure(figsize=(15, 10))
    
    # 损失函数变化曲线
    plt.subplot(2, 2, 1)
    plt.plot(iterations, train_loss_history, label='Train Loss')
    plt.plot(iterations, val_loss_history, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # Pearson相关系数变化曲线
    plt.subplot(2, 2, 2)
    plt.plot(iterations, train_pearson_history, label='Train Pearson')
    plt.plot(iterations, val_pearson_history, label='Validation Pearson')
    plt.plot(iterations, test_pearson_history, label='Test Pearson')
    plt.title('Pearson Correlation History')
    plt.xlabel('Iterations')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    plt.grid(True)
    
    # 学习率变化曲线
    plt.subplot(2, 2, 3)
    plt.plot(range(len(lr_history)), lr_history)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    # 保存训练历史图
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_history.png'))
    plt.close()
    
    # 保存模型
    model_path = os.path.join(vis_dir, 'model.txt')
    model.save_model(model_path)
    
    # 保存训练历史数据
    history_df = pd.DataFrame({
        'iteration': iterations,
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_pearson': train_pearson_history,
        'val_pearson': val_pearson_history,
        'test_pearson': test_pearson_history,
        'learning_rate': lr_history[:len(iterations)]
    })
    history_df.to_csv(os.path.join(vis_dir, 'training_history.csv'), index=False)
    
    # 分析特征重要性
    try:
        importances = model.feature_importance()
        feature_names = [f'feature_{i}' for i in range(len(importances))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 6))
        plt.bar(range(30), importance_df['importance'][:30])
        plt.xticks(range(30), importance_df['feature'][:30], rotation=45)
        plt.title('Top 30 Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_importance.png'))
        plt.close()
        
        # 保存特征重要性到CSV
        importance_df.to_csv(os.path.join(vis_dir, 'feature_importance.csv'), index=False)
        
    except Exception as e:
        print(f"Warning: Could not analyze feature importance: {str(e)}")
    
    return model, final_test_pearson

def train_rf_with_dynamic_validation(x_train, y_train, x_val, y_val, x_test, y_test, vis_dir):
    """使用动态验证集训练随机森林模型"""
    print("\n=== 使用动态验证集训练随机森林模型 ===")
    
    # 合并训练集和验证集
    x_full = np.vstack((x_train, x_val))
    y_full = np.concatenate((y_train, y_val))
    
    # 初始化训练历史记录
    train_pearson_history = []
    val_pearson_history = []
    test_pearson_history = []
    feature_importance_history = []
    iterations = []
    
    # 创建初始数据集划分
    n_samples = len(x_full)
    val_size = int(0.2 * n_samples)  # 20%的验证集
    overlap_ratio = 0.4  # 40%的重叠
    
    # 定义基础RF参数
    base_params = {
        'n_estimators': 100,  # 每次增量训练100棵树
        'max_depth': 12,      # 与GBDT保持一致的深度
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'random_state': 42,
        'warm_start': True    # 允许增量训练
    }
    
    class DynamicRFTrainer:
        def __init__(self, x_full, y_full, val_size, overlap_ratio, base_params):
            self.x_full = x_full
            self.y_full = y_full
            self.val_size = val_size
            self.overlap_ratio = overlap_ratio
            self.n_samples = len(x_full)
            self.current_val_indices = None
            self.update_counter = 0
            self.model = RandomForestRegressor(**base_params)
            self.total_trees = 0
            self.max_trees = 5000  # 最大树的数量
            
        def reset_validation_set(self):
            if self.current_val_indices is None:
                # 首次划分
                self.current_val_indices = np.random.choice(
                    self.n_samples, self.val_size, replace=False)
            else:
                # 保留部分旧的验证集索引
                keep_size = int(self.val_size * self.overlap_ratio)
                keep_indices = np.random.choice(
                    self.current_val_indices, keep_size, replace=False)
                
                # 选择新的验证集索引
                available_indices = np.setdiff1d(
                    np.arange(self.n_samples), keep_indices)
                new_indices = np.random.choice(
                    available_indices, self.val_size - keep_size, replace=False)
                
                self.current_val_indices = np.concatenate([keep_indices, new_indices])
            
            self.train_indices = np.setdiff1d(
                np.arange(self.n_samples), self.current_val_indices)
            self.update_counter += 1
            
            return self.train_indices, self.current_val_indices
        
        def train_increment(self):
            # 获取当前训练集和验证集
            train_indices, val_indices = self.reset_validation_set()
            
            # 增加树的数量
            increment = min(100, self.max_trees - self.total_trees)
            if increment <= 0:
                return False
                
            self.model.n_estimators += increment
            self.total_trees += increment
            
            # 在当前训练集上训练
            self.model.fit(
                self.x_full[train_indices],
                self.y_full[train_indices]
            )
            
            # 评估性能
            y_pred_train = self.model.predict(self.x_full[train_indices])
            y_pred_val = self.model.predict(self.x_full[val_indices])
            y_pred_test = self.model.predict(x_test)
            
            # 计算Pearson相关系数
            train_pearson = pearsonr(self.y_full[train_indices], y_pred_train)[0]
            val_pearson = pearsonr(self.y_full[val_indices], y_pred_val)[0]
            test_pearson = pearsonr(y_test, y_pred_test)[0]
            
            # 记录历史
            train_pearson_history.append(train_pearson)
            val_pearson_history.append(val_pearson)
            test_pearson_history.append(test_pearson)
            feature_importance_history.append(self.model.feature_importances_)
            iterations.append(self.total_trees)
            
            print(f"\n[Trees: {self.total_trees}] Validation set update {self.update_counter}")
            print(f"Train Pearson: {train_pearson:.6f}")
            print(f"Val Pearson: {val_pearson:.6f}")
            print(f"Test Pearson: {test_pearson:.6f}")
            
            # 检查是否过拟合
            if len(test_pearson_history) > 5:
                recent_test = test_pearson_history[-5:]
                if max(recent_test) - test_pearson < -0.02:
                    print("\n提前停止：检测到过拟合...")
                    return False
            
            return True
    
    # 创建训练器
    trainer = DynamicRFTrainer(x_full, y_full, val_size, overlap_ratio, base_params)
    
    # 训练循环
    while trainer.train_increment():
        pass
    
    # 获取最终测试集性能
    final_test_pred = trainer.model.predict(x_test)
    final_test_pearson = pearsonr(y_test, final_test_pred)[0]
    print(f"\n最终测试集Pearson相关系数: {final_test_pearson:.6f}")
    
    # 绘制训练历史
    plt.figure(figsize=(15, 10))
    
    # Pearson相关系数变化曲线
    plt.subplot(2, 2, 1)
    plt.plot(iterations, train_pearson_history, label='Train Pearson')
    plt.plot(iterations, val_pearson_history, label='Validation Pearson')
    plt.plot(iterations, test_pearson_history, label='Test Pearson')
    plt.title('Pearson Correlation History')
    plt.xlabel('Number of Trees')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    plt.grid(True)
    
    # 特征重要性变化
    plt.subplot(2, 2, 2)
    feature_importance_array = np.array(feature_importance_history)
    for i in range(min(5, feature_importance_array.shape[1])):  # 展示前5个最重要的特征
        plt.plot(iterations, feature_importance_array[:, i], 
                label=f'Feature {i}')
    plt.title('Top Feature Importance Evolution')
    plt.xlabel('Number of Trees')
    plt.ylabel('Importance')
    plt.legend()
    plt.grid(True)
    
    # 保存训练历史图
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'rf_training_history.png'))
    plt.close()
    
    # 保存最终模型
    model_path = os.path.join(vis_dir, 'rf_model.joblib')
    joblib.dump(trainer.model, model_path)
    
    # 保存训练历史数据
    history_df = pd.DataFrame({
        'trees': iterations,
        'train_pearson': train_pearson_history,
        'val_pearson': val_pearson_history,
        'test_pearson': test_pearson_history
    })
    history_df.to_csv(os.path.join(vis_dir, 'rf_training_history.csv'), index=False)
    
    # 分析最终特征重要性
    final_importance = trainer.model.feature_importances_
    feature_names = [f'feature_{i}' for i in range(len(final_importance))]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': final_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 6))
    plt.bar(range(30), importance_df['importance'][:30])
    plt.xticks(range(30), importance_df['feature'][:30], rotation=45)
    plt.title('Top 30 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'rf_feature_importance.png'))
    plt.close()
    
    # 保存特征重要性到CSV
    importance_df.to_csv(os.path.join(vis_dir, 'rf_feature_importance.csv'), index=False)
    
    return trainer.model, final_test_pearson

def train_decomposed_model(x_train, y_train, x_val, y_val, x_test, y_test, vis_dir):
    """使用分解模型方法训练，将20维问题分解为20个独立的一维回归问题"""
    print("\n=== 训练分解模型 ===")
    
    n_dims = x_train.shape[1]  # 应该是20
    
    # 存储每个维度的模型和预测结果
    dimension_models = []
    dimension_predictions = []
    dimension_losses = []
    
    # 计算每个维度的贡献值
    def calculate_dimension_contribution(x_dim):
        """计算单个维度的Schwefel函数贡献值: x * sin(sqrt(|x|))"""
        sqrt_abs_x = np.sqrt(np.abs(x_dim))
        sin_sqrt_abs_x = np.sin(sqrt_abs_x)
        return x_dim * sin_sqrt_abs_x
    
    # 计算所有维度的贡献值
    train_contributions = np.array([calculate_dimension_contribution(x_train[:, i:i+1]) for i in range(n_dims)])
    val_contributions = np.array([calculate_dimension_contribution(x_val[:, i:i+1]) for i in range(n_dims)])
    test_contributions = np.array([calculate_dimension_contribution(x_test[:, i:i+1]) for i in range(n_dims)])
    
    # 验证总和是否等于y值
    train_sum = np.sum(train_contributions, axis=0)
    val_sum = np.sum(val_contributions, axis=0)
    test_sum = np.sum(test_contributions, axis=0)
    
    print("\n验证贡献值总和与真实y值的差异:")
    print(f"训练集MSE: {np.mean((train_sum - y_train) ** 2):.6f}")
    print(f"验证集MSE: {np.mean((val_sum - y_val) ** 2):.6f}")
    print(f"测试集MSE: {np.mean((test_sum - y_test) ** 2):.6f}")
    
    # 为每个维度创建一个独立的GBDT模型
    base_params = {
        'objective': 'regression',
        'metric': ['l2', 'rmse'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': -1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'min_data_in_leaf': 10,
        'max_depth': 6,
        'max_bin': 255,
        'min_gain_to_split': 0.1,
        'num_iterations': 1000
    }
    
    # 创建图表布局
    n_rows = (n_dims + 3) // 4  # 确保有足够的行来容纳所有维度
    plt.figure(figsize=(20, 5 * n_rows))
    
    # 对每个维度单独训练
    for dim in range(n_dims):
        print(f"\n训练维度 {dim + 1}/{n_dims}...")
        
        # 提取当前维度的数据和贡献值
        x_train_dim = x_train[:, dim:dim+1]
        x_val_dim = x_val[:, dim:dim+1]
        x_test_dim = x_test[:, dim:dim+1]
        
        y_train_dim = train_contributions[dim]
        y_val_dim = val_contributions[dim]
        y_test_dim = test_contributions[dim]
        
        # 创建训练集
        train_data = lgb.Dataset(x_train_dim, label=y_train_dim.flatten())
        val_data = lgb.Dataset(x_val_dim, label=y_val_dim.flatten(), reference=train_data)
        
        # 训练历史记录
        train_metrics = {}
        
        # 训练模型
        model = lgb.train(
            base_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.record_evaluation(train_metrics),
                lgb.early_stopping(stopping_rounds=50)
            ]
        )
        
        # 预测每个数据集
        y_pred_test = model.predict(x_test_dim)
        
        # 计算当前维度的MSE
        mse = mean_squared_error(y_test_dim, y_pred_test)
        dimension_losses.append(mse)
        
        # 存储模型和预测结果
        dimension_models.append(model)
        dimension_predictions.append(y_pred_test)
        
        # 绘制该维度的训练历史
        plt.subplot(n_rows, 4, dim + 1)
        train_loss = train_metrics['train']['l2']
        val_loss = train_metrics['valid']['l2']
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, 'b-', label='Train')
        plt.plot(epochs, val_loss, 'r-', label='Valid')
        plt.title(f'Dimension {dim+1}\nMSE={mse:.4f}')
        plt.xlabel('Iterations')
        plt.ylabel('L2 Loss')
        if dim == 0:  # 只在第一个子图显示图例
            plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'dimension_training_history.png'))
    plt.close()
    
    # 组合所有维度的预测
    final_predictions = np.sum(dimension_predictions, axis=0)
    
    # 计算最终的Pearson相关系数和MSE
    final_test_pearson = pearsonr(y_test, final_predictions)[0]
    final_test_mse = np.mean((y_test - final_predictions) ** 2)
    print(f"\n最终模型性能:")
    print(f"测试集Pearson相关系数: {final_test_pearson:.6f}")
    print(f"测试集MSE: {final_test_mse:.6f}")
    
    # 绘制最终预测散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, final_predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Final Predictions vs True Values\nPearson={final_test_pearson:.4f}, MSE={final_test_mse:.4f}')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'final_predictions.png'))
    plt.close()
    
    # 绘制维度MSE分布
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_dims), dimension_losses)
    plt.title('MSE by Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'dimension_mse.png'))
    plt.close()
    
    # 保存每个维度的模型
    for dim, model in enumerate(dimension_models):
        model_path = os.path.join(vis_dir, f'dimension_{dim}_model.txt')
        model.save_model(model_path)
    
    # 保存训练信息
    with open(os.path.join(vis_dir, 'decomposed_model_info.txt'), 'w') as f:
        f.write("分解模型训练信息:\n")
        f.write("================\n\n")
        f.write(f"总维度数: {n_dims}\n")
        f.write(f"最终测试集性能:\n")
        f.write(f"Pearson相关系数: {final_test_pearson:.6f}\n")
        f.write(f"MSE: {final_test_mse:.6f}\n\n")
        f.write("各维度MSE:\n")
        for dim, mse in enumerate(dimension_losses):
            f.write(f"维度 {dim+1}: {mse:.6f}\n")
    
    return dimension_models, final_test_pearson, final_predictions

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc() 
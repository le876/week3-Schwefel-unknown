import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, mutual_info_regression, SelectKBest
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import KFold
from sklearn.base import clone

# 设置matplotlib使用英文默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['text.usetex'] = False

def feature_engineering(x):
    """特征工程函数 - 针对Schwefel函数优化"""
    n_samples, n_features = x.shape
    features = []
    
    # 1. 基础特征
    features.append(x)  # 原始特征
    features.append(x**2)  # 平方
    features.append(x**3)  # 立方
    
    # 2. Schwefel函数相关特征
    sqrt_abs_x = np.sqrt(np.abs(x))
    sin_sqrt_abs_x = np.sin(sqrt_abs_x)
    features.append(sin_sqrt_abs_x)  # sin(sqrt(|x|))
    features.append(x * sin_sqrt_abs_x)  # x * sin(sqrt(|x|)) - 与目标函数直接相关
    features.append(np.abs(x) * sin_sqrt_abs_x)  # |x| * sin(sqrt(|x|))
    features.append(sqrt_abs_x)  # sqrt(|x|)
    features.append(np.abs(x))  # |x|
    
    # 3. 交互特征 - 保留但减少数量，选择最有信息量的交互
    # 每个特征只与其后面的5个特征（或更少）进行交互
    for i in range(n_features):
        for j in range(i+1, min(i+6, n_features)):
            features.append(x[:, i:i+1] * x[:, j:j+1])
            # 添加Schwefel函数交互特征
            features.append(sin_sqrt_abs_x[:, i:i+1] * sin_sqrt_abs_x[:, j:j+1])
    
    # 4. 统计特征
    features.append(np.mean(x, axis=1, keepdims=True))
    features.append(np.std(x, axis=1, keepdims=True))
    features.append(np.max(x, axis=1, keepdims=True))
    features.append(np.min(x, axis=1, keepdims=True))
    # Schwefel函数特有统计特征
    features.append(np.mean(x * sin_sqrt_abs_x, axis=1, keepdims=True))  # 均值 x*sin(sqrt(|x|))
    features.append(np.sum(x * sin_sqrt_abs_x, axis=1, keepdims=True))   # 求和 x*sin(sqrt(|x|))
    
    # 5. PCA特征
    pca = PCA(n_components=5)
    pca_features = pca.fit_transform(x)
    features.append(pca_features)
    
    # 6. 聚类特征
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)  # 固定n_init避免警告
    cluster_features = kmeans.fit_predict(x).reshape(-1, 1)
    features.append(cluster_features)
    
    # 确保所有特征都是2D数组
    features_2d = []
    for feature in features:
        if feature.ndim == 1:
            feature = feature.reshape(-1, 1)
        features_2d.append(feature)
    
    return np.hstack(features_2d)

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

def select_features(x_train, y_train, x_val, x_test):
    """特征选择函数"""
    # 1. 方差过滤
    variance_selector = VarianceThreshold(threshold=0.01)
    x_train_variance = variance_selector.fit_transform(x_train)
    x_val_variance = variance_selector.transform(x_val)
    x_test_variance = variance_selector.transform(x_test)
    
    # 2. 互信息过滤
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(200, x_train_variance.shape[1]))
    x_train_mi = mi_selector.fit_transform(x_train_variance, y_train)
    x_val_mi = mi_selector.transform(x_val_variance)
    x_test_mi = mi_selector.transform(x_test_variance)
    
    # 3. 基于模型的特征选择
    model_selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    x_train_selected = model_selector.fit_transform(x_train_mi, y_train)
    x_val_selected = model_selector.transform(x_val_mi)
    x_test_selected = model_selector.transform(x_test_mi)
    
    return x_train_selected, x_val_selected, x_test_selected

def train_weighted_ensemble(x_train, y_train, x_val, y_val, x_test, y_test, vis_dir):
    """训练加权集成模型"""
    # 定义多个RF模型配置
    rf_configs = [
        {
            'name': 'RF-Wide-1',
            'params': {
                'n_estimators': 5000,        # 保持大量树
                'max_depth': 35,             # 减小深度，让每棵树更宽
                'min_samples_split': 2,      # 更容易分裂
                'min_samples_leaf': 1,
                'max_features': 0.9,         # 使用更多特征提高宽度
                'bootstrap': True,
                'max_samples': 0.9,          # 使用90%的样本
                'n_jobs': -1,
                'random_state': 42
            }
        },
        {
            'name': 'RF-Wide-2',
            'params': {
                'n_estimators': 6000,        # 增加树的数量
                'max_depth': 30,             # 更浅的树
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 0.7,         # 使用较少特征提高随机性
                'bootstrap': True,
                'max_samples': 0.8,          # 使用80%的样本增加随机性
                'n_jobs': -1,
                'random_state': 42
            }
        },
        {
            'name': 'RF-Wide-3',
            'params': {
                'n_estimators': 7000,        # 更多树
                'max_depth': None,           # 不限制深度，但依赖min_samples控制
                'min_samples_split': 3,      
                'min_samples_leaf': 2,       # 稍大的叶子节点，防止过拟合
                'max_features': 0.8,
                'bootstrap': True,
                'max_samples': 0.7,          # 较小的样本比例，提高多样性
                'n_jobs': -1,
                'random_state': 42
            }
        }
    ]
    
    # 初始化模型列表和验证集预测
    models = []
    val_predictions = np.zeros((x_val.shape[0], len(rf_configs)))
    test_predictions = np.zeros((x_test.shape[0], len(rf_configs)))
    
    # 用于记录训练过程
    training_history = {
        'RF-Wide-1': {'mse': [], 'pearson': []},
        'RF-Wide-2': {'mse': [], 'pearson': []},
        'RF-Wide-3': {'mse': [], 'pearson': []}
    }
    
    # 用于记录每个模型在测试集上的预测
    model_predictions = {}
    
    # 训练每个模型
    for i, config in enumerate(rf_configs):
        print(f"--- Training {config['name']} ---")
        model = RandomForestRegressor(**config['params'])
        
        # 记录训练过程
        n_steps = 10  # 减少记录点数，因为树的数量增加了
        step_size = config['params']['n_estimators'] // n_steps
        
        for step in range(n_steps):
            # 临时设置树的数量
            temp_n_estimators = (step + 1) * step_size
            model.set_params(n_estimators=temp_n_estimators)
            model.fit(x_train, y_train)
            
            # 在验证集上预测
            val_pred = model.predict(x_val)
            
            # 计算性能指标
            val_mse = mean_squared_error(y_val, val_pred)
            val_corr, _ = pearsonr(y_val, val_pred)
            
            # 记录历史
            training_history[config['name']]['mse'].append(val_mse)
            training_history[config['name']]['pearson'].append(val_corr)
        
        # 使用全部树训练最终模型
        model.set_params(n_estimators=config['params']['n_estimators'])
        model.fit(x_train, y_train)
        models.append(model)
        
        # 在验证集和测试集上预测
        val_predictions[:, i] = model.predict(x_val)
        test_predictions[:, i] = model.predict(x_test)
        model_predictions[config['name']] = test_predictions[:, i]
        
        # 计算验证集性能
        val_mse = mean_squared_error(y_val, val_predictions[:, i])
        val_r2 = r2_score(y_val, val_predictions[:, i])
        val_corr, _ = pearsonr(y_val, val_predictions[:, i])
        print(f"{config['name']} Validation MSE: {val_mse:.6f}")
        print(f"{config['name']} Validation R²: {val_r2:.6f}")
        print(f"{config['name']} Validation Pearson Correlation: {val_corr:.6f}")
        
        # 绘制特征重要性
        try:
            importances = model.feature_importances_
            plt.figure(figsize=(10, 8))
            n_features = min(30, len(importances))
            indices = np.argsort(importances)[-n_features:]
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [f"Feature_{idx}" for idx in indices])
            plt.title(f'Feature Importance - {config["name"]}')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'feature_importance_{config["name"]}.png'))
            plt.close()
        except Exception as e:
            print(f"Could not plot feature importance for {config['name']}: {str(e)}")
    
    # 绘制训练过程
    plt.figure(figsize=(15, 5))
    
    # MSE变化趋势
    plt.subplot(1, 2, 1)
    for name in training_history:
        plt.plot(training_history[name]['mse'], label=name)
    plt.xlabel('Training Steps')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE vs Training Steps')
    plt.legend()
    
    # Pearson相关系数变化趋势
    plt.subplot(1, 2, 2)
    for name in training_history:
        plt.plot(training_history[name]['pearson'], label=name)
    plt.xlabel('Training Steps')
    plt.ylabel('Validation Pearson Correlation')
    plt.title('Validation Pearson Correlation vs Training Steps')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_history.png'))
    plt.close()
    
    # 计算每个模型的权重（基于验证集性能）
    weights = np.zeros(len(models))
    for i in range(len(models)):
        val_corr, _ = pearsonr(y_val, val_predictions[:, i])
        weights[i] = val_corr ** 2  # 使用相关系数的平方作为权重
    
    # 归一化权重
    weights = weights / np.sum(weights)
    
    # 加权集成预测
    val_ensemble = np.average(val_predictions, axis=1, weights=weights)
    test_ensemble = np.average(test_predictions, axis=1, weights=weights)
    
    # 计算集成模型的性能
    val_ensemble_mse = mean_squared_error(y_val, val_ensemble)
    val_ensemble_r2 = r2_score(y_val, val_ensemble)
    val_ensemble_corr, _ = pearsonr(y_val, val_ensemble)
    
    test_ensemble_mse = mean_squared_error(y_test, test_ensemble)
    test_ensemble_r2 = r2_score(y_test, test_ensemble)
    test_ensemble_corr, _ = pearsonr(y_test, test_ensemble)
    
    print("\nEnsemble Model Performance:")
    print(f"Validation MSE: {val_ensemble_mse:.6f}")
    print(f"Validation R²: {val_ensemble_r2:.6f}")
    print(f"Validation Pearson Correlation: {val_ensemble_corr:.6f}")
    print(f"Test MSE: {test_ensemble_mse:.6f}")
    print(f"Test R²: {test_ensemble_r2:.6f}")
    print(f"Test Pearson Correlation: {test_ensemble_corr:.6f}")
    
    # 绘制所有模型预测的比较图
    plt.figure(figsize=(20, 10))
    
    # 预测散点图矩阵
    models_to_plot = list(model_predictions.keys()) + ['Ensemble']
    model_predictions['Ensemble'] = test_ensemble
    
    for i, model_name1 in enumerate(models_to_plot):
        # 真实值 vs 预测值散点图
        plt.subplot(len(models_to_plot), len(models_to_plot), i * len(models_to_plot) + 1)
        plt.scatter(y_test, model_predictions[model_name1], alpha=0.5, s=5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        if i == 0:
            plt.title('True Values')
        if i == len(models_to_plot) - 1:
            plt.xlabel('True Values')
        plt.ylabel(f'{model_name1}')
        
        # 模型之间的预测对比
        for j, model_name2 in enumerate(models_to_plot[1:], start=1):
            if j <= i:
                ax = plt.subplot(len(models_to_plot), len(models_to_plot), i * len(models_to_plot) + j + 1)
                if model_name1 != model_name2:
                    plt.scatter(model_predictions[model_name2], model_predictions[model_name1], alpha=0.5, s=5)
                    corr, _ = pearsonr(model_predictions[model_name2], model_predictions[model_name1])
                    plt.title(f'Corr: {corr:.3f}')
                else:
                    plt.text(0.5, 0.5, 'Same Model', ha='center', va='center', transform=ax.transAxes)
                if i == len(models_to_plot) - 1:
                    plt.xlabel(f'{model_name2}')
                if j == 1:
                    plt.ylabel(f'{model_name1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'model_comparison_matrix.png'))
    plt.close()
    
    # 绘制每个模型的预测误差分布
    plt.figure(figsize=(15, 5))
    
    for i, model_name in enumerate(models_to_plot):
        plt.subplot(1, len(models_to_plot), i+1)
        errors = y_test - model_predictions[model_name]
        plt.hist(errors, bins=50, alpha=0.7)
        plt.title(f'{model_name}\nMean: {np.mean(errors):.2f}, Std: {np.std(errors):.2f}')
        plt.xlabel('Prediction Error')
        if i == 0:
            plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'prediction_error_distribution.png'))
    plt.close()
    
    # 绘制预测结果对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_ensemble, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values (Ensemble)')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'prediction_scatter.png'))
    plt.close()
    
    # 绘制残差图
    plt.figure(figsize=(10, 6))
    residuals = y_test - test_ensemble
    plt.scatter(test_ensemble, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'residual_plot.png'))
    plt.close()
    
    return models, weights

def main():
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join('schwefel/visualizations', timestamp)
    os.makedirs(vis_dir, exist_ok=True)
    
    # 加载和预处理数据
    x_train, x_val, x_test, y_train, y_val, y_test, (scaler_input, scaler_features) = preprocess_data()
    
    # 特征选择前分析特征与目标的相关性
    print("分析特征与目标的相关性...")
    analyze_feature_correlations(x_train, y_train, vis_dir)
    
    # 特征选择
    x_train_selected, x_val_selected, x_test_selected = select_features(x_train, y_train, x_val, x_test)
    
    # 训练模型
    models, weights = train_weighted_ensemble(x_train_selected, y_train, x_val_selected, y_val, 
                                            x_test_selected, y_test, vis_dir)
    
    # 执行参数敏感性分析（可选，影响训练时间）
    # print("执行参数敏感性分析...")
    # parameter_sensitivity_analysis(x_train_selected, y_train, x_val_selected, y_val, vis_dir)
    
    # 保存训练信息
    with open(os.path.join(vis_dir, 'training_info.txt'), 'w') as f:
        f.write("Training Information:\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of features after selection: {x_train_selected.shape[1]}\n")
        f.write(f"Model weights: {weights}\n")
        f.write("\nModel Parameters:\n")
        for i, model in enumerate(models):
            f.write(f"\nModel {i+1}:\n")
            f.write(str(model.get_params()))

def analyze_feature_correlations(x_train, y_train, vis_dir):
    """分析特征与目标的相关性"""
    # 计算每个特征与目标的相关性
    correlations = []
    for i in range(x_train.shape[1]):
        corr, _ = pearsonr(x_train[:, i], y_train)
        correlations.append((i, corr))
    
    # 按相关性绝对值排序
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # 绘制相关性前30的特征
    plt.figure(figsize=(12, 8))
    top_n = min(30, len(correlations))
    indices = [correlations[i][0] for i in range(top_n)]
    corrs = [correlations[i][1] for i in range(top_n)]
    
    plt.barh(range(top_n), corrs)
    plt.yticks(range(top_n), [f"Feature_{idx}" for idx in indices])
    plt.xlabel('Pearson Correlation with Target')
    plt.title('Top Features Correlation with Target')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_target_correlation.png'))
    plt.close()
    
    # 计算前10特征之间的相关性矩阵
    top_indices = [correlations[i][0] for i in range(min(10, len(correlations)))]
    top_features = x_train[:, top_indices]
    
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(top_features.T)
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Pearson Correlation')
    plt.xticks(range(len(top_indices)), [f"F_{idx}" for idx in top_indices], rotation=45)
    plt.yticks(range(len(top_indices)), [f"F_{idx}" for idx in top_indices])
    plt.title('Correlation Matrix of Top Features')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_correlation_matrix.png'))
    plt.close()

def parameter_sensitivity_analysis(x_train, y_train, x_val, y_val, vis_dir):
    """参数敏感性分析"""
    # 分析树的数量对性能的影响
    n_estimators_range = [100, 500, 1000, 2000, 3000, 5000]
    pearson_scores = []
    
    for n_est in n_estimators_range:
        model = RandomForestRegressor(n_estimators=n_est, max_depth=None, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        corr, _ = pearsonr(y_val, pred)
        pearson_scores.append(corr)
    
    plt.figure(figsize=(12, 6))
    plt.plot(n_estimators_range, pearson_scores, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Pearson Correlation')
    plt.title('Model Performance vs Number of Trees')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'n_estimators_sensitivity.png'))
    plt.close()
    
    # 分析树的深度对性能的影响
    max_depth_range = [10, 20, 30, 40, 50, None]
    depth_names = [str(d) if d is not None else 'None' for d in max_depth_range]
    pearson_scores = []
    
    for depth in max_depth_range:
        model = RandomForestRegressor(n_estimators=1000, max_depth=depth, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        corr, _ = pearsonr(y_val, pred)
        pearson_scores.append(corr)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(max_depth_range)), pearson_scores, marker='o')
    plt.xticks(range(len(max_depth_range)), depth_names)
    plt.xlabel('Max Depth')
    plt.ylabel('Pearson Correlation')
    plt.title('Model Performance vs Max Depth')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'max_depth_sensitivity.png'))
    plt.close()
    
    # 分析max_features参数对性能的影响
    max_features_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    pearson_scores = []
    
    for max_feat in max_features_range:
        model = RandomForestRegressor(n_estimators=1000, max_features=max_feat, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        corr, _ = pearsonr(y_val, pred)
        pearson_scores.append(corr)
    
    plt.figure(figsize=(12, 6))
    plt.plot(max_features_range, pearson_scores, marker='o')
    plt.xlabel('Max Features Ratio')
    plt.ylabel('Pearson Correlation')
    plt.title('Model Performance vs Max Features')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'max_features_sensitivity.png'))
    plt.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc() 
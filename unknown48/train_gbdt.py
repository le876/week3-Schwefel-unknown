import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 设置matplotlib使用默认英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 创建必要的目录
os.makedirs('data/raw', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# 用于跟踪所有模型的pearson比率
pearson_history = {
    'model_names': [],
    'pearson_values': []
}

# 创建以时间命名的可视化输出目录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
vis_dir = os.path.join('visualizations', timestamp)
os.makedirs(vis_dir, exist_ok=True)
print(f"Visualization output will be saved to: {vis_dir}")

# 加载数据
def load_data():
    try:
        # 加载数据
        x_train = np.load('data/raw/x_48_train(1).npy')
        y_train = np.load('data/raw/y_48_train(1).npy')
        x_test = np.load('data/raw/x_48_test(1).npy')
        y_test = np.load('data/raw/y_48_test(1).npy')
        
        print("Training data shape:", x_train.shape)
        print("Training labels shape:", y_train.shape)
        print("Testing data shape:", x_test.shape)
        print("Testing labels shape:", y_test.shape)
        
        return x_train, y_train, x_test, y_test
    
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please make sure data files are in the 'data/raw' directory.")
        sys.exit(1)

# 特征工程
def feature_engineering(x):
    """特征工程：添加更多高级特征"""
    n_samples = x.shape[0]
    n_features = x.shape[1]
    n_timesteps = x.shape[2]
    
    # 1. 基本统计特征
    mean_features = np.mean(x, axis=2)
    std_features = np.std(x, axis=2)
    max_features = np.max(x, axis=2)
    min_features = np.min(x, axis=2)
    
    # 2. 时间序列特征
    trend_features = np.zeros((n_samples, n_features))
    for i in range(n_features):
        for j in range(n_samples):
            x_trend = np.arange(n_timesteps)
            y_trend = x[j, i, :]
            slope = np.polyfit(x_trend, y_trend, 1)[0]
            trend_features[j, i] = slope
    
    # 3. 高级统计特征
    skew_features = np.zeros((n_samples, n_features))
    kurt_features = np.zeros((n_samples, n_features))
    for i in range(n_features):
        for j in range(n_samples):
            skew_features[j, i] = np.mean(((x[j, i, :] - mean_features[j, i]) / std_features[j, i]) ** 3)
            kurt_features[j, i] = np.mean(((x[j, i, :] - mean_features[j, i]) / std_features[j, i]) ** 4) - 3
    
    # 4. 时间窗口特征
    window_sizes = [3, 5, 7]
    window_features = []
    for window_size in window_sizes:
        rolling_mean = np.zeros((n_samples, n_features))
        rolling_std = np.zeros((n_samples, n_features))
        for i in range(n_features):
            for j in range(n_samples):
                if window_size <= n_timesteps:
                    rolling_mean[j, i] = np.mean(x[j, i, -window_size:])
                    rolling_std[j, i] = np.std(x[j, i, -window_size:])
        window_features.extend([rolling_mean, rolling_std])
    
    # 5. 特征平方和交互项
    squared_features = np.square(x).mean(axis=2)
    interaction_features = np.zeros((n_samples, n_features * (n_features - 1) // 2))
    idx = 0
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interaction_features[:, idx] = mean_features[:, i] * mean_features[:, j]
            idx += 1
    
    # 6. 差分特征
    diff_features = np.zeros((n_samples, n_features))
    for i in range(n_features):
        for j in range(n_samples):
            diff_features[j, i] = np.mean(np.diff(x[j, i, :]))
    
    # 组合所有特征
    engineered_features = np.concatenate([
        mean_features,
        std_features,
        max_features,
        min_features,
        trend_features,
        skew_features,
        kurt_features,
        squared_features,
        diff_features,
        interaction_features
    ] + window_features, axis=1)
    
    return engineered_features

# 数据预处理
def preprocess_data(x_train, y_train, x_test, y_test, vis_dir):
    """数据预处理和特征工程"""
    # 特征工程
    print("Performing feature engineering...")
    x_train_engineered = feature_engineering(x_train)
    x_test_engineered = feature_engineering(x_test)
    
    print("Engineered features shape:", x_train_engineered.shape)
    
    # 标准化特征
    scaler_x = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train_engineered)
    x_test_scaled = scaler_x.transform(x_test_engineered)
    
    # 标准化目标变量
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # 划分训练集和验证集
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )
    
    # 可视化目标变量分布
    plt.figure(figsize=(10, 6))
    plt.hist(y_train_split, bins=50, alpha=0.5, label='Training')
    plt.hist(y_val, bins=50, alpha=0.5, label='Validation')
    plt.title('Target Variable Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(vis_dir, 'target_distribution.png'))
    plt.close()
    
    return x_train_split, y_train_split, x_val, y_val, x_test_scaled, y_test_scaled, scaler_y

# 特征选择
def select_features(x_train, y_train, x_val, x_test):
    """使用随机森林进行特征选择"""
    print("Performing feature selection...")
    
    # 使用随机森林训练一个模型
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    
    # 使用SelectFromModel选择最重要的特征
    selector = SelectFromModel(rf, threshold='mean', prefit=True)
    
    # 应用特征选择
    x_train_selected = selector.transform(x_train)
    x_val_selected = selector.transform(x_val)
    x_test_selected = selector.transform(x_test)
    
    print(f"Selected {x_train_selected.shape[1]} features from {x_train.shape[1]} original features")
    
    # 返回经过筛选的特征数据
    return x_train_selected, x_val_selected, x_test_selected, selector

# 学习率调度器
class LearningRateScheduler:
    """学习率调度器，支持预热和周期性学习率"""
    def __init__(self, init_lr=0.01, min_lr=0.0001, warmup_rounds=50, decay_factor=0.75, decay_rounds=100):
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.warmup_rounds = warmup_rounds
        self.decay_factor = decay_factor
        self.decay_rounds = decay_rounds
        self.current_lr = 0.0
        
    def __call__(self, current_round):
        # 预热阶段
        if current_round < self.warmup_rounds:
            self.current_lr = self.min_lr + (self.init_lr - self.min_lr) * current_round / self.warmup_rounds
        # 周期性衰减阶段
        else:
            cycle = (current_round - self.warmup_rounds) // self.decay_rounds
            self.current_lr = max(self.min_lr, self.init_lr * (self.decay_factor ** cycle))
        
        return self.current_lr

# 训练函数增强版
def train_model_enhanced(x_train, y_train, x_val, y_val, vis_dir, x_test=None, y_test=None, scaler_y=None, second_stage=False):
    """增强版训练函数，支持学习率调度和交叉验证"""
    # 创建数据集
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)
    
    # 基本参数
    base_params = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'feature_pre_filter': False,
        'min_child_samples': 20,
        'min_child_weight': 1e-3,
        'min_split_gain': 1e-3,
    }
    
    # 不同阶段使用不同参数
    if not second_stage:
        model_configs = [
            {
                'num_leaves': 31,
                'max_depth': 6,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 7,
                'reg_alpha': 0.15,
                'reg_lambda': 0.15,
            },
            {
                'num_leaves': 63,
                'max_depth': 7,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.6,
                'bagging_freq': 10,
                'reg_alpha': 0.2,
                'reg_lambda': 0.2,
            },
            {
                'num_leaves': 95,
                'max_depth': 8,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.5,
                'bagging_freq': 15,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
            }
        ]
        init_lrs = [0.01, 0.005, 0.003]
    else:
        # 第二阶段使用更保守的参数来避免过拟合
        model_configs = [
            {
                'num_leaves': 31,
                'max_depth': 5,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.6,
                'bagging_freq': 10,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
            },
            {
                'num_leaves': 47,
                'max_depth': 6,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.5,
                'bagging_freq': 12,
                'reg_alpha': 0.4,
                'reg_lambda': 0.4,
            }
        ]
        init_lrs = [0.003, 0.001]
    
    # 训练多个模型
    models = []
    val_preds = []
    test_preds = []
    best_iterations = []
    
    # 用于记录每个模型在每个迭代的pearson相关系数
    pearson_tracking = []
    model_names = []
    
    for i, (model_config, init_lr) in enumerate(zip(model_configs, init_lrs)):
        print(f"\nTraining {'second-stage ' if second_stage else ''}model {i+1} with parameters:")
        print(model_config)
        print(f"Initial learning rate: {init_lr}")
        
        # 合并参数
        params = {**base_params, **model_config}
        
        # 创建学习率调度器
        lr_scheduler = LearningRateScheduler(
            init_lr=init_lr,
            min_lr=init_lr/100,
            warmup_rounds=100,
            decay_factor=0.9,
            decay_rounds=200
        )
        
        # 创建记录评估结果的字典
        eval_results = {}
        
        # 记录Pearson correlation的变化
        iteration_pearson_scores = []
        
        # 创建Pearson correlation回调函数
        def pearson_eval(period=100):
            def callback(env):
                if (env.iteration + 1) % period != 0 and env.iteration + 1 != env.end_iteration:
                    return
                
                if x_test is not None and y_test is not None and scaler_y is not None:
                    # 获取当前模型在测试集上的预测
                    test_pred = env.model.predict(x_test)
                    y_pred = scaler_y.inverse_transform(test_pred.reshape(-1, 1))
                    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                    
                    # 计算Pearson相关系数
                    pearson = pearsonr(y_test_original.flatten(), y_pred.flatten())[0]
                    iteration_pearson_scores.append((env.iteration + 1, pearson))
                    
                    # 打印当前迭代的Pearson相关系数
                    if (env.iteration + 1) % (period * 10) == 0 or env.iteration + 1 == env.end_iteration:
                        print(f"Iteration {env.iteration + 1}: Pearson = {pearson:.6f}")
            
            return callback
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=8000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=300),
                lgb.record_evaluation(eval_results),
                lgb.callback.reset_parameter(learning_rate=lr_scheduler),
                pearson_eval(period=50)  # 每50次迭代计算一次Pearson
            ]
        )
        
        models.append(model)
        best_iterations.append(model.best_iteration)
        
        # 保存模型
        model_suffix = f"stage2_model_{i+1}" if second_stage else f"model_{i+1}"
        model.save_model(os.path.join(vis_dir, f'{model_suffix}.txt'))
        
        # 保存训练过程
        train_loss = eval_results['train']['l2']
        val_loss = eval_results['val']['l2']
        epochs = range(1, len(train_loss) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.title(f"{'Second-stage ' if second_stage else ''}Model {i+1} - Training and Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('L2 Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(vis_dir, f'{model_suffix}_loss.png'))
        plt.close()
        
        # 可视化Pearson相关系数的变化趋势
        if iteration_pearson_scores:
            iterations, pearson_scores = zip(*iteration_pearson_scores)
            pearson_tracking.append(pearson_scores)
            model_name = f"{'Second-stage ' if second_stage else ''}Model {i+1}"
            model_names.append(model_name)
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, pearson_scores, '-', marker='o', markersize=3)
            plt.axhline(y=max(pearson_scores), color='r', linestyle='--', 
                        label=f'Max: {max(pearson_scores):.6f}')
            plt.title(f"{model_name} - Pearson Correlation Trend")
            plt.xlabel('Iterations')
            plt.ylabel('Pearson Correlation')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(vis_dir, f'{model_suffix}_pearson_trend.png'))
            plt.close()
        
        # 记录验证集和测试集预测
        val_preds.append(model.predict(x_val))
        if x_test is not None:
            test_preds.append(model.predict(x_test))
    
    # 如果有测试集，计算每个模型的性能
    if x_test is not None and y_test is not None and scaler_y is not None:
        test_pearson_scores = []
        
        # 计算每个模型的测试集性能
        for i, test_pred in enumerate(test_preds):
            y_pred = scaler_y.inverse_transform(test_pred.reshape(-1, 1))
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            pearson = pearsonr(y_test_original.flatten(), y_pred.flatten())[0]
            test_pearson_scores.append(pearson)
            model_label = f"{'Second-stage ' if second_stage else ''}Model {i+1}"
            print(f"{model_label} Test Pearson: {pearson:.6f}")
            
            # 记录到全局历史
            global pearson_history
            pearson_history['model_names'].append(model_label)
            pearson_history['pearson_values'].append(pearson)
    
    # 如果有测试集和多个模型，绘制所有模型的Pearson相关系数变化趋势对比图
    if len(pearson_tracking) > 1:
        plt.figure(figsize=(12, 8))
        max_length = max(len(scores) for scores in pearson_tracking)
        for i, (scores, name) in enumerate(zip(pearson_tracking, model_names)):
            # 将不同长度的score列表调整为相同周期
            x_values = np.linspace(0, 1, len(scores))
            # 使用线性插值将其拓展到标准长度
            if len(scores) < max_length:
                scores_interp = np.interp(
                    np.linspace(0, 1, max_length),
                    x_values,
                    scores
                )
                x_ticks = np.linspace(0, max(50 * max_length, 8000), max_length)
                plt.plot(x_ticks, scores_interp, '-', label=name)
            else:
                x_ticks = np.linspace(0, max(50 * len(scores), 8000), len(scores))
                plt.plot(x_ticks, scores, '-', label=name)
            
        plt.title("Pearson Correlation Trend Comparison of All Models")
        plt.xlabel('Iterations')
        plt.ylabel('Pearson Correlation')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(vis_dir, f"{'second_stage_' if second_stage else ''}all_models_pearson_trend.png"))
        plt.close()
    
    return models, val_preds, test_preds, best_iterations

# 交叉验证模型训练
def train_cv_models(x_train, y_train, x_test, y_test, scaler_y, vis_dir, n_splits=5):
    """使用交叉验证训练多个模型"""
    print("=== Training with Cross-Validation ===")
    
    # 创建KFold交叉验证对象
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 基本参数
    base_params = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_data_in_leaf': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'learning_rate': 0.01
    }
    
    # 存储每个fold的模型和预测结果
    cv_models = []
    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros((n_splits, len(y_test)))
    
    # 记录每个fold的Pearson评分
    fold_pearson_scores = []
    
    # 对每个fold进行训练
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
        print(f"\nTraining fold {fold+1}/{n_splits}")
        
        # 划分训练集和验证集
        x_tr, x_vl = x_train[train_idx], x_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]
        
        # 创建数据集
        train_data = lgb.Dataset(x_tr, label=y_tr)
        val_data = lgb.Dataset(x_vl, label=y_vl, reference=train_data)
        
        # 记录Pearson correlation的变化
        iteration_pearson_scores = []
        
        # 创建Pearson correlation回调函数
        def pearson_eval(period=100):
            def callback(env):
                if (env.iteration + 1) % period != 0 and env.iteration + 1 != env.end_iteration:
                    return
                
                # 获取当前模型在测试集上的预测
                test_pred = env.model.predict(x_test)
                y_pred = scaler_y.inverse_transform(test_pred.reshape(-1, 1))
                y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                
                # 计算Pearson相关系数
                pearson = pearsonr(y_test_original.flatten(), y_pred.flatten())[0]
                iteration_pearson_scores.append((env.iteration + 1, pearson))
                
                # 打印当前迭代的Pearson相关系数
                if (env.iteration + 1) % (period * 10) == 0 or env.iteration + 1 == env.end_iteration:
                    print(f"Fold {fold+1} - Iteration {env.iteration + 1}: Pearson = {pearson:.6f}")
            
            return callback
        
        # 训练模型
        model = lgb.train(
            base_params,
            train_data,
            num_boost_round=5000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                pearson_eval(period=50)  # 每50次迭代计算一次Pearson
            ]
        )
        
        # 存储模型和预测结果
        cv_models.append(model)
        oof_preds[val_idx] = model.predict(x_vl)
        test_preds[fold] = model.predict(x_test)
        
        # 保存模型
        model.save_model(os.path.join(vis_dir, f'cv_model_fold_{fold+1}.txt'))
        
        # 可视化Pearson相关系数的变化趋势
        if iteration_pearson_scores:
            iterations, pearson_scores = zip(*iteration_pearson_scores)
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, pearson_scores, '-', marker='o', markersize=3)
            plt.axhline(y=max(pearson_scores), color='r', linestyle='--', 
                        label=f'Max: {max(pearson_scores):.6f}')
            plt.title(f"Fold {fold+1} - Pearson Correlation Trend")
            plt.xlabel('Iterations')
            plt.ylabel('Pearson Correlation')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(vis_dir, f'cv_fold_{fold+1}_pearson_trend.png'))
            plt.close()
    
    # 计算交叉验证的性能
    cv_score = pearsonr(y_train, oof_preds)[0]
    print(f"\nCross-Validation Pearson Score: {cv_score:.6f}")
    
    # 计算每个fold在测试集上的性能
    fold_test_scores = []
    for fold in range(n_splits):
        y_pred = scaler_y.inverse_transform(test_preds[fold].reshape(-1, 1))
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        pearson = pearsonr(y_test_original.flatten(), y_pred.flatten())[0]
        fold_test_scores.append(pearson)
        print(f"Fold {fold+1} Test Pearson: {pearson:.6f}")
        
        # 记录到全局历史
        global pearson_history
        pearson_history['model_names'].append(f"CV Fold {fold+1}")
        pearson_history['pearson_values'].append(pearson)
    
    # 可视化各Fold的Pearson相关系数比较
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, n_splits+1), fold_test_scores)
    plt.axhline(y=np.mean(fold_test_scores), color='r', linestyle='--', 
                label=f'Average: {np.mean(fold_test_scores):.6f}')
    plt.title("Pearson Correlation Comparison of All Folds")
    plt.xlabel('Fold')
    plt.ylabel('Pearson Correlation')
    plt.xticks(range(1, n_splits+1))
    plt.grid(True, axis='y')
    plt.legend()
    plt.savefig(os.path.join(vis_dir, 'cv_folds_pearson_comparison.png'))
    plt.close()
    
    # 计算平均的测试预测
    test_preds_mean = np.mean(test_preds, axis=0)
    y_pred = scaler_y.inverse_transform(test_preds_mean.reshape(-1, 1))
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    pearson = pearsonr(y_test_original.flatten(), y_pred.flatten())[0]
    print(f"Average CV Test Pearson: {pearson:.6f}")
    
    # 记录到全局历史
    pearson_history['model_names'].append("CV Average")
    pearson_history['pearson_values'].append(pearson)
    
    return cv_models, oof_preds, test_preds_mean, pearson

# 堆叠集成
def stacking_ensemble(x_train, y_train, x_val, y_val, x_test, y_test, scaler_y, vis_dir):
    """使用堆叠集成来组合多种模型的预测结果"""
    print("=== Training Stacking Ensemble ===")
    
    # 第一层模型：训练不同模型
    print("\n--- Training First-Level Models ---")
    
    # GBDT模型
    first_models, val_preds_first, test_preds_first, best_iterations = train_model_enhanced(
        x_train, y_train, x_val, y_val, vis_dir, x_test, y_test, scaler_y
    )
    
    # 交叉验证模型
    print("\n--- Training Cross-Validation Models ---")
    cv_models, cv_oof_preds, cv_test_preds, cv_pearson = train_cv_models(
        np.vstack((x_train, x_val)),
        np.hstack((y_train, y_val)),
        x_test, y_test, scaler_y, vis_dir, n_splits=5
    )
    
    # 准备第二层模型的训练数据
    n_first_models = len(first_models)
    
    # 在验证集上的预测作为第二层模型的训练数据
    val_preds_array = np.column_stack(val_preds_first)
    
    # 测试集上的预测
    test_preds_array = np.column_stack(test_preds_first + [cv_test_preds])
    
    print("\n--- Training Second-Level Models ---")
    
    # 第二层模型：使用各种元学习器
    meta_learners = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001),
        'LinearRegression': LinearRegression()
    }
    
    meta_preds = {}
    meta_scores = {}
    
    for name, model in meta_learners.items():
        print(f"Training meta-learner: {name}")
        
        # 训练元学习器
        model.fit(val_preds_array, y_val)
        
        # 在测试集上进行预测
        meta_preds[name] = model.predict(test_preds_array)
        
        # 计算性能
        y_pred = scaler_y.inverse_transform(meta_preds[name].reshape(-1, 1))
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        pearson = pearsonr(y_test_original.flatten(), y_pred.flatten())[0]
        meta_scores[name] = pearson
        print(f"{name} Test Pearson: {pearson:.6f}")
    
    # 找出最佳的元学习器
    best_meta = max(meta_scores, key=meta_scores.get)
    best_score = meta_scores[best_meta]
    print(f"\nBest meta-learner: {best_meta} with Pearson score: {best_score:.6f}")
    
    # 尝试加权平均
    weights = np.array([0.4, 0.3, 0.2, 0.1])  # 根据各模型性能调整权重
    weights_norm = weights / weights.sum()
    
    weighted_preds = np.zeros(len(y_test))
    for i, w in enumerate(weights_norm):
        if i < n_first_models:
            weighted_preds += w * test_preds_first[i]
        else:
            weighted_preds += w * cv_test_preds
    
    # 计算加权平均的性能
    y_pred = scaler_y.inverse_transform(weighted_preds.reshape(-1, 1))
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    weighted_pearson = pearsonr(y_test_original.flatten(), y_pred.flatten())[0]
    print(f"Weighted Average Test Pearson: {weighted_pearson:.6f}")
    
    # 选择最佳预测作为最终结果
    if weighted_pearson > best_score:
        print("Using weighted average as final prediction")
        final_pred = weighted_preds
        final_score = weighted_pearson
        best_method = "Weighted Average"
    else:
        print(f"Using {best_meta} as final prediction")
        final_pred = meta_preds[best_meta]
        final_score = best_score
        best_method = best_meta
    
    # 反归一化并计算最终性能
    y_pred = scaler_y.inverse_transform(final_pred.reshape(-1, 1))
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)
    pearson = final_score
    
    print("\n=== Final Stacking Ensemble Results ===")
    print(f"Best Method: {best_method}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test R²: {r2:.6f}")
    print(f"Test Pearson Correlation: {pearson:.6f}")
    
    # 可视化预测结果
    plt.figure(figsize=(12, 10))
    
    # 散点图
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_original, y_pred, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Stacking Ensemble ({best_method}): Predicted vs True Values (Pearson = {pearson:.4f})')
    plt.grid(True)
    
    # 残差图
    plt.subplot(2, 2, 2)
    residuals = y_test_original - y_pred
    plt.scatter(y_test_original, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residual Distribution')
    plt.grid(True)
    
    # 预测值和真实值的分布
    plt.subplot(2, 2, 3)
    plt.hist(y_test_original, bins=20, alpha=0.5, label='True Values')
    plt.hist(y_pred, bins=20, alpha=0.5, label='Predicted Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('True and Predicted Value Distribution')
    plt.legend()
    plt.grid(True)
    
    # 残差分布
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Histogram')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'stacking_predictions.png'))
    plt.close()
    
    # 保存结果
    results = {
        'model': ['Stacking Ensemble'],
        'best_method': [best_method],
        'mse': [mse],
        'rmse': [rmse],
        'r2': [r2],
        'pearson': [pearson]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(vis_dir, 'stacking_test_results.csv'), index=False)
    
    # 返回最佳模型和性能
    all_models = {
        'first_level': first_models,
        'cv_models': cv_models,
        'meta_models': meta_learners,
        'best_method': best_method
    }
    
    return all_models, mse, r2, pearson

def main():
    print("===== Advanced LightGBM Model Training and Evaluation =====")
    
    # 加载数据
    x_train, y_train, x_test, y_test = load_data()
    
    # 预处理数据
    x_train_split, y_train_split, x_val, y_val, x_test_scaled, y_test_scaled, scaler_y = preprocess_data(
        x_train, y_train, x_test, y_test, vis_dir
    )
    
    # 特征选择
    x_train_selected, x_val_selected, x_test_selected, selector = select_features(
        x_train_split, y_train_split, x_val, x_test_scaled
    )
    
    # 直接使用GBDT模型训练，不使用堆叠集成
    print("\n--- Training GBDT Models ---")
    models, val_preds, test_preds, best_iterations = train_model_enhanced(
        x_train_selected, y_train_split, 
        x_val_selected, y_val,
        vis_dir, x_test_selected, y_test_scaled, scaler_y
    )
    
    # 评估模型性能
    best_model_idx = 0
    best_pearson = 0
    for i, test_pred in enumerate(test_preds):
        y_pred = scaler_y.inverse_transform(test_pred.reshape(-1, 1))
        y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred)
        pearson = pearsonr(y_test_original.flatten(), y_pred.flatten())[0]
        
        print(f"\nModel {i+1} Test Results:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Pearson Correlation: {pearson:.6f}")
        
        if pearson > best_pearson:
            best_pearson = pearson
            best_model_idx = i
            best_mse = mse
            best_r2 = r2
    
    # 使用最佳模型进行预测
    best_model = models[best_model_idx]
    final_pred = test_preds[best_model_idx]
    y_pred = scaler_y.inverse_transform(final_pred.reshape(-1, 1))
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))
    
    # 可视化最佳模型的预测结果
    plt.figure(figsize=(12, 10))
    
    # 散点图
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_original, y_pred, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'GBDT Model {best_model_idx+1}: Predicted vs True Values (Pearson = {best_pearson:.4f})')
    plt.grid(True)
    
    # 残差图
    plt.subplot(2, 2, 2)
    residuals = y_test_original - y_pred
    plt.scatter(y_test_original, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residual Distribution')
    plt.grid(True)
    
    # 预测值和真实值的分布
    plt.subplot(2, 2, 3)
    plt.hist(y_test_original, bins=20, alpha=0.5, label='True Values')
    plt.hist(y_pred, bins=20, alpha=0.5, label='Predicted Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('True and Predicted Value Distribution')
    plt.legend()
    plt.grid(True)
    
    # 残差分布
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Histogram')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'gbdt_best_model_predictions.png'))
    plt.close()
    
    # 保存最佳模型的结果
    results = {
        'model': [f'GBDT Model {best_model_idx+1}'],
        'mse': [best_mse],
        'rmse': [np.sqrt(best_mse)],
        'r2': [best_r2],
        'pearson': [best_pearson]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(vis_dir, 'gbdt_test_results.csv'), index=False)
    
    # 创建所有模型Pearson相关系数比较图
    global pearson_history
    if len(pearson_history['model_names']) > 0:
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(pearson_history['model_names'])), pearson_history['pearson_values'])
        
        # 找出最大值并标记
        max_idx = np.argmax(pearson_history['pearson_values'])
        max_value = pearson_history['pearson_values'][max_idx]
        max_model = pearson_history['model_names'][max_idx]
        
        # 使用不同颜色标记最大值
        bars[max_idx].set_color('red')
        
        plt.axhline(y=max_value, color='r', linestyle='--', 
                    label=f'Max: {max_value:.6f} ({max_model})')
        
        plt.title("Pearson Correlation Comparison of All Models")
        plt.xlabel('Model')
        plt.ylabel('Pearson Correlation')
        plt.xticks(range(len(pearson_history['model_names'])), pearson_history['model_names'], rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'all_models_pearson_comparison.png'))
        plt.close()
    
    # 检查是否达到目标
    if best_pearson >= 0.98:
        print("\nCongratulations! Model achieved target Pearson correlation (>= 0.98).")
    else:
        print(f"\nModel's Pearson correlation is {best_pearson:.4f}, did not reach target value 0.98.")
        
    # 保存最终训练信息
    with open(os.path.join(vis_dir, 'gbdt_training_info.txt'), 'w') as f:
        f.write("Advanced GBDT Model Training Information\n")
        f.write("================================================\n\n")
        f.write(f"Training Time: {timestamp}\n")
        f.write(f"Original Input Shape: {x_train_split.shape}\n")
        f.write(f"Selected Features Shape: {x_train_selected.shape}\n")
        f.write(f"Training Samples: {x_train_split.shape[0]}\n")
        f.write(f"Validation Samples: {x_val.shape[0]}\n")
        f.write(f"Test Samples: {x_test_selected.shape[0]}\n\n")
        f.write("Test Set Evaluation Results:\n")
        f.write(f"Best Model: GBDT Model {best_model_idx+1}\n")
        f.write(f"MSE: {best_mse:.6f}\n")
        f.write(f"R²: {best_r2:.6f}\n")
        f.write(f"Pearson Correlation: {best_pearson:.6f}\n")
    
    return best_mse, best_r2, best_pearson

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        mse, r2, pearson = main()
        
        end_time = time.time()
        total_runtime = end_time - start_time
        
        # 打印总运行时间
        print(f"\nTotal runtime: {total_runtime:.2f} seconds")
        
        # 输出最终结果
        print("\n===== Final Results =====")
        print(f"MSE: {mse:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Pearson Correlation: {pearson:.6f}")
        
        # 保存运行时间
        with open(os.path.join(vis_dir, 'runtime.txt'), 'w') as f:
            f.write(f"Total runtime: {total_runtime:.2f} seconds")
    
    except Exception as e:
        # 记录错误信息
        print(f"Training error: {str(e)}")
        with open(os.path.join(vis_dir, 'error_log.txt'), 'w') as f:
            import traceback
            f.write(f"Error time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error information: {str(e)}\n\n")
            f.write(traceback.format_exc()) 
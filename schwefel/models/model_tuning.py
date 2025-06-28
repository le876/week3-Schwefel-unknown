import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.load_data import get_data
from models.baseline_model import train_model

def tune_random_forest(add_squared_features=False, cv=5, n_iter=20):
    """
    调优随机森林模型
    
    参数:
    - add_squared_features: 是否添加平方特征
    - cv: 交叉验证折数
    - n_iter: 随机搜索迭代次数
    
    返回:
    - best_params: 最佳参数
    - best_score: 最佳得分
    """
    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = get_data(add_squared_features=add_squared_features)
    
    # 合并训练集和验证集用于交叉验证
    x_train_full = np.vstack((x_train, x_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    # 定义参数网格
    param_dist = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    
    # 创建随机森林模型
    rf = RandomForestRegressor(random_state=42)
    
    # 使用随机搜索进行调优
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合模型
    random_search.fit(x_train_full, y_train_full)
    
    # 获取最佳参数和得分
    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # 转换回MSE
    
    # 使用最佳参数训练模型并评估
    best_model, metrics = train_model(
        model_type='rf',
        add_squared_features=add_squared_features,
        **best_params
    )
    
    # 保存调优结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tuning_results', f"rf_tuning_{timestamp}")
    os.makedirs(tuning_dir, exist_ok=True)
    
    # 保存调优过程中的所有结果
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(os.path.join(tuning_dir, 'tuning_results.csv'), index=False)
    
    # 保存最佳参数
    with open(os.path.join(tuning_dir, 'best_params.txt'), 'w') as f:
        f.write(f"最佳参数:\n{best_params}\n\n")
        f.write(f"交叉验证MSE: {best_score:.6f}\n\n")
        f.write(f"测试集MSE: {metrics['test_mse']:.6f}\n")
        f.write(f"测试集R²: {metrics['test_r2']:.6f}\n")
        f.write(f"测试集Pearson相关系数: {metrics['test_pearson']:.6f}\n")
    
    return best_params, best_score, metrics

def tune_gradient_boosting(add_squared_features=False, cv=5, n_iter=20):
    """
    调优梯度提升回归模型
    
    参数:
    - add_squared_features: 是否添加平方特征
    - cv: 交叉验证折数
    - n_iter: 随机搜索迭代次数
    
    返回:
    - best_params: 最佳参数
    - best_score: 最佳得分
    """
    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = get_data(add_squared_features=add_squared_features)
    
    # 合并训练集和验证集用于交叉验证
    x_train_full = np.vstack((x_train, x_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    # 定义参数网格
    param_dist = {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # 创建梯度提升模型
    gbr = GradientBoostingRegressor(random_state=42)
    
    # 使用随机搜索进行调优
    random_search = RandomizedSearchCV(
        estimator=gbr,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合模型
    random_search.fit(x_train_full, y_train_full)
    
    # 获取最佳参数和得分
    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # 转换回MSE
    
    # 使用最佳参数训练模型并评估
    best_model, metrics = train_model(
        model_type='gbr',
        add_squared_features=add_squared_features,
        **best_params
    )
    
    # 保存调优结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tuning_results', f"gbr_tuning_{timestamp}")
    os.makedirs(tuning_dir, exist_ok=True)
    
    # 保存调优过程中的所有结果
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(os.path.join(tuning_dir, 'tuning_results.csv'), index=False)
    
    # 保存最佳参数
    with open(os.path.join(tuning_dir, 'best_params.txt'), 'w') as f:
        f.write(f"最佳参数:\n{best_params}\n\n")
        f.write(f"交叉验证MSE: {best_score:.6f}\n\n")
        f.write(f"测试集MSE: {metrics['test_mse']:.6f}\n")
        f.write(f"测试集R²: {metrics['test_r2']:.6f}\n")
        f.write(f"测试集Pearson相关系数: {metrics['test_pearson']:.6f}\n")
    
    return best_params, best_score, metrics

def tune_ridge(add_squared_features=False, cv=5):
    """
    调优岭回归模型
    
    参数:
    - add_squared_features: 是否添加平方特征
    - cv: 交叉验证折数
    
    返回:
    - best_params: 最佳参数
    - best_score: 最佳得分
    """
    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = get_data(add_squared_features=add_squared_features)
    
    # 合并训练集和验证集用于交叉验证
    x_train_full = np.vstack((x_train, x_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    # 定义参数网格
    param_grid = {
        'alpha': np.logspace(-4, 2, 20),
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    
    # 创建岭回归模型
    ridge = Ridge(random_state=42)
    
    # 使用网格搜索进行调优
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合模型
    grid_search.fit(x_train_full, y_train_full)
    
    # 获取最佳参数和得分
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # 转换回MSE
    
    # 使用最佳参数训练模型并评估
    best_model, metrics = train_model(
        model_type='ridge',
        add_squared_features=add_squared_features,
        **best_params
    )
    
    # 保存调优结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tuning_results', f"ridge_tuning_{timestamp}")
    os.makedirs(tuning_dir, exist_ok=True)
    
    # 保存调优过程中的所有结果
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(os.path.join(tuning_dir, 'tuning_results.csv'), index=False)
    
    # 保存最佳参数
    with open(os.path.join(tuning_dir, 'best_params.txt'), 'w') as f:
        f.write(f"最佳参数:\n{best_params}\n\n")
        f.write(f"交叉验证MSE: {best_score:.6f}\n\n")
        f.write(f"测试集MSE: {metrics['test_mse']:.6f}\n")
        f.write(f"测试集R²: {metrics['test_r2']:.6f}\n")
        f.write(f"测试集Pearson相关系数: {metrics['test_pearson']:.6f}\n")
    
    return best_params, best_score, metrics

def tune_svr(add_squared_features=False, cv=5, n_iter=20):
    """
    调优支持向量回归模型
    
    参数:
    - add_squared_features: 是否添加平方特征
    - cv: 交叉验证折数
    - n_iter: 随机搜索迭代次数
    
    返回:
    - best_params: 最佳参数
    - best_score: 最佳得分
    """
    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = get_data(add_squared_features=add_squared_features)
    
    # 合并训练集和验证集用于交叉验证
    x_train_full = np.vstack((x_train, x_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    # 定义参数网格
    param_dist = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': np.logspace(-3, 3, 10),
        'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 10)),
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]
    }
    
    # 创建SVR模型
    svr = SVR()
    
    # 使用随机搜索进行调优
    random_search = RandomizedSearchCV(
        estimator=svr,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合模型
    random_search.fit(x_train_full, y_train_full)
    
    # 获取最佳参数和得分
    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # 转换回MSE
    
    # 使用最佳参数训练模型并评估
    best_model, metrics = train_model(
        model_type='svr',
        add_squared_features=add_squared_features,
        **best_params
    )
    
    # 保存调优结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tuning_results', f"svr_tuning_{timestamp}")
    os.makedirs(tuning_dir, exist_ok=True)
    
    # 保存调优过程中的所有结果
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(os.path.join(tuning_dir, 'tuning_results.csv'), index=False)
    
    # 保存最佳参数
    with open(os.path.join(tuning_dir, 'best_params.txt'), 'w') as f:
        f.write(f"最佳参数:\n{best_params}\n\n")
        f.write(f"交叉验证MSE: {best_score:.6f}\n\n")
        f.write(f"测试集MSE: {metrics['test_mse']:.6f}\n")
        f.write(f"测试集R²: {metrics['test_r2']:.6f}\n")
        f.write(f"测试集Pearson相关系数: {metrics['test_pearson']:.6f}\n")
    
    return best_params, best_score, metrics

def tune_mlp(add_squared_features=False, cv=5, n_iter=20):
    """
    调优多层感知机模型
    
    参数:
    - add_squared_features: 是否添加平方特征
    - cv: 交叉验证折数
    - n_iter: 随机搜索迭代次数
    
    返回:
    - best_params: 最佳参数
    - best_score: 最佳得分
    """
    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = get_data(add_squared_features=add_squared_features)
    
    # 合并训练集和验证集用于交叉验证
    x_train_full = np.vstack((x_train, x_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    # 定义参数网格
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 100), (100, 50, 25)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'alpha': np.logspace(-5, 1, 10),
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'max_iter': [500, 1000, 2000]
    }
    
    # 创建MLP模型
    mlp = MLPRegressor(random_state=42)
    
    # 使用随机搜索进行调优
    random_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合模型
    random_search.fit(x_train_full, y_train_full)
    
    # 获取最佳参数和得分
    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # 转换回MSE
    
    # 使用最佳参数训练模型并评估
    best_model, metrics = train_model(
        model_type='mlp',
        add_squared_features=add_squared_features,
        **best_params
    )
    
    # 保存调优结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tuning_results', f"mlp_tuning_{timestamp}")
    os.makedirs(tuning_dir, exist_ok=True)
    
    # 保存调优过程中的所有结果
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(os.path.join(tuning_dir, 'tuning_results.csv'), index=False)
    
    # 保存最佳参数
    with open(os.path.join(tuning_dir, 'best_params.txt'), 'w') as f:
        f.write(f"最佳参数:\n{best_params}\n\n")
        f.write(f"交叉验证MSE: {best_score:.6f}\n\n")
        f.write(f"测试集MSE: {metrics['test_mse']:.6f}\n")
        f.write(f"测试集R²: {metrics['test_r2']:.6f}\n")
        f.write(f"测试集Pearson相关系数: {metrics['test_pearson']:.6f}\n")
    
    return best_params, best_score, metrics

def compare_models(add_squared_features=False):
    """
    比较不同模型的性能
    
    参数:
    - add_squared_features: 是否添加平方特征
    
    返回:
    - results: 包含各模型性能的DataFrame
    """
    models = [
        ('RandomForest', 'rf', {'n_estimators': 100, 'random_state': 42}),
        ('GradientBoosting', 'gbr', {'n_estimators': 100, 'random_state': 42}),
        ('Ridge', 'ridge', {'alpha': 1.0, 'random_state': 42}),
        ('Lasso', 'lasso', {'alpha': 1.0, 'random_state': 42}),
        ('SVR', 'svr', {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}),
        ('MLP', 'mlp', {'hidden_layer_sizes': (100,), 'random_state': 42, 'max_iter': 1000})
    ]
    
    results = []
    
    for name, model_type, params in models:
        print(f"训练模型: {name}")
        model, metrics = train_model(model_type=model_type, add_squared_features=add_squared_features, **params)
        
        results.append({
            'Model': name,
            'Train MSE': metrics['train_mse'],
            'Val MSE': metrics['val_mse'],
            'Test MSE': metrics['test_mse'],
            'Train R²': metrics['train_r2'],
            'Val R²': metrics['val_r2'],
            'Test R²': metrics['test_r2'],
            'Train Pearson': metrics['train_pearson'],
            'Val Pearson': metrics['val_pearson'],
            'Test Pearson': metrics['test_pearson']
        })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存比较结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tuning_results', f"model_comparison_{timestamp}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(comparison_dir, 'model_comparison.csv'), index=False)
    
    # 绘制比较图
    plt.figure(figsize=(15, 10))
    
    # MSE比较
    plt.subplot(2, 2, 1)
    plt.bar(results_df['Model'], results_df['Test MSE'])
    plt.title('Test MSE Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('MSE')
    
    # R²比较
    plt.subplot(2, 2, 2)
    plt.bar(results_df['Model'], results_df['Test R²'])
    plt.title('Test R² Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('R²')
    
    # Pearson相关系数比较
    plt.subplot(2, 2, 3)
    plt.bar(results_df['Model'], results_df['Test Pearson'])
    plt.title('Test Pearson Correlation Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Pearson Correlation')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target (0.9)')
    plt.legend()
    
    # 训练集vs测试集Pearson相关系数比较
    plt.subplot(2, 2, 4)
    x = np.arange(len(results_df['Model']))
    width = 0.35
    plt.bar(x - width/2, results_df['Train Pearson'], width, label='Train')
    plt.bar(x + width/2, results_df['Test Pearson'], width, label='Test')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target (0.9)')
    plt.title('Train vs Test Pearson Correlation')
    plt.xticks(x, results_df['Model'], rotation=45)
    plt.ylabel('Pearson Correlation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'model_comparison.png'))
    
    return results_df

if __name__ == "__main__":
    # 比较基线模型
    print("比较基线模型性能...")
    results = compare_models(add_squared_features=True)
    print("\n基线模型比较结果:")
    print(results[['Model', 'Test MSE', 'Test R²', 'Test Pearson']])
    
    # 调优表现最好的模型
    best_model = results.loc[results['Test Pearson'].idxmax(), 'Model']
    print(f"\n调优表现最好的模型: {best_model}")
    
    if best_model == 'RandomForest':
        best_params, best_score, metrics = tune_random_forest(add_squared_features=True)
    elif best_model == 'GradientBoosting':
        best_params, best_score, metrics = tune_gradient_boosting(add_squared_features=True)
    elif best_model == 'Ridge':
        best_params, best_score, metrics = tune_ridge(add_squared_features=True)
    elif best_model == 'SVR':
        best_params, best_score, metrics = tune_svr(add_squared_features=True)
    elif best_model == 'MLP':
        best_params, best_score, metrics = tune_mlp(add_squared_features=True)
    
    print(f"\n调优后的测试集Pearson相关系数: {metrics['test_pearson']:.6f}") 
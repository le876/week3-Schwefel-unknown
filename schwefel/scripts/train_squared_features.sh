#!/bin/bash

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建日志目录
mkdir -p logs

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_squared_features_${TIMESTAMP}.log"

echo "开始训练带平方特征的模型..." | tee -a $LOG_FILE
echo "时间: $(date)" | tee -a $LOG_FILE
echo "-----------------------------------" | tee -a $LOG_FILE

# 运行模型比较（使用平方特征）
echo "比较不同模型性能（使用平方特征）..." | tee -a $LOG_FILE
python -c "from models.model_tuning import compare_models; compare_models(add_squared_features=True)" | tee -a $LOG_FILE

echo "-----------------------------------" | tee -a $LOG_FILE

# 选择表现最好的模型进行调优
echo "调优表现最好的模型..." | tee -a $LOG_FILE
python -c "
from models.model_tuning import compare_models, tune_random_forest, tune_gradient_boosting, tune_ridge, tune_svr, tune_mlp
import pandas as pd

# 比较模型性能
results = compare_models(add_squared_features=True)
results_df = pd.DataFrame(results)

# 找出Pearson相关系数最高的模型
best_model = results_df.loc[results_df['Test Pearson'].idxmax(), 'Model']
print(f'表现最好的模型: {best_model}')

# 调优最佳模型
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

print(f'调优后的测试集Pearson相关系数: {metrics[\"test_pearson\"]:.6f}')
" | tee -a $LOG_FILE

echo "-----------------------------------" | tee -a $LOG_FILE
echo "训练完成!" | tee -a $LOG_FILE
echo "日志保存在: $LOG_FILE" 
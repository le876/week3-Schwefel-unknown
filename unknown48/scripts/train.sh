#!/bin/bash

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建日志目录
mkdir -p logs

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "开始训练基线模型..." | tee -a $LOG_FILE
echo "时间: $(date)" | tee -a $LOG_FILE
echo "-----------------------------------" | tee -a $LOG_FILE

# 运行模型比较
echo "比较不同模型性能..." | tee -a $LOG_FILE
python -c "from models.model_tuning import compare_models; compare_models(add_squared_features=False)" | tee -a $LOG_FILE

echo "-----------------------------------" | tee -a $LOG_FILE
echo "训练完成!" | tee -a $LOG_FILE
echo "日志保存在: $LOG_FILE" 
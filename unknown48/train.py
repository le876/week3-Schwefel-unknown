import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import importlib
import subprocess

# 确保必要的目录存在
os.makedirs('data/raw', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 计算Pearson相关系数
def pearson_correlation(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# 训练和评估模型
def train_and_evaluate():
    start_time = time.time()
    print("开始训练和评估模型...")
    
    # 训练基本模型
    print("\n========== 训练基本CNN模型 ==========")
    if os.path.exists('cnn_model.py'):
        try:
            # 导入并运行CNN模型训练
            cnn_module = importlib.import_module('cnn_model')
            if hasattr(cnn_module, 'main'):
                cnn_module.main()
            else:
                print("警告: cnn_model.py中没有找到main函数")
        except Exception as e:
            print(f"运行CNN模型时出错: {str(e)}")
    else:
        print("警告: 未找到cnn_model.py")
    
    # 训练混合模型
    print("\n========== 训练混合模型 ==========")
    if os.path.exists('hybrid_model.py'):
        try:
            # 导入并运行混合模型训练
            hybrid_module = importlib.import_module('hybrid_model')
            if hasattr(hybrid_module, 'main'):
                hybrid_module.main()
            else:
                print("警告: hybrid_model.py中没有找到main函数")
        except Exception as e:
            print(f"运行混合模型时出错: {str(e)}")
    else:
        print("警告: 未找到hybrid_model.py")
        
    # 训练2D CNN模型
    print("\n========== 训练2D CNN模型 ==========")
    if os.path.exists('cnn_model_2d.py'):
        try:
            # 导入并运行2D CNN模型训练
            cnn_2d_module = importlib.import_module('cnn_model_2d')
            if hasattr(cnn_2d_module, 'main'):
                cnn_2d_module.main()
            else:
                print("警告: cnn_model_2d.py中没有找到main函数")
        except Exception as e:
            print(f"运行2D CNN模型时出错: {str(e)}")
    else:
        print("警告: 未找到cnn_model_2d.py")
        
    # 训练2D混合模型
    print("\n========== 训练2D混合模型 ==========")
    if os.path.exists('hybrid_model_2d.py'):
        try:
            # 导入并运行2D混合模型训练
            hybrid_2d_module = importlib.import_module('hybrid_model_2d')
            if hasattr(hybrid_2d_module, 'main'):
                hybrid_2d_module.main()
            else:
                print("警告: hybrid_model_2d.py中没有找到main函数")
        except Exception as e:
            print(f"运行2D混合模型时出错: {str(e)}")
    else:
        print("警告: 未找到hybrid_model_2d.py")
    
    # 收集和比较结果
    results = []
    
    # 检查CNN模型结果
    if os.path.exists('results/cnn_model_metrics.csv'):
        cnn_results = pd.read_csv('results/cnn_model_metrics.csv')
        if not cnn_results.empty:
            results.append({
                'model': 'CNN模型',
                'pearson': cnn_results['pearson'].iloc[0],
                'mse': cnn_results['mse'].iloc[0],
                'r2': cnn_results['r2'].iloc[0]
            })
    
    # 检查混合模型结果
    if os.path.exists('results/hybrid_model_metrics.csv'):
        hybrid_results = pd.read_csv('results/hybrid_model_metrics.csv')
        if not hybrid_results.empty:
            results.append({
                'model': '混合模型',
                'pearson': hybrid_results['pearson'].iloc[0],
                'mse': hybrid_results['mse'].iloc[0],
                'r2': hybrid_results['r2'].iloc[0]
            })
            
    # 检查2D CNN模型结果
    if os.path.exists('results/cnn_model_2d_metrics.csv'):
        cnn_2d_results = pd.read_csv('results/cnn_model_2d_metrics.csv')
        if not cnn_2d_results.empty:
            results.append({
                'model': '2D CNN模型',
                'pearson': cnn_2d_results['pearson'].iloc[0],
                'mse': cnn_2d_results['mse'].iloc[0],
                'r2': cnn_2d_results['r2'].iloc[0]
            })
    
    # 检查2D混合模型结果
    if os.path.exists('results/hybrid_model_2d_metrics.csv'):
        hybrid_2d_results = pd.read_csv('results/hybrid_model_2d_metrics.csv')
        if not hybrid_2d_results.empty:
            results.append({
                'model': '2D混合模型',
                'pearson': hybrid_2d_results['pearson'].iloc[0],
                'mse': hybrid_2d_results['mse'].iloc[0],
                'r2': hybrid_2d_results['r2'].iloc[0]
            })
    
    if results:
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 打印模型性能比较
        print("\n========== 模型性能比较 ==========")
        print(results_df)
        
        # 找出最佳模型
        best_idx = results_df['pearson'].idxmax()
        best_model = results_df.iloc[best_idx]
        print(f"\n最佳模型: {best_model['model']}")
        print(f"Pearson相关系数: {best_model['pearson']:.6f}")
        print(f"MSE: {best_model['mse']:.6f}")
        print(f"R²: {best_model['r2']:.6f}")
        
        # 检查是否达到目标
        if best_model['pearson'] >= 0.9:
            print(f"\n成功达到Pearson相关系数≥0.9的目标！")
        else:
            print(f"\n未达到Pearson相关系数≥0.9的目标。最高相关系数: {best_model['pearson']:.6f}")
            
        # 绘制模型性能比较图
        plt.figure(figsize=(12, 8))
        
        # 性能条形图
        bar_width = 0.25
        index = np.arange(len(results_df))
        
        plt.bar(index, results_df['pearson'], bar_width, label='Pearson相关系数')
        plt.bar(index + bar_width, results_df['r2'], bar_width, label='R²')
        
        plt.xlabel('模型')
        plt.ylabel('性能指标')
        plt.title('模型性能比较')
        plt.xticks(index + bar_width / 2, results_df['model'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加目标线
        plt.axhline(y=0.9, color='r', linestyle='--', label='目标 Pearson = 0.9')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png')
        plt.close()
        
        # 保存最终比较结果
        results_df.to_csv('results/final_comparison.csv', index=False)
    else:
        print("\n没有找到任何模型的结果")
    
    # 计算总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n训练和评估总耗时: {int(hours)}时{int(minutes)}分{seconds:.2f}秒")

if __name__ == "__main__":
    train_and_evaluate() 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import torch
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

class DecomposedNetworkVisualizer:
    """分解神经网络可视化工具类"""
    
    def __init__(self, output_dir=None):
        """初始化可视化工具"""
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'schwefel/visualizations/{timestamp}'
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # 设置绘图样式
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.grid'] = True
    
    def visualize_network_structure(self, model, dim_idx=0):
        """可视化网络结构
        
        Args:
            model: 分解神经网络模型
            dim_idx: 要可视化的维度索引
        """
        # 获取指定维度的子网络
        if hasattr(model, 'sub_networks'):
            subnet = model.sub_networks[dim_idx]
        else:
            subnet = model
        
        # 获取网络结构信息
        layers = []
        for name, module in subnet.named_modules():
            if isinstance(module, torch.nn.Linear):
                layers.append({
                    'name': name,
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'weights': module.weight.detach().cpu().numpy()
                })
        
        # 绘制网络结构
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 设置层之间的距离
        layer_distance = 3.0
        node_distance = 0.5
        max_nodes = max([layer['out_features'] for layer in layers] + [layers[0]['in_features']])
        
        # 绘制输入层
        input_nodes = layers[0]['in_features']
        input_positions = []
        for i in range(input_nodes):
            y_pos = (max_nodes - input_nodes) * node_distance / 2 + i * node_distance
            input_positions.append((0, y_pos))
            circle = plt.Circle((0, y_pos), 0.2, color='skyblue', ec='blue', zorder=2)
            ax.add_artist(circle)
            if input_nodes <= 10:  # 只有在节点数较少时才添加标签
                ax.text(0, y_pos, f'X_{i}', ha='center', va='center', fontsize=9)
        
        # 绘制隐藏层和输出层
        prev_positions = input_positions
        for l, layer in enumerate(layers):
            x_pos = (l + 1) * layer_distance
            current_positions = []
            
            for i in range(layer['out_features']):
                y_pos = (max_nodes - layer['out_features']) * node_distance / 2 + i * node_distance
                current_positions.append((x_pos, y_pos))
                
                # 绘制节点
                if l == len(layers) - 1:  # 输出层
                    circle = plt.Circle((x_pos, y_pos), 0.2, color='lightgreen', ec='green', zorder=2)
                else:  # 隐藏层
                    circle = plt.Circle((x_pos, y_pos), 0.2, color='lightyellow', ec='orange', zorder=2)
                ax.add_artist(circle)
                
                # 计算权重归一化值用于连接线透明度
                if layer['weights'].shape[0] > 0:
                    weights = layer['weights'][i, :]
                    abs_weights = np.abs(weights)
                    max_weight = np.max(abs_weights) if np.max(abs_weights) > 0 else 1.0
                    norm_weights = abs_weights / max_weight
                    
                    # 绘制连接
                    for j, (prev_x, prev_y) in enumerate(prev_positions):
                        weight = weights[j]
                        alpha = norm_weights[j] * 0.8 + 0.2  # 确保最小透明度
                        color = 'red' if weight < 0 else 'blue'
                        line = plt.Line2D([prev_x, x_pos], [prev_y, y_pos], 
                                         color=color, alpha=alpha, zorder=1, lw=1.5*norm_weights[j])
                        ax.add_artist(line)
            
            prev_positions = current_positions
        
        # 设置图形属性
        ax.set_xlim(-layer_distance, (len(layers) + 1) * layer_distance)
        ax.set_ylim(-1, max_nodes * node_distance + 1)
        ax.axis('off')
        
        # 添加图例
        input_patch = plt.Circle((0, 0), 0.2, color='skyblue', ec='blue')
        hidden_patch = plt.Circle((0, 0), 0.2, color='lightyellow', ec='orange')
        output_patch = plt.Circle((0, 0), 0.2, color='lightgreen', ec='green')
        pos_line = plt.Line2D([0], [0], color='blue', lw=2)
        neg_line = plt.Line2D([0], [0], color='red', lw=2)
        
        ax.legend([input_patch, hidden_patch, output_patch, pos_line, neg_line],
                 ['输入层', '隐藏层', '输出层', '正权重', '负权重'],
                 loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=5)
        
        plt.title(f'维度 {dim_idx} 神经网络结构')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dim_{dim_idx}_network_structure.png', dpi=300)
        plt.close()
        
        print(f"网络结构可视化已保存至: {self.output_dir}/dim_{dim_idx}_network_structure.png")
    
    def visualize_weights_distribution(self, model):
        """可视化所有子网络权重分布"""
        if not hasattr(model, 'sub_networks'):
            print("模型不包含多个子网络，无法进行比较可视化")
            return
        
        n_dims = len(model.sub_networks)
        weights_stats = []
        
        for dim in range(n_dims):
            subnet = model.sub_networks[dim]
            subnet_weights = []
            
            for name, param in subnet.named_parameters():
                if 'weight' in name:
                    subnet_weights.extend(param.detach().cpu().numpy().flatten())
            
            weights_stats.append({
                'dimension': dim,
                'mean': np.mean(subnet_weights),
                'std': np.std(subnet_weights),
                'min': np.min(subnet_weights),
                'max': np.max(subnet_weights),
                'weights': subnet_weights
            })
        
        # 创建权重分布对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # 均值分布
        means = [stat['mean'] for stat in weights_stats]
        axes[0].bar(range(n_dims), means)
        axes[0].set_title('各维度权重均值')
        axes[0].set_xlabel('维度')
        axes[0].set_ylabel('权重均值')
        
        # 标准差分布
        stds = [stat['std'] for stat in weights_stats]
        axes[1].bar(range(n_dims), stds)
        axes[1].set_title('各维度权重标准差')
        axes[1].set_xlabel('维度')
        axes[1].set_ylabel('权重标准差')
        
        # 最大值分布
        maxs = [stat['max'] for stat in weights_stats]
        axes[2].bar(range(n_dims), maxs)
        axes[2].set_title('各维度权重最大值')
        axes[2].set_xlabel('维度')
        axes[2].set_ylabel('权重最大值')
        
        # 箱线图比较
        boxplot_data = [stat['weights'] for stat in weights_stats]
        axes[3].boxplot(boxplot_data)
        axes[3].set_title('各维度权重分布箱线图')
        axes[3].set_xlabel('维度')
        axes[3].set_ylabel('权重值')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/weights_distribution.png')
        plt.close()
        
        print(f"权重分布可视化已保存至: {self.output_dir}/weights_distribution.png")
    
    def visualize_dimension_importance(self, importance_metrics, sort_by=None):
        """可视化维度重要性
        
        Args:
            importance_metrics: 包含多种重要性指标的字典
                {
                    'abs_mean_contribution': [...],
                    'variance_contribution': [...],
                    'performance_impact': [...],
                    'correlation': [...]
                }
            sort_by: 排序依据的指标名称
        """
        n_dims = len(importance_metrics[list(importance_metrics.keys())[0]])
        
        # 创建DataFrame以便于处理
        df = pd.DataFrame({
            'dimension': range(n_dims)
        })
        
        # 添加各种重要性指标
        for metric_name, values in importance_metrics.items():
            df[metric_name] = values
        
        # 如果指定了排序依据，按该指标排序
        if sort_by is not None and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
        
        # 1. 绘制条形图对比各指标
        metrics = list(importance_metrics.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            sorted_df = df.sort_values(by=metric, ascending=False).reset_index(drop=True)
            axes[i].bar(sorted_df['dimension'], sorted_df[metric])
            axes[i].set_title(f'维度重要性 - {metric}')
            axes[i].set_xlabel('维度')
            axes[i].set_ylabel('重要性得分')
            
            # 添加数值标签
            for j, v in enumerate(sorted_df[metric]):
                axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dimension_importance_bars.png')
        plt.close()
        
        # 2. 绘制热力图
        # 对每个指标进行归一化处理
        normalized_df = df.copy()
        for metric in metrics:
            if metric != 'dimension':
                normalized_df[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        heatmap_data = normalized_df.drop('dimension', axis=1).T
        heatmap_data.columns = normalized_df['dimension']
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=.5)
        plt.title('维度重要性热力图（归一化）')
        plt.xlabel('维度')
        plt.ylabel('评估指标')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dimension_importance_heatmap.png')
        plt.close()
        
        # 3. 绘制雷达图 (选择前5个维度)
        top_dims = df.nlargest(5, metrics[0])['dimension'].values
        radar_df = normalized_df[normalized_df['dimension'].isin(top_dims)]
        
        # 准备雷达图数据
        labels = metrics
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for _, row in radar_df.iterrows():
            dim = int(row['dimension'])
            values = row.drop('dimension').values.tolist()
            values += values[:1]  # 闭合雷达图
            
            ax.plot(angles, values, linewidth=2, label=f'维度 {dim}')
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('前5个重要维度的雷达图比较')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dimension_importance_radar.png')
        plt.close()
        
        # 4. 生成综合权重
        # 用所有指标的均值作为综合权重
        df['综合权重'] = df.drop('dimension', axis=1).mean(axis=1)
        df = df.sort_values(by='综合权重', ascending=False).reset_index(drop=True)
        
        plt.figure(figsize=(12, 6))
        plt.bar(df['dimension'], df['综合权重'])
        plt.title('维度综合重要性权重')
        plt.xlabel('维度')
        plt.ylabel('综合权重')
        
        # 添加数值标签
        for i, v in enumerate(df['综合权重']):
            plt.text(df['dimension'].iloc[i], v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dimension_importance_combined.png')
        plt.close()
        
        # 保存维度重要性结果
        df.to_csv(f'{self.output_dir}/dimension_importance.csv', index=False)
        
        print(f"维度重要性可视化已保存至: {self.output_dir}")
        
        return df
    
    def visualize_3d_prediction_surface(self, model, dim1=0, dim2=1, resolution=50, range_min=-500, range_max=500):
        """可视化两个维度组合的预测表面
        
        Args:
            model: 神经网络模型
            dim1: 第一个维度索引
            dim2: 第二个维度索引
            resolution: 分辨率
            range_min: 输入值范围最小值
            range_max: 输入值范围最大值
        """
        # 创建网格
        x = np.linspace(range_min, range_max, resolution)
        y = np.linspace(range_min, range_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # 准备输入数据
        n_dims = 20  # Schwefel函数的维度数
        inputs = np.zeros((resolution * resolution, n_dims))
        
        # 设置选定的两个维度
        grid_values = np.column_stack((X.flatten(), Y.flatten()))
        inputs[:, dim1] = grid_values[:, 0]
        inputs[:, dim2] = grid_values[:, 1]
        
        # 转换为tensor并预测
        with torch.no_grad():
            model_inputs = torch.FloatTensor(inputs)
            predictions = model(model_inputs).cpu().numpy()
        
        # 重塑为网格形状
        Z = predictions.reshape(resolution, resolution)
        
        # 可视化3D表面
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建自定义颜色映射
        colors = [(0.0, 'blue'), (0.5, 'white'), (1.0, 'red')]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        
        # 绘制3D表面
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.8)
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # 设置标题和标签
        ax.set_title(f'维度 {dim1} 和维度 {dim2} 的预测表面')
        ax.set_xlabel(f'维度 {dim1}')
        ax.set_ylabel(f'维度 {dim2}')
        ax.set_zlabel('预测值')
        
        # 保存图像
        plt.savefig(f'{self.output_dir}/prediction_surface_dim{dim1}_dim{dim2}.png', dpi=300)
        plt.close()
        
        print(f"预测表面可视化已保存至: {self.output_dir}/prediction_surface_dim{dim1}_dim{dim2}.png")
    
    def visualize_contribution_distribution(self, contributions, true_values=None):
        """可视化各维度贡献分布
        
        Args:
            contributions: 形状为(样本数, 维度数)的贡献数组
            true_values: 可选，真实总值
        """
        n_samples, n_dims = contributions.shape
        
        # 1. 计算每个维度的贡献统计量
        stats = pd.DataFrame({
            'dimension': range(n_dims),
            'mean': np.mean(contributions, axis=0),
            'abs_mean': np.mean(np.abs(contributions), axis=0),
            'std': np.std(contributions, axis=0),
            'min': np.min(contributions, axis=0),
            'max': np.max(contributions, axis=0),
        })
        
        # 按绝对平均贡献排序
        stats = stats.sort_values(by='abs_mean', ascending=False).reset_index(drop=True)
        
        # 2. 绘制贡献分布条形图
        plt.figure(figsize=(12, 6))
        plt.bar(stats['dimension'], stats['abs_mean'], alpha=0.7)
        plt.errorbar(stats['dimension'], stats['abs_mean'], stats['std'], fmt='none', color='black', alpha=0.5)
        
        plt.title('各维度绝对平均贡献值')
        plt.xlabel('维度')
        plt.ylabel('绝对平均贡献值')
        plt.xticks(stats['dimension'])
        plt.grid(axis='y')
        
        # 添加数值标签
        for i, v in enumerate(stats['abs_mean']):
            plt.text(stats['dimension'].iloc[i], v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/contribution_distribution.png')
        plt.close()
        
        # 3. 绘制贡献热力图
        # 选择前10个样本的贡献
        if n_samples > 10:
            sample_indices = np.random.choice(n_samples, 10, replace=False)
            sample_contributions = contributions[sample_indices]
        else:
            sample_contributions = contributions
        
        # 按统计量中的维度顺序排列
        ordered_dims = stats['dimension'].values
        sample_contributions = sample_contributions[:, ordered_dims]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(sample_contributions, cmap='coolwarm', center=0, 
                   xticklabels=[f'D{i}' for i in ordered_dims],
                   yticklabels=[f'S{i}' for i in range(len(sample_contributions))])
        
        plt.title('样本-维度贡献热力图')
        plt.xlabel('维度')
        plt.ylabel('样本')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/contribution_heatmap.png')
        plt.close()
        
        # 4. 如果提供了真实值，绘制贡献堆叠图
        if true_values is not None and len(true_values) >= 10:
            # 选择前10个样本
            if n_samples > 10:
                indices = np.random.choice(n_samples, 10, replace=False)
                sample_true = true_values[indices]
                sample_contribs = contributions[indices]
            else:
                indices = range(len(true_values))
                sample_true = true_values
                sample_contribs = contributions
            
            # 计算累积贡献
            plt.figure(figsize=(14, 8))
            bottom = np.zeros(len(indices))
            
            for dim in ordered_dims:
                plt.bar(range(len(indices)), sample_contribs[:, dim], bottom=bottom, label=f'维度 {dim}')
                bottom += sample_contribs[:, dim]
            
            # 添加真实值线
            plt.plot(range(len(indices)), sample_true, 'k--', marker='o', label='真实值')
            
            plt.title('样本贡献堆叠图与真实值对比')
            plt.xlabel('样本索引')
            plt.ylabel('贡献值')
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/contribution_stacked.png')
            plt.close()
        
        # 5. 保存贡献统计信息
        stats.to_csv(f'{self.output_dir}/contribution_stats.csv', index=False)
        
        print(f"贡献分布可视化已保存至: {self.output_dir}")
        
        return stats

# 示例用法
if __name__ == '__main__':
    # 创建输出目录
    visualizer = DecomposedNetworkVisualizer()
    
    # 随机生成一些贡献数据用于测试
    np.random.seed(42)
    contributions = np.random.normal(0, 100, (100, 20))
    true_values = np.sum(contributions, axis=1) + np.random.normal(0, 10, 100)
    
    # 可视化贡献分布
    visualizer.visualize_contribution_distribution(contributions, true_values)
    
    # 构建重要性指标示例
    importance_metrics = {
        'abs_mean_contribution': np.mean(np.abs(contributions), axis=0),
        'variance_contribution': np.var(contributions, axis=0),
        'performance_impact': np.random.rand(20),  # 模拟性能影响
        'correlation': np.abs(np.random.rand(20))  # 模拟相关系数
    }
    
    # 可视化维度重要性
    visualizer.visualize_dimension_importance(importance_metrics, sort_by='abs_mean_contribution') 
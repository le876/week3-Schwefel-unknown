import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Input, Concatenate, GlobalAveragePooling2D, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import os

# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 创建输出目录
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 加载数据
def load_data():
    print("加载数据...")
    x_train = np.load('data/raw/x_48_train(1).npy')
    y_train = np.load('data/raw/y_48_train(1).npy')
    x_test = np.load('data/raw/x_48_test(1).npy')
    y_test = np.load('data/raw/y_48_test(1).npy')
    
    print(f"训练集形状: {x_train.shape}, {y_train.shape}")
    print(f"测试集形状: {x_test.shape}, {y_test.shape}")
    
    return x_train, y_train, x_test, y_test

# 处理数据为2D形式
def reshape_to_2d(x_data):
    # 从(samples, seq_length, features)转换为(samples, seq_length, features, 1)
    # 这种形式适合2D卷积
    return np.expand_dims(x_data, axis=-1)

# 提取统计特征
def extract_statistical_features(x_data):
    n_samples, seq_length, n_features = x_data.shape
    
    # 基本统计特征
    x_mean = np.mean(x_data, axis=1)  # 均值
    x_std = np.std(x_data, axis=1)    # 标准差
    x_max = np.max(x_data, axis=1)    # 最大值
    x_min = np.min(x_data, axis=1)    # 最小值
    x_median = np.median(x_data, axis=1)  # 中位数
    
    # 高级统计特征
    x_range = x_max - x_min  # 范围
    x_quantile25 = np.quantile(x_data, 0.25, axis=1)  # 25%分位数
    x_quantile75 = np.quantile(x_data, 0.75, axis=1)  # 75%分位数
    x_iqr = x_quantile75 - x_quantile25  # 四分位距
    
    # 计算每个特征的斜度和峰度
    x_skew = np.zeros((n_samples, n_features))
    x_kurtosis = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        for j in range(n_features):
            x_skew[i, j] = np.mean(((x_data[i, :, j] - x_mean[i, j]) / x_std[i, j]) ** 3) if x_std[i, j] > 0 else 0
            x_kurtosis[i, j] = np.mean(((x_data[i, :, j] - x_mean[i, j]) / x_std[i, j]) ** 4) if x_std[i, j] > 0 else 0
    
    # 计算时间序列的趋势（使用简单的线性回归斜率）
    x_trend = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            # 简单线性回归: y = ax + b
            x_vals = np.arange(seq_length)
            y_vals = x_data[i, :, j]
            x_mean_val = np.mean(x_vals)
            y_mean_val = np.mean(y_vals)
            numerator = np.sum((x_vals - x_mean_val) * (y_vals - y_mean_val))
            denominator = np.sum((x_vals - x_mean_val) ** 2)
            x_trend[i, j] = numerator / denominator if denominator != 0 else 0
    
    # 自相关特征 (lag-1自相关)
    x_autocorr = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            series = x_data[i, :, j]
            series_shifted = np.concatenate([np.array([0]), series[:-1]])
            # 计算lag-1自相关
            numerator = np.sum((series[1:] - np.mean(series[1:])) * (series_shifted[1:] - np.mean(series_shifted[1:])))
            denominator = np.sqrt(np.sum((series[1:] - np.mean(series[1:])) ** 2) * np.sum((series_shifted[1:] - np.mean(series_shifted[1:])) ** 2))
            x_autocorr[i, j] = numerator / denominator if denominator != 0 else 0
    
    # 交叉特征（特征之间的互相关）
    x_cross_corr = np.zeros((n_samples, n_features * (n_features - 1) // 2))
    idx = 0
    for i in range(n_features):
        for j in range(i+1, n_features):
            for k in range(n_samples):
                series1 = x_data[k, :, i]
                series2 = x_data[k, :, j]
                # 计算相关系数
                numerator = np.sum((series1 - np.mean(series1)) * (series2 - np.mean(series2)))
                denominator = np.sqrt(np.sum((series1 - np.mean(series1)) ** 2) * np.sum((series2 - np.mean(series2)) ** 2))
                x_cross_corr[k, idx] = numerator / denominator if denominator != 0 else 0
            idx += 1
    
    # 波动性特征 (变化率)
    x_volatility = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            series = x_data[i, :, j]
            # 计算相邻点之间的变化
            changes = np.diff(series)
            # 计算变化的绝对值的平均值
            x_volatility[i, j] = np.mean(np.abs(changes)) if len(changes) > 0 else 0
    
    # 周期性特征 (使用自相关的方法简单估计)
    x_cyclicality = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            series = x_data[i, :, j]
            # 计算滞后2-10的自相关，找出最大值
            max_acf = 0
            for lag in range(2, min(11, seq_length // 2)):
                series_lagged = np.concatenate([np.zeros(lag), series[:-lag]])
                # 计算自相关
                numerator = np.sum((series[lag:] - np.mean(series[lag:])) * (series_lagged[lag:] - np.mean(series_lagged[lag:])))
                denominator = np.sqrt(np.sum((series[lag:] - np.mean(series[lag:])) ** 2) * 
                                    np.sum((series_lagged[lag:] - np.mean(series_lagged[lag:])) ** 2))
                acf = numerator / denominator if denominator != 0 else 0
                max_acf = max(max_acf, abs(acf))
            x_cyclicality[i, j] = max_acf
    
    # 合并所有统计特征
    statistical_features = np.concatenate([
        x_mean, x_std, x_max, x_min, x_median, 
        x_range, x_quantile25, x_quantile75, x_iqr,
        x_skew, x_kurtosis, x_trend, x_autocorr, 
        x_cross_corr, x_volatility, x_cyclicality
    ], axis=1)
    
    return statistical_features

# 构建增强的2D CNN子模型
def build_enhanced_cnn_submodel(input_shape):
    input_layer = Input(shape=input_shape)
    
    # 时间维度上的卷积 (捕捉时序模式)
    time_conv1 = Conv2D(64, kernel_size=(5, 3), padding='same', activation='relu')(input_layer)
    time_bn1 = BatchNormalization()(time_conv1)
    time_conv2 = Conv2D(64, kernel_size=(5, 3), padding='same', activation='relu')(time_bn1)
    time_bn2 = BatchNormalization()(time_conv2)
    time_pool = MaxPooling2D(pool_size=(2, 1))(time_bn2)  # 只在时间维度上池化
    
    # 特征维度上的卷积 (捕捉特征间关系)
    feat_conv1 = Conv2D(64, kernel_size=(3, 5), padding='same', activation='relu')(input_layer)
    feat_bn1 = BatchNormalization()(feat_conv1)
    feat_conv2 = Conv2D(64, kernel_size=(3, 5), padding='same', activation='relu')(feat_bn1)
    feat_bn2 = BatchNormalization()(feat_conv2)
    feat_pool = MaxPooling2D(pool_size=(1, 2))(feat_bn2)  # 只在特征维度上池化
    
    # 同时捕捉时序和特征关系的卷积
    joint_conv1 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    joint_bn1 = BatchNormalization()(joint_conv1)
    joint_conv2 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu')(joint_bn1)
    joint_bn2 = BatchNormalization()(joint_conv2)
    joint_pool = MaxPooling2D(pool_size=(2, 2))(joint_bn2)
    
    # 合并三个分支
    concat = Concatenate()([time_pool, feat_pool, joint_pool])
    
    # 深层卷积层
    conv1 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(concat)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    
    conv2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    bn2 = BatchNormalization()(conv2)
    gap = GlobalAveragePooling2D()(bn2)
    
    return input_layer, gap

# 构建混合模型（2D CNN + 统计特征）
def build_hybrid_model_2d(sequence_input_shape, statistical_input_shape):
    # 2D CNN分支 - 处理时序数据
    sequence_input, cnn_features = build_enhanced_cnn_submodel(sequence_input_shape)
    
    # 统计特征分支 - 处理统计特征
    statistical_input = Input(shape=(statistical_input_shape,), name='statistical_input')
    
    # 统计特征处理
    dense_stat1 = Dense(256, activation='relu')(statistical_input)
    bn_stat1 = BatchNormalization()(dense_stat1)
    drop_stat1 = Dropout(0.3)(bn_stat1)
    
    dense_stat2 = Dense(128, activation='relu')(drop_stat1)
    bn_stat2 = BatchNormalization()(dense_stat2)
    drop_stat2 = Dropout(0.3)(bn_stat2)
    
    # 合并CNN特征和统计特征
    concat_all = Concatenate()([cnn_features, drop_stat2])
    
    # 全连接层处理合并特征
    dense1 = Dense(256, activation='relu')(concat_all)
    bn1 = BatchNormalization()(dense1)
    drop1 = Dropout(0.5)(bn1)
    
    dense2 = Dense(128, activation='relu')(drop1)
    bn2 = BatchNormalization()(dense2)
    drop2 = Dropout(0.3)(bn2)
    
    dense3 = Dense(64, activation='relu')(drop2)
    bn3 = BatchNormalization()(dense3)
    drop3 = Dropout(0.2)(bn3)
    
    # 输出层
    output = Dense(1, activation='linear')(drop3)
    
    # 创建模型
    model = Model(inputs=[sequence_input, statistical_input], outputs=output)
    
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

# 计算Pearson相关系数
def pearson_correlation(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# 训练混合模型
def train_hybrid_model_2d(model, x_train_seq, x_train_stat, y_train, x_val_seq, x_val_stat, y_val, epochs=150, batch_size=32):
    print("训练2D混合模型...")
    
    # 定义回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ModelCheckpoint('models/hybrid_model_2d_best.h5', monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)
    ]
    
    # 训练模型
    history = model.fit(
        {'input_1': x_train_seq, 'statistical_input': x_train_stat},
        y_train,
        validation_data=({'input_1': x_val_seq, 'statistical_input': x_val_stat}, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存训练历史
    pd.DataFrame(history.history).to_csv('results/hybrid_model_2d_history.csv', index=False)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('2D Hybrid Model - Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/hybrid_model_2d_loss_curve.png')
    plt.close()
    
    return model, history

# 评估混合模型
def evaluate_hybrid_model_2d(model, x_test_seq, x_test_stat, y_test):
    print("评估2D混合模型...")
    
    # 预测
    y_pred = model.predict({'input_1': x_test_seq, 'statistical_input': x_test_stat}).flatten()
    
    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson = pearson_correlation(y_test, y_pred)
    
    # 打印结果
    print(f"2D混合模型 - 测试集MSE: {mse:.6f}")
    print(f"2D混合模型 - 测试集R²: {r2:.6f}")
    print(f"2D混合模型 - 测试集Pearson相关系数: {pearson:.6f}")
    
    # 保存结果
    results = {
        'model': 'hybrid_model_2d',
        'mse': mse,
        'r2': r2,
        'pearson': pearson
    }
    pd.DataFrame([results]).to_csv('results/hybrid_model_2d_metrics.csv', index=False)
    
    # 绘制预测vs真实值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'2D Hybrid Model - Predictions vs Actual (Pearson: {pearson:.4f})')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('results/hybrid_model_2d_predictions.png')
    plt.close()
    
    return pearson

# 数据预处理
def preprocess_data(x_train, y_train, x_test, y_test, val_split=0.2):
    # 从训练集划分验证集
    val_size = int(len(x_train) * val_split)
    indices = np.random.permutation(len(x_train))
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    
    # 提取统计特征
    print("提取统计特征...")
    x_train_stat = extract_statistical_features(x_train)
    x_val_stat = extract_statistical_features(x_val)
    x_test_stat = extract_statistical_features(x_test)
    
    # 转换为2D格式（添加通道维度）
    x_train_2d = reshape_to_2d(x_train)
    x_val_2d = reshape_to_2d(x_val)
    x_test_2d = reshape_to_2d(x_test)
    
    # 标准化统计特征
    print("标准化统计特征...")
    scaler = StandardScaler()
    x_train_stat = scaler.fit_transform(x_train_stat)
    x_val_stat = scaler.transform(x_val_stat)
    x_test_stat = scaler.transform(x_test_stat)
    
    print(f"训练集形状(划分后): 序列={x_train_2d.shape}, 统计特征={x_train_stat.shape}, 目标={y_train.shape}")
    print(f"验证集形状: 序列={x_val_2d.shape}, 统计特征={x_val_stat.shape}, 目标={y_val.shape}")
    print(f"测试集形状: 序列={x_test_2d.shape}, 统计特征={x_test_stat.shape}, 目标={y_test.shape}")
    
    return x_train_2d, x_train_stat, y_train, x_val_2d, x_val_stat, y_val, x_test_2d, x_test_stat, y_test

# 主函数
def main():
    # 加载数据
    x_train, y_train, x_test, y_test = load_data()
    
    # 数据预处理
    x_train_2d, x_train_stat, y_train, x_val_2d, x_val_stat, y_val, x_test_2d, x_test_stat, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # 获取输入形状
    sequence_input_shape = x_train_2d.shape[1:]  # (seq_length, features, 1)
    statistical_input_shape = x_train_stat.shape[1]
    print(f"序列输入形状: {sequence_input_shape}")
    print(f"统计特征输入形状: {statistical_input_shape}")
    
    # 建立并训练混合模型
    hybrid_model = build_hybrid_model_2d(sequence_input_shape, statistical_input_shape)
    hybrid_model.summary()
    
    hybrid_model, _ = train_hybrid_model_2d(
        hybrid_model, x_train_2d, x_train_stat, y_train, 
        x_val_2d, x_val_stat, y_val
    )
    
    # 评估混合模型
    pearson_score = evaluate_hybrid_model_2d(hybrid_model, x_test_2d, x_test_stat, y_test)
    
    # 检查是否达到目标
    if pearson_score >= 0.9:
        print(f"\n成功达到Pearson相关系数≥0.9的目标! 当前相关系数: {pearson_score:.6f}")
    else:
        print(f"\n未达到Pearson相关系数≥0.9的目标。当前相关系数: {pearson_score:.6f}")
        print("建议尝试以下改进方法:")
        print("1. 增加模型复杂度或进一步调整架构")
        print("2. 进一步优化特征工程，增加更多时序相关特征")
        print("3. 尝试集成多个模型的预测结果")
        print("4. 尝试更复杂的学习率调度和正则化策略")

if __name__ == "__main__":
    main() 
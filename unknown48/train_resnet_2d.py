import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, add, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置matplotlib使用默认字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 创建必要的目录
os.makedirs('data/raw', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# 创建以时间命名的可视化输出目录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
vis_dir = os.path.join('visualizations', timestamp)
os.makedirs(vis_dir, exist_ok=True)
print(f"Visualization output will be saved to: {vis_dir}")

# 定义Pearson相关系数计算函数
def pearson_correlation(y_true, y_pred):
    return pearsonr(y_true.flatten(), y_pred.flatten())[0]

# 定义自定义回调函数，在每个epoch结束后计算并打印Pearson相关系数
class PearsonCallback(Callback):
    def __init__(self, validation_data, training_data=None, vis_dir=None):
        super(PearsonCallback, self).__init__()
        self.x_val, self.y_val = validation_data
        self.training_data = training_data
        self.best_val_pearson = -1.0
        self.vis_dir = vis_dir
        self.train_pearson_history = []
        self.val_pearson_history = []
        self.epochs = []
        
    def on_epoch_end(self, epoch, logs=None):
        y_pred_val = self.model.predict(self.x_val, verbose=0)
        val_pearson = pearson_correlation(self.y_val, y_pred_val)
        
        # 训练集Pearson相关系数
        train_pearson = None
        if self.training_data is not None:
            x_train, y_train = self.training_data
            y_pred_train = self.model.predict(x_train, verbose=0)
            train_pearson = pearson_correlation(y_train, y_pred_train)
            logs['train_pearson'] = train_pearson
            self.train_pearson_history.append(train_pearson)
        
        logs['val_pearson'] = val_pearson
        self.val_pearson_history.append(val_pearson)
        self.epochs.append(epoch)
        
        # 记录最佳Pearson相关系数
        if val_pearson > self.best_val_pearson:
            self.best_val_pearson = val_pearson
            
            # 在达到新的最佳Pearson值时生成预测图
            if self.vis_dir:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.y_val, y_pred_val, alpha=0.5)
                plt.plot([self.y_val.min(), self.y_val.max()], 
                         [self.y_val.min(), self.y_val.max()], 'r--')
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Validation Set: Predicted vs True (Pearson = {val_pearson:.4f})')
                plt.savefig(os.path.join(self.vis_dir, f'best_val_predictions_epoch_{epoch+1}.png'))
                plt.close()
        
        print(f"\nEpoch {epoch+1} - val_pearson: {val_pearson:.4f}", end="")
        if train_pearson is not None:
            print(f" - train_pearson: {train_pearson:.4f}", end="")
        print(f" - best_val_pearson: {self.best_val_pearson:.4f}")
        
        # 每10个epoch更新一次Pearson趋势图
        if (epoch + 1) % 10 == 0 and self.vis_dir:
            self.update_pearson_trend()
    
    def update_pearson_trend(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.val_pearson_history, 'b-', label='Validation Pearson')
        if self.train_pearson_history:
            plt.plot(self.epochs, self.train_pearson_history, 'r-', label='Training Pearson')
        plt.axhline(y=0.9, color='g', linestyle='--', label='Target: 0.9')
        plt.xlabel('Epoch')
        plt.ylabel('Pearson Correlation')
        plt.title('Pearson Correlation Trend During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.vis_dir, 'pearson_trend.png'))
        plt.close()

# 加载数据
def load_data():
    try:
        # 尝试加载训练数据和测试数据
        x_train = np.load('data/raw/x_48_train(1).npy')
        y_train = np.load('data/raw/y_48_train(1).npy')
        x_test = np.load('data/raw/x_48_test(1).npy')
        y_test = np.load('data/raw/y_48_test(1).npy')
        
        print("训练集形状:", x_train.shape, y_train.shape)
        print("测试集形状:", x_test.shape, y_test.shape)
        
        return x_train, y_train, x_test, y_test
    except FileNotFoundError:
        # 如果在data/raw中找不到，尝试上级目录
        try:
            x_train = np.load('../x_48_train(1).npy')
            y_train = np.load('../y_48_train(1).npy')
            x_test = np.load('../x_48_test(1).npy')
            y_test = np.load('../y_48_test(1).npy')
            
            # 保存到data/raw目录
            np.save('data/raw/x_48_train(1).npy', x_train)
            np.save('data/raw/y_48_train(1).npy', y_train)
            np.save('data/raw/x_48_test(1).npy', x_test)
            np.save('data/raw/y_48_test(1).npy', y_test)
            
            print("训练集形状:", x_train.shape, y_train.shape)
            print("测试集形状:", x_test.shape, y_test.shape)
            
            return x_train, y_train, x_test, y_test
        except FileNotFoundError:
            print("错误：无法找到数据文件。请确保数据集已下载并放在正确的位置。")
            sys.exit(1)

# 定义残差块
def residual_block(x, filters, kernel_size=3, strides=1, activation='relu', reg=1e-4):
    # 第一个卷积层
    residual = Conv2D(filters, kernel_size, strides=strides, padding='same',
                      kernel_regularizer=l2(reg))(x)
    residual = BatchNormalization()(residual)
    residual = Activation(activation)(residual)
    
    # 第二个卷积层
    residual = Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=l2(reg))(residual)
    residual = BatchNormalization()(residual)
    
    # 如果输入和输出的维度不同，则使用1x1卷积调整维度
    if strides != 1 or x.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=strides, padding='same',
                         kernel_regularizer=l2(reg))(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x
    
    # 添加shortcut连接
    x = add([residual, shortcut])
    x = Activation(activation)(x)
    
    return x

# 构建增强版ResNet 2D CNN模型
def build_enhanced_resnet_2d_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # 初始卷积层
    x = Conv2D(32, 3, padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 残差块阶段1
    x = residual_block(x, 32, reg=1e-4)
    x = residual_block(x, 32, reg=1e-4)
    
    # 残差块阶段2
    x = residual_block(x, 64, strides=2, reg=1e-4)
    x = residual_block(x, 64, reg=1e-4)
    
    # 残差块阶段3
    x = residual_block(x, 128, strides=2, reg=1e-4)
    x = residual_block(x, 128, reg=1e-4)
    
    # 全局平均池化
    x = GlobalAveragePooling2D()(x)
    
    # 全连接层
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    
    # 输出层
    outputs = Dense(1, activation='linear', kernel_regularizer=l2(1e-4))(x)
    
    model = Model(inputs, outputs)
    
    # 使用Adam优化器，学习率设置为0.0005
    optimizer = Adam(learning_rate=0.0005)
    
    # 编译模型，使用MSE作为损失函数
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

# 数据预处理
def preprocess_data(x_train, y_train, x_test, y_test, vis_dir):
    # 标准化特征
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # 重塑数据以适应StandardScaler
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
    
    # 标准化
    x_train_scaled = scaler_x.fit_transform(x_train_reshaped)
    x_test_scaled = scaler_x.transform(x_test_reshaped)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # 可视化目标变量分布（原始和标准化后）
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_train, bins=30, alpha=0.7)
    plt.title('Original Target Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(y_train_scaled, bins=30, alpha=0.7)
    plt.title('Standardized Target Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'target_distribution.png'))
    plt.close()
    
    # 恢复原始形状
    x_train_scaled = x_train_scaled.reshape(x_train.shape)
    x_test_scaled = x_test_scaled.reshape(x_test.shape)
    
    # 分割训练集和验证集
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )
    
    # 将数据重塑为2D CNN的输入格式 (样本数, 高度, 宽度, 通道数)
    x_train_reshaped = x_train_split.reshape(x_train_split.shape[0], x_train_split.shape[1], x_train_split.shape[2], 1)
    x_val_reshaped = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], x_test_scaled.shape[1], x_test_scaled.shape[2], 1)
    
    # 可视化数据样本
    visualize_sample_data(x_train, x_train_scaled, vis_dir)
    
    print("Preprocessed data shapes:")
    print("Training set:", x_train_reshaped.shape, y_train_split.shape)
    print("Validation set:", x_val_reshaped.shape, y_val.shape)
    print("Test set:", x_test_reshaped.shape, y_test_scaled.shape)
    
    return x_train_reshaped, y_train_split, x_val_reshaped, y_val, x_test_reshaped, y_test_scaled, scaler_y

# 可视化数据样本
def visualize_sample_data(x_raw, x_scaled, vis_dir):
    # 选择前3个样本的第1个特征进行可视化
    plt.figure(figsize=(15, 10))
    
    # 原始数据
    for i in range(3):
        plt.subplot(3, 2, i*2+1)
        plt.plot(x_raw[i, :, 0])
        plt.title(f'Sample {i+1} Original Time Series (Feature 1)')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    
    # 标准化后的数据
    for i in range(3):
        plt.subplot(3, 2, i*2+2)
        plt.plot(x_scaled[i, :, 0])
        plt.title(f'Sample {i+1} Standardized Time Series (Feature 1)')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'sample_data_visualization.png'))
    plt.close()
    
    # 可视化热图
    plt.figure(figsize=(10, 6))
    sample_idx = 0  # 第一个样本
    plt.imshow(x_scaled[sample_idx], aspect='auto', cmap='viridis')
    plt.colorbar(label='Standardized Value')
    plt.title(f'Sample {sample_idx+1} Feature Heatmap')
    plt.xlabel('Feature')
    plt.ylabel('Time Step')
    plt.savefig(os.path.join(vis_dir, 'sample_heatmap.png'))
    plt.close()

# 训练模型
def train_model(x_train, y_train, x_val, y_val, input_shape, vis_dir):
    print("\nStarting training of enhanced ResNet 2D CNN model...")
    
    # 构建模型
    model = build_enhanced_resnet_2d_model(input_shape)
    
    # 保存模型结构图
    tf.keras.utils.plot_model(
        model, 
        to_file=os.path.join(vis_dir, 'model_architecture.png'),
        show_shapes=True,
        show_layer_names=True
    )
    
    model.summary()
    
    # 将模型摘要保存到文件
    with open(os.path.join(vis_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # 定义回调函数
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(vis_dir, 'best_model.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=50,  # 增加早停耐心值
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=0.2,
        patience=10,  # 增加学习率调整耐心值
        min_lr=1e-6,
        verbose=1
    )
    
    pearson_callback = PearsonCallback(
        validation_data=(x_val, y_val),
        training_data=(x_train[:100], y_train[:100]),  # 使用部分训练数据计算Pearson
        vis_dir=vis_dir
    )
    
    # 训练模型
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        epochs=200,  # 增加训练轮数
        batch_size=16,  # 减小批次大小
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, model_checkpoint, reduce_lr, pearson_callback],
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed, time taken: {training_time:.2f} seconds")
    
    # 保存训练历史
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(vis_dir, 'training_history.csv'), index=False)
    
    # 绘制训练历史
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # MAE曲线
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    # Pearson相关系数曲线
    plt.subplot(2, 2, 3)
    if 'train_pearson' in history.history:
        plt.plot(history.history['train_pearson'], label='Training Pearson')
    if 'val_pearson' in history.history:
        plt.plot(history.history['val_pearson'], label='Validation Pearson')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target: 0.9')
    plt.title('Pearson Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    plt.grid(True)
    
    # 学习率曲线（如果有记录）
    plt.subplot(2, 2, 4)
    if 'lr' in history.history:
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_history.png'))
    plt.close()
    
    return model

# 评估模型
def evaluate_model(model, x_test, y_test, scaler_y, vis_dir):
    print("\nEvaluating model on test set...")
    
    # 模型预测
    y_pred_scaled = model.predict(x_test)
    
    # 反标准化预测结果
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    # 计算评估指标
    mse = np.mean((y_test_original - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_original - y_pred))
    r2 = 1 - np.sum((y_test_original - y_pred) ** 2) / np.sum((y_test_original - np.mean(y_test_original)) ** 2)
    pearson = pearson_correlation(y_test_original, y_pred)
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R²: {r2:.6f}")
    print(f"Test Pearson Correlation: {pearson:.6f}")
    
    # 保存结果
    results = {
        'model': ['ResNet_2D_CNN'],
        'mse': [mse],
        'rmse': [rmse],
        'mae': [mae],
        'r2': [r2],
        'pearson': [pearson]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(vis_dir, 'test_results.csv'), index=False)
    
    # 绘制预测与真实值对比图
    plt.figure(figsize=(12, 10))
    
    # 散点图
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_original, y_pred, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs True Values (Pearson = {pearson:.4f})')
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
    plt.savefig(os.path.join(vis_dir, 'test_predictions.png'))
    plt.close()
    
    # 保存样本预测结果
    sample_results = pd.DataFrame({
        'True Values': y_test_original.flatten(),
        'Predicted Values': y_pred.flatten(),
        'Residuals': residuals.flatten()
    })
    sample_results.to_csv(os.path.join(vis_dir, 'sample_predictions.csv'), index=False)
    
    return mse, r2, pearson

def main():
    print("===== ResNet 2D CNN Model Training and Evaluation =====")
    
    # 加载数据
    x_train, y_train, x_test, y_test = load_data()
    
    # 预处理数据
    x_train_reshaped, y_train_split, x_val_reshaped, y_val, x_test_reshaped, y_test_scaled, scaler_y = preprocess_data(
        x_train, y_train, x_test, y_test, vis_dir
    )
    
    # 训练模型
    input_shape = (x_train_reshaped.shape[1], x_train_reshaped.shape[2], x_train_reshaped.shape[3])
    model = train_model(x_train_reshaped, y_train_split, x_val_reshaped, y_val, input_shape, vis_dir)
    
    # 评估模型
    mse, r2, pearson = evaluate_model(model, x_test_reshaped, y_test_scaled, scaler_y, vis_dir)
    
    # 检查是否达到目标
    if pearson >= 0.9:
        print("\nCongratulations! Model achieved target Pearson correlation (>= 0.9).")
    else:
        print(f"\nModel's Pearson correlation is {pearson:.4f}, did not reach target value 0.9.")
    
    # 保存最终训练信息
    with open(os.path.join(vis_dir, 'training_info.txt'), 'w') as f:
        f.write("ResNet 2D CNN Model Training Information\n")
        f.write("=====================================\n\n")
        f.write(f"Training Time: {timestamp}\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Training Samples: {x_train_reshaped.shape[0]}\n")
        f.write(f"Validation Samples: {x_val_reshaped.shape[0]}\n")
        f.write(f"Test Samples: {x_test_reshaped.shape[0]}\n\n")
        f.write("Test Set Evaluation Results:\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"R²: {r2:.6f}\n")
        f.write(f"Pearson Correlation: {pearson:.6f}\n")
    
    return mse, r2, pearson

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        mse, r2, pearson = main()
        
        end_time = time.time()
        total_runtime = end_time - start_time
        
        # 打印总运行时间
        print(f"\nTotal Runtime: {total_runtime:.2f} seconds")
        
        # 输出最终结果
        print("\n===== Final Results =====")
        print(f"MSE: {mse:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Pearson Correlation: {pearson:.6f}")
        
        # 保存运行时间
        with open(os.path.join(vis_dir, 'runtime.txt'), 'w') as f:
            f.write(f"Total Runtime: {total_runtime:.2f} seconds")
    
    except Exception as e:
        # 记录错误信息
        print(f"Error during training: {str(e)}")
        with open(os.path.join(vis_dir, 'error_log.txt'), 'w') as f:
            import traceback
            f.write(f"Error Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error Message: {str(e)}\n\n")
            f.write(traceback.format_exc()) 
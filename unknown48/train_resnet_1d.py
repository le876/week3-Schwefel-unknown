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
from tensorflow.keras.layers import Input, Conv1D, Conv2D, BatchNormalization, Activation, add, Dense, Dropout, GlobalAveragePooling1D, GlobalAveragePooling2D, MaxPooling1D, MaxPooling2D, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K

# 设置matplotlib使用默认英文字体
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

# 定义残差块 (1D版本)
def residual_block_1d(x, filters, kernel_size=3, strides=1, activation='relu', reg=1e-4):
    # 第一个卷积层
    residual = Conv1D(filters, kernel_size, strides=strides, padding='same',
                    kernel_regularizer=l2(reg))(x)
    residual = BatchNormalization()(residual)
    residual = Activation(activation)(residual)
    
    # 第二个卷积层
    residual = Conv1D(filters, kernel_size, padding='same',
                    kernel_regularizer=l2(reg))(residual)
    residual = BatchNormalization()(residual)
    
    # 如果输入和输出的维度不同，则使用1x1卷积调整维度
    if strides != 1 or x.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=strides, padding='same',
                        kernel_regularizer=l2(reg))(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x
    
    # 添加shortcut连接
    x = add([residual, shortcut])
    x = Activation(activation)(x)
    
    return x

# 定义残差块 (2D版本)
def residual_block_2d(x, filters, kernel_size=3, strides=1, activation='relu', reg=1e-4):
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

# 构建1D ResNet CNN模型
def build_resnet_1d_model(input_shape):
    """构建简化版的ResNet 1D模型"""
    inputs = Input(shape=input_shape)
    
    # 第一个卷积块
    x = Conv1D(32, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 第一个残差块
    x = residual_block_1d(x, 32, kernel_size=3, reg=1e-4)
    
    # 第二个卷积块
    x = Conv1D(64, kernel_size=3, padding='same', strides=2, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 第二个残差块
    x = residual_block_1d(x, 64, kernel_size=3, reg=1e-4)
    
    # 第三个卷积块
    x = Conv1D(128, kernel_size=3, padding='same', strides=2, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 全局平均池化
    x = GlobalAveragePooling1D()(x)
    
    # 全连接层
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1)(x)
    
    return Model(inputs=inputs, outputs=outputs)

# 构建2D ResNet CNN模型
def build_resnet_2d_model(input_shape):
    """构建简化版的ResNet 2D模型"""
    inputs = Input(shape=input_shape)
    
    # 第一个卷积块
    x = Conv2D(32, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 第一个残差块
    x = residual_block_2d(x, 32, kernel_size=3, reg=1e-4)
    
    # 第二个卷积块
    x = Conv2D(64, kernel_size=3, padding='same', strides=2, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 第二个残差块
    x = residual_block_2d(x, 64, kernel_size=3, reg=1e-4)
    
    # 第三个卷积块
    x = Conv2D(128, kernel_size=3, padding='same', strides=2, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 全局平均池化
    x = GlobalAveragePooling2D()(x)
    
    # 全连接层
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1)(x)
    
    return Model(inputs=inputs, outputs=outputs)

# 定义自定义的Pearson损失函数
def pearson_loss(y_true, y_pred):
    """改进的Pearson损失函数"""
    # 标准化
    y_true = (y_true - K.mean(y_true)) / (K.std(y_true) + K.epsilon())
    y_pred = (y_pred - K.mean(y_pred)) / (K.std(y_pred) + K.epsilon())
    
    # 计算相关系数
    pearson = K.sum(y_true * y_pred) / (K.sqrt(K.sum(K.square(y_true))) * K.sqrt(K.sum(K.square(y_pred))) + K.epsilon())
    
    # 添加MSE损失作为正则化项
    mse = K.mean(K.square(y_true - y_pred))
    
    # 组合损失
    return -pearson + 0.1 * mse

# 数据预处理 - 根据模型类型处理数据
def preprocess_data(x_train, y_train, x_test, y_test, model_type, vis_dir):
    """根据模型类型进行数据预处理"""
    # 标准化特征
    scaler_x = StandardScaler()
    x_train_reshaped = x_train.reshape(-1, x_train.shape[-1])
    x_test_reshaped = x_test.reshape(-1, x_test.shape[-1])
    x_train_scaled = scaler_x.fit_transform(x_train_reshaped).reshape(x_train.shape)
    x_test_scaled = scaler_x.transform(x_test_reshaped).reshape(x_test.shape)
    
    # 标准化目标变量
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(x_train_scaled))
    x_val = x_train_scaled[train_size:]
    y_val = y_train_scaled[train_size:]
    x_train = x_train_scaled[:train_size]
    y_train = y_train_scaled[:train_size]
    
    # 根据模型类型重塑数据
    if model_type == '1d':
        # 1D模型：将时间步长作为通道，特征数作为空间维度
        x_train_reshaped = x_train
        x_val_reshaped = x_val
        x_test_reshaped = x_test_scaled
    else:  # 2d
        # 2D模型：将特征和时间步长都作为空间维度
        x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_val_reshaped = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
        x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], x_test_scaled.shape[1], x_test_scaled.shape[2], 1)
    
    # 可视化目标变量分布
    plt.figure(figsize=(10, 6))
    plt.hist(y_train, bins=50, alpha=0.5, label='Training')
    plt.hist(y_val, bins=50, alpha=0.5, label='Validation')
    plt.title('Target Variable Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(vis_dir, 'target_distribution.png'))
    plt.close()
    
    return x_train_reshaped, y_train, x_val_reshaped, y_val, x_test_reshaped, y_test_scaled, scaler_y

# 训练模型
def train_model(x_train, y_train, x_val, y_val, input_shape, model_type, vis_dir):
    """根据模型类型训练模型"""
    # 构建模型
    if model_type == '1d':
        model = build_resnet_1d_model(input_shape)
    else:  # 2d
        model = build_resnet_2d_model(input_shape)
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=pearson_loss,
        metrics=['mse', 'mae']
    )
    
    # 早停策略
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # 学习率调整
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # 模型检查点
    checkpoint = ModelCheckpoint(
        os.path.join(vis_dir, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Pearson相关系数回调
    pearson_callback = PearsonCallback(
        validation_data=(x_val, y_val),
        training_data=(x_train, y_train),
        vis_dir=vis_dir
    )
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint, pearson_callback],
        verbose=1
    )
    
    return model, history

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
        'model': ['ResNet_1D_CNN'],
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
    # 设置模型类型
    model_type = '1d'  # 或 '2d'
    print(f"===== ResNet {model_type.upper()} CNN Model Training and Evaluation =====")
    
    # 加载数据
    x_train, y_train, x_test, y_test = load_data()
    
    # 预处理数据
    x_train_reshaped, y_train_split, x_val_reshaped, y_val, x_test_reshaped, y_test_scaled, scaler_y = preprocess_data(
        x_train, y_train, x_test, y_test, model_type, vis_dir
    )
    
    # 训练模型
    input_shape = x_train_reshaped.shape[1:]  # 根据模型类型自动获取输入形状
    model, history = train_model(x_train_reshaped, y_train_split, x_val_reshaped, y_val, input_shape, model_type, vis_dir)
    
    # 评估模型
    mse, r2, pearson = evaluate_model(model, x_test_reshaped, y_test_scaled, scaler_y, vis_dir)
    
    # 检查是否达到目标
    if pearson >= 0.9:
        print("\nCongratulations! Model achieved target Pearson correlation (>= 0.9).")
    else:
        print(f"\nModel's Pearson correlation is {pearson:.4f}, did not reach target value 0.9.")
    
    # 保存最终训练信息
    with open(os.path.join(vis_dir, 'training_info.txt'), 'w') as f:
        f.write(f"ResNet {model_type.upper()} CNN Model Training Information\n")
        f.write("=====================================\n\n")
        f.write(f"Training Time: {timestamp}\n")
        f.write(f"Model Type: {model_type.upper()}\n")
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
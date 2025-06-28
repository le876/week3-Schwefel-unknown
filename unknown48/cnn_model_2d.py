import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Concatenate, GlobalAveragePooling2D, Activation, Add, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
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

# 构建基础2D CNN模型
def build_cnn_2d_model(input_shape):
    model = Sequential()
    
    # 第一层卷积
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 第二层卷积
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 第三层卷积
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 全局平均池化
    model.add(GlobalAveragePooling2D())
    
    # 全连接层
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

# 构建多尺度2D CNN模型
def build_multiscale_cnn_2d_model(input_shape):
    input_layer = Input(shape=input_shape)
    
    # 小尺度卷积分支
    conv_s1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    bn_s1 = BatchNormalization()(conv_s1)
    conv_s2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(bn_s1)
    bn_s2 = BatchNormalization()(conv_s2)
    pool_s = MaxPooling2D(pool_size=(2, 2))(bn_s2)
    
    # 中尺度卷积分支
    conv_m1 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    bn_m1 = BatchNormalization()(conv_m1)
    conv_m2 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu')(bn_m1)
    bn_m2 = BatchNormalization()(conv_m2)
    pool_m = MaxPooling2D(pool_size=(2, 2))(bn_m2)
    
    # 大尺度卷积分支
    conv_l1 = Conv2D(64, kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    bn_l1 = BatchNormalization()(conv_l1)
    conv_l2 = Conv2D(64, kernel_size=(7, 7), padding='same', activation='relu')(bn_l1)
    bn_l2 = BatchNormalization()(conv_l2)
    pool_l = MaxPooling2D(pool_size=(2, 2))(bn_l2)
    
    # 合并不同尺度的特征
    concat = Concatenate()([pool_s, pool_m, pool_l])
    
    # 共享卷积层
    conv_shared = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(concat)
    bn_shared = BatchNormalization()(conv_shared)
    pool_shared = MaxPooling2D(pool_size=(2, 2))(bn_shared)
    
    # 深层特征提取
    conv_deep = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(pool_shared)
    bn_deep = BatchNormalization()(conv_deep)
    gap = GlobalAveragePooling2D()(bn_deep)
    
    # 全连接层
    dense1 = Dense(128, activation='relu')(gap)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(64, activation='relu')(drop1)
    drop2 = Dropout(0.3)(dense2)
    output = Dense(1, activation='linear')(drop2)
    
    # 构建模型
    model = Model(inputs=input_layer, outputs=output)
    
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

# 构建ResNet风格2D CNN模型
def build_resnet_cnn_2d_model(input_shape):
    def residual_block(x, filters, kernel_size=(3, 3)):
        # 残差连接
        res = Conv2D(filters, kernel_size, padding='same')(x)
        res = BatchNormalization()(res)
        res = Activation('relu')(res)
        res = Conv2D(filters, kernel_size, padding='same')(res)
        res = BatchNormalization()(res)
        
        # 如果输入和输出维度不同，则使用1x1卷积进行投影
        if x.shape[-1] != filters:
            x = Conv2D(filters, (1, 1), padding='same')(x)
        
        # 添加跳跃连接
        out = Add()([res, x])
        out = Activation('relu')(out)
        return out
    
    input_layer = Input(shape=input_shape)
    
    # 初始卷积层
    x = Conv2D(64, (7, 7), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # 残差块
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # 全局平均池化
    x = GlobalAveragePooling2D()(x)
    
    # 全连接层
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)
    
    # 构建模型
    model = Model(inputs=input_layer, outputs=output)
    
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

# 构建强化的2D CNN模型 (特别针对时序和特征关系)
def build_enhanced_cnn_2d_model(input_shape):
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
    
    # 全连接层
    dense1 = Dense(256, activation='relu')(gap)
    bn3 = BatchNormalization()(dense1)
    drop1 = Dropout(0.5)(bn3)
    
    dense2 = Dense(128, activation='relu')(drop1)
    bn4 = BatchNormalization()(dense2)
    drop2 = Dropout(0.3)(bn4)
    
    dense3 = Dense(64, activation='relu')(drop2)
    bn5 = BatchNormalization()(dense3)
    drop3 = Dropout(0.2)(bn5)
    
    output = Dense(1, activation='linear')(drop3)
    
    # 构建模型
    model = Model(inputs=input_layer, outputs=output)
    
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

# 计算Pearson相关系数
def pearson_correlation(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# 训练模型
def train_model(model, x_train, y_train, x_val, y_val, model_name, epochs=100, batch_size=32):
    print(f"训练{model_name}模型...")
    
    # 定义回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(f'models/{model_name}_best.h5', monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存训练历史
    pd.DataFrame(history.history).to_csv(f'results/{model_name}_history.csv', index=False)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/{model_name}_loss_curve.png')
    plt.close()
    
    return model, history

# 评估模型
def evaluate_model(model, x_test, y_test, model_name):
    print(f"评估{model_name}模型...")
    
    # 预测
    y_pred = model.predict(x_test).flatten()
    
    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson = pearson_correlation(y_test, y_pred)
    
    # 打印结果
    print(f"{model_name} - 测试集MSE: {mse:.6f}")
    print(f"{model_name} - 测试集R²: {r2:.6f}")
    print(f"{model_name} - 测试集Pearson相关系数: {pearson:.6f}")
    
    # 保存结果
    results = {
        'model': model_name,
        'mse': mse,
        'r2': r2,
        'pearson': pearson
    }
    pd.DataFrame([results]).to_csv(f'results/{model_name}_metrics.csv', index=False)
    
    # 绘制预测vs真实值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'{model_name} - Predictions vs Actual (Pearson: {pearson:.4f})')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(f'results/{model_name}_predictions.png')
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
    
    # 转换为2D格式（添加通道维度）
    x_train_2d = reshape_to_2d(x_train)
    x_val_2d = reshape_to_2d(x_val)
    x_test_2d = reshape_to_2d(x_test)
    
    print(f"训练集形状(划分后): {x_train_2d.shape}, {y_train.shape}")
    print(f"验证集形状: {x_val_2d.shape}, {y_val.shape}")
    print(f"测试集形状: {x_test_2d.shape}, {y_test.shape}")
    
    return x_train_2d, y_train, x_val_2d, y_val, x_test_2d, y_test

# 主函数
def main():
    # 加载数据
    x_train, y_train, x_test, y_test = load_data()
    
    # 数据预处理
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # 获取输入形状
    input_shape = x_train.shape[1:]  # (seq_length, features, 1)
    print(f"输入形状: {input_shape}")
    
    # 建立并训练基础2D CNN模型
    basic_cnn = build_cnn_2d_model(input_shape)
    basic_cnn.summary()
    basic_cnn, _ = train_model(basic_cnn, x_train, y_train, x_val, y_val, "basic_cnn_2d")
    basic_pearson = evaluate_model(basic_cnn, x_test, y_test, "basic_cnn_2d")
    
    # 建立并训练多尺度2D CNN模型
    multiscale_cnn = build_multiscale_cnn_2d_model(input_shape)
    multiscale_cnn.summary()
    multiscale_cnn, _ = train_model(multiscale_cnn, x_train, y_train, x_val, y_val, "multiscale_cnn_2d")
    multiscale_pearson = evaluate_model(multiscale_cnn, x_test, y_test, "multiscale_cnn_2d")
    
    # 建立并训练ResNet风格2D CNN模型
    resnet_cnn = build_resnet_cnn_2d_model(input_shape)
    resnet_cnn.summary()
    resnet_cnn, _ = train_model(resnet_cnn, x_train, y_train, x_val, y_val, "resnet_cnn_2d")
    resnet_pearson = evaluate_model(resnet_cnn, x_test, y_test, "resnet_cnn_2d")
    
    # 建立并训练强化的2D CNN模型
    enhanced_cnn = build_enhanced_cnn_2d_model(input_shape)
    enhanced_cnn.summary()
    enhanced_cnn, _ = train_model(enhanced_cnn, x_train, y_train, x_val, y_val, "enhanced_cnn_2d")
    enhanced_pearson = evaluate_model(enhanced_cnn, x_test, y_test, "enhanced_cnn_2d")
    
    # 比较模型性能
    print("\n模型性能比较:")
    print(f"基础2D CNN - Pearson相关系数: {basic_pearson:.6f}")
    print(f"多尺度2D CNN - Pearson相关系数: {multiscale_pearson:.6f}")
    print(f"ResNet风格2D CNN - Pearson相关系数: {resnet_pearson:.6f}")
    print(f"强化2D CNN - Pearson相关系数: {enhanced_pearson:.6f}")
    
    # 确定最佳模型
    models = {
        "basic_cnn_2d": basic_pearson,
        "multiscale_cnn_2d": multiscale_pearson,
        "resnet_cnn_2d": resnet_pearson,
        "enhanced_cnn_2d": enhanced_pearson
    }
    best_model_name = max(models, key=models.get)
    best_pearson = models[best_model_name]
    
    print(f"\n最佳模型: {best_model_name}, Pearson相关系数: {best_pearson:.6f}")
    
    # 保存最终结果
    final_results = [
        {'model': 'basic_cnn_2d', 'pearson': basic_pearson},
        {'model': 'multiscale_cnn_2d', 'pearson': multiscale_pearson},
        {'model': 'resnet_cnn_2d', 'pearson': resnet_pearson},
        {'model': 'enhanced_cnn_2d', 'pearson': enhanced_pearson}
    ]
    pd.DataFrame(final_results).to_csv('results/model_comparison_2d.csv', index=False)
    
    # 检查是否达到目标
    if best_pearson >= 0.9:
        print("\n成功达到Pearson相关系数≥0.9的目标!")
    else:
        print(f"\n未达到Pearson相关系数≥0.9的目标。当前最佳: {best_pearson:.6f}")
        print("建议尝试以下改进方法:")
        print("1. 进一步调整模型架构，增加层数或调整卷积核大小")
        print("2. 结合统计特征与2D CNN特征")
        print("3. 尝试数据增强技术增加训练样本数量")
        print("4. 调整超参数，如学习率、批量大小等")

if __name__ == "__main__":
    main() 
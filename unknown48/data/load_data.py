import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_raw_data():
    """
    加载原始数据
    """
    # 获取当前文件所在目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(current_dir, 'raw')
    
    # 加载训练集和测试集
    x_train = np.load(os.path.join(raw_dir, 'x_48_train(1).npy'))
    y_train = np.load(os.path.join(raw_dir, 'y_48_train(1).npy'))
    x_test = np.load(os.path.join(raw_dir, 'x_48_test(1).npy'))
    y_test = np.load(os.path.join(raw_dir, 'y_48_test(1).npy'))
    
    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, y_train, x_test, y_test, add_squared_features=False, validation_split=0.2, random_state=42):
    """
    预处理数据：标准化特征，可选添加平方特征，划分验证集
    
    参数:
    - add_squared_features: 是否添加特征的平方作为新特征
    - validation_split: 验证集比例
    - random_state: 随机种子
    
    返回:
    - x_train_processed, y_train: 处理后的训练集
    - x_val_processed, y_val: 处理后的验证集
    - x_test_processed, y_test: 处理后的测试集
    - scaler: 标准化器，用于后续转换
    """
    # 划分训练集和验证集
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train, y_train, test_size=validation_split, random_state=random_state
    )
    
    # 标准化特征
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_split)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    
    # 如果需要添加平方特征
    if add_squared_features:
        x_train_squared = np.square(x_train_scaled)
        x_val_squared = np.square(x_val_scaled)
        x_test_squared = np.square(x_test_scaled)
        
        x_train_processed = np.hstack((x_train_scaled, x_train_squared))
        x_val_processed = np.hstack((x_val_scaled, x_val_squared))
        x_test_processed = np.hstack((x_test_scaled, x_test_squared))
    else:
        x_train_processed = x_train_scaled
        x_val_processed = x_val_scaled
        x_test_processed = x_test_scaled
    
    # 保存处理后的数据
    processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(os.path.join(processed_dir, 'x_train_processed.npy'), x_train_processed)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train_split)
    np.save(os.path.join(processed_dir, 'x_val_processed.npy'), x_val_processed)
    np.save(os.path.join(processed_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(processed_dir, 'x_test_processed.npy'), x_test_processed)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    
    return x_train_processed, y_train_split, x_val_processed, y_val, x_test_processed, y_test, scaler

def get_data(add_squared_features=False, validation_split=0.2, random_state=42):
    """
    获取处理好的数据，如果处理后的数据存在则直接加载，否则重新处理
    """
    processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
    
    # 检查处理后的数据是否存在
    if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) >= 6:
        x_train = np.load(os.path.join(processed_dir, 'x_train_processed.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
        x_val = np.load(os.path.join(processed_dir, 'x_val_processed.npy'))
        y_val = np.load(os.path.join(processed_dir, 'y_val.npy'))
        x_test = np.load(os.path.join(processed_dir, 'x_test_processed.npy'))
        y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
        
        # 重新创建scaler（仅用于转换新数据）
        x_train_raw, _, x_test_raw, _ = load_raw_data()
        scaler = StandardScaler()
        scaler.fit(x_train_raw)
        
        return x_train, y_train, x_val, y_val, x_test, y_test, scaler
    else:
        # 加载原始数据并处理
        x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_raw_data()
        return preprocess_data(
            x_train_raw, y_train_raw, x_test_raw, y_test_raw, 
            add_squared_features, validation_split, random_state
        )

if __name__ == "__main__":
    # 测试数据加载和预处理
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = get_data(add_squared_features=True)
    print(f"训练集形状: {x_train.shape}, {y_train.shape}")
    print(f"验证集形状: {x_val.shape}, {y_val.shape}")
    print(f"测试集形状: {x_test.shape}, {y_test.shape}") 
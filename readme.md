# Week3 ����ѧϰ��Ŀ����

����Ŀ�������������Ļ���ѧϰ���񣬷ֱ����20D Schwefel���ݼ��Լ�unknow48���ݼ��Ļع����⣬�����˴Ӵ�ͳ����ѧϰ�����ѧϰ�Ķ��ֽ�ģ������Ŀ��Ϊ����������pearson ratio��

## ��Ŀ����

### ����1��Schwefel����������ƽ�
- **Ŀ��**��ʹ��������ƽ�20άSchwefel����
- **�����ص�**��1000��20ά������ÿ��ά�ȶ�����ǿ
- **ģ�ͷ���**�����ں���ά�ȷֽ��������ܹ�
- **���ս��**��Pearson ratio= 0.892

### ����2��Unknown48ʱ�����ݻع�Ԥ��
- **Ŀ��**������άʱ�����ݽ��лع�Ԥ��
- **�����ص�**�� (1000��ѵ��������, 48��ʱ�䲽, 9������) ��ʱ��ṹ��������Ŀ�����������Խ���
- **ģ�ͷ���**��GBDT
- **���ս��**��Pearson ratio= 0.947

## ��Ŀ�ṹ

```
week3/
������ schwefel/                       # Schwefel�����ƽ�����
��   ������ data/                      # ����Ŀ¼
��   ������ models/                    # ģ�ͱ���
��   ������ results/                   # ������
��   ������ visualizations/            # ���ӻ����
��   ������ decomposed_nn.py          # �ֽ�������
��   ������ auto_decomposed_nn.py     # �Զ��ֽ�������
��   ������ train_rf.py               # ���ɭ��ѵ��
��   ������ train_gbdt.py             # �ݶ�������ѵ��
��   ������ data_analysis.py          # ���ݷ���
��   ������ model_explanation.md      # ģ����ϸ˵��
��   ������ requirements.txt          # �����б�
��   ������ README.md                 # ������˵��
������ unknown48/                      # Unknown48ʱ��ع�����
��   ������ data/                      # ����Ŀ¼
��   ������ models/                    # ģ�ͱ���
��   ������ results/                   # ������
��   ������ visualizations/            # ���ӻ����
��   ������ cnn_model.py              # һάCNNģ��
��   ������ cnn_model_2d.py           # ��άCNNģ��
��   ������ hybrid_model.py           # CNN+LSTM���ģ��
��   ������ train_resnet_1d.py        # һάResNet
��   ������ train_resnet_2d.py        # ��άResNet
��   ������ train_gbdt.py             # �ݶ�������
��   ������ train.py                  # ��ͳMLģ��
��   ������ data_analysis.py          # ���ݷ���
��   ������ ��������.md                # ���ݷ�������
��   ������ requirements.txt          # �����б�
��   ������ README.md                 # ������˵��
������ data_analysis_results/          # �������ݷ������
������ visualizations/                 # ȫ�ֿ��ӻ����
������ best_model.pth                 # ���ģ��Ȩ��
������ best_pearson_model.pth         # ���Pearsonģ��
������ best_val_model.pth             # �����֤ģ��
������ readme.md                      # ����Ŀ˵�������ļ���
```

## ����ջ

### ���Ŀ��
- **Python 3.8+**����Ҫ�������
- **PyTorch**�����ѧϰ���
- **Scikit-learn**����ͳ����ѧϰ
- **NumPy/Pandas**�����ݴ���
- **Matplotlib/Seaborn**�����ݿ��ӻ�

### ����ѧϰ����
- **���ѧϰ**��CNN��LSTM��ResNet�����ģ��
- **��ͳML**�����ɭ�֡��ݶ���������֧��������
- **����ѧϰ**��ͶƱ�ع顢�ѵ��ع�
- **��������**��ʱ��������ͳ����������������

## ���ݷ�����ѵ�����

### Schwefel���ݼ�

#### ���ݷ������
<div align="center">
<img src="schwefel/analysis_results/20250320_224945/test_correlation_matrix.png" alt="Schwefel���ݼ�ά������Է���" width="600"/>
<p><em>Schwefel���ݼ�ά������Է���</em></p>
</div>

��ϸ�����������`schwefel/analysis_results/20250320_224945/`

#### �ع�����
<div align="center">
<img src="schwefel/visualizations/auto_decomposed_20250409_213546/prediction_scatter.png" alt="Schwefel����Ԥ����ɢ��ͼ" width="600"/>
<p><em>Schwefel����Ԥ������Ԥ��ֵ vs ��ʵֵ (Pearson = 0.892)</em></p>
</div>

��ϸѵ���������`schwefel/visualizations/auto_decomposed_20250409_213546/`

### Unknown48���ݼ�

#### ���ݷ������
<div align="center">
<img src="unknown48/data_analysis_results/time_series_all_features_sample_0.png" alt="Unknown48ʱ�������������ӻ�" width="700"/>
<p><em>Unknown48���ݼ�ʱ���������ӻ�������0��9��������48��ʱ�䲽�ı仯��</em></p>
</div>

<div align="center">
<img src="unknown48/data_analysis_results/correlation_matrix.png" alt="Unknown48��������Ծ���" width="600"/>
<p><em>Unknown48���ݼ���������Ծ���</em></p>
</div>

��ϸ�����������`unknown48/data_analysis_results/`

#### �ع�����
<div align="center">
<img src="unknown48/visualizations/20250321_203139/gbdt_best_model_predictions.png" alt="Unknown48 GBDTģ��Ԥ����" width="600"/>
<p><em>Unknown48���GBDTģ��Ԥ������Ԥ��ֵ vs ��ʵֵ (Pearson = 0.947)</em></p>
</div>

��ϸѵ���������`unknown48/visualizations/20250321_203139/`

## ���ݷ������ֽ��

### Schwefel�����ֽ�������ܹ�

<div align="center">
<img src="schwefel/visualizations/auto_decomposed_20250409_213546/dimension_contributions.png" alt="Schwefel������ά�ȹ��׷���" width="700"/>
<p><em>Schwefel������ά�ȹ��׷�����չʾ20��ά�ȶ�����Ԥ��Ĺ��׷ֲ�</em></p>
</div>

<div align="center">
<img src="schwefel/visualizations/auto_decomposed_20250409_213546/training_history.png" alt="Schwefel������ѵ����ʷ" width="600"/>
<p><em>�Զ��ֽ�������ѵ�����̣���ʧ������Pearson���ϵ���仯</em></p>
</div>

### Unknown48ʱ�����ݷ���

<div align="center">
<img src="unknown48/data_analysis_results/feature_target_correlation.png" alt="Unknown48������Ŀ����������" width="600"/>
<p><em>Unknown48���ݼ�������Ŀ�����������Է���</em></p>
</div>


## ��������

### 1. ����������װ

```bash
# ��װ������ѧ�����
pip install numpy>=1.20.0 scipy>=1.7.0 pandas>=1.3.0
pip install scikit-learn>=1.0.0 matplotlib>=3.4.0 joblib>=1.0.0
```

### 2. ���ѧϰ���

```bash
# PyTorch (�Ƽ�)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ��CPU�汾
pip install torch torchvision
```

### 3. ���⹤�߰�

```bash
# �ݶ�������
pip install xgboost lightgbm catboost

# ���ӻ���ǿ
pip install seaborn plotly

# �������Ż�
pip install optuna hyperopt
```

## �����ļ�˵��

### ����λ��
- **Schwefel����**��λ�� `schwefel/data/raw/` Ŀ¼
  - `Schwefel_x_train.npy`��`Schwefel_y_train.npy`
  - `Schwefel_x_test.npy`��`Schwefel_y_test.npy`

- **Unknown48����**��λ�� `unknown48/data/raw/` Ŀ¼
  - `x_48_train(1).npy`��`y_48_train(1).npy`
  - `x_48_test(1).npy`��`y_48_test(1).npy`

## ���ٿ�ʼ

### Schwefel�����ƽ�����

```bash
# ����schwefelĿ¼
cd schwefel/

# ���ݷ���
python data_analysis.py

# ѵ���Զ��ֽ������磨�Ƽ���
python auto_decomposed_nn.py

# ѵ����ͳ�����Ա�
python train_rf.py
python train_gbdt.py
```

### Unknown48ʱ��ع�����

```bash
# ����unknown48Ŀ¼
cd unknown48/

# ���ݷ���
python data_analysis.py

# ѵ�����ѧϰģ��
python hybrid_model.py      # CNN+LSTM���ģ�ͣ��Ƽ���
python cnn_model.py         # һάCNN
python train_resnet_1d.py   # һάResNet

# ѵ����ͳ����
python train_gbdt.py        # �ݶ�������
python train.py             # ���ִ�ͳģ��
```

## ʵ�鷽��

### 1. Schwefel�����ֽⷽ��

**���ⱳ��**��Schwefel�������и�ά�ȶ�������ѧ����
**�������**����Ʒֽ������磬���ò����������
**��������**��
- �����CNN���������ģ�Ͳ�����
- ���ָ�Ԥ�⾫�ȣ�Pearson > 0.89��

### 2. ʱ�����ݶ�ά�Ƚ�ģ

**���ⱳ��**��Unknown48���ݾ��и��ӵ�ʱ��������ϵ
**ʵ�鷽��**�����ֻ���ѧϰ�ܹ���ϵͳ�ԶԱȣ�������
- һά/��άCNN
- CNN+LSTM���ģ��
- GBDT
**���ս��**��
- **��ѷ���**��GBDT
- **���Լ�Pearson���ϵ��**��0.947
- **ģ���ص�**��pearson ratio�ϸߣ����Ǵ�pred vs trueͼ�пɿ���ģ���������������


## ���ӻ�����

��Ŀ�����ḻ�Ŀ��ӻ�������

### ���ݷ������ӻ�
- �����ֲ�ͼ������ͼ
- ���������ͼ
- ���ɷַ���ͼ
- ʱ��ģʽ���ӻ�

### ģ�����ܿ��ӻ�
- ѵ��/��֤��ʧ����
- Pearson���ϵ���仯����
- Ԥ��ֵvs��ʵֵɢ��ͼ
- �в����ͼ

### ģ�ͽ��Ϳ��ӻ�
- ������Ҫ������
- ά�ȹ��׷�����Schwefel��
- ע����Ȩ�ؿ��ӻ������ģ�ͣ�

## �ĵ�˵��

- **`schwefel/README.md`**��Schwefel������ϸ˵��
- **`schwefel/model_explanation.md`**���ֽ������缼��ϸ��
- **`unknown48/README.md`**��Unknown48������ϸ˵��
- **`unknown48/��������.md`**��ʱ��������ȷ�������

## ���л���Ҫ��

- **����ϵͳ**��Linux/Windows/macOS
- **Python�汾**��3.8+
- **�ڴ�Ҫ��**��8GB+ (�Ƽ�16GB)
- **GPUҪ��**����ѡ��CUDA����GPU�ɼ������ѧϰѵ��
- **�洢�ռ�**��2GB+ (�������ݡ�ģ�ͺͽ��)

## todo
���ʵ�����������ģ�͵����ܱ���
---

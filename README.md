# kaggle
<img width="1167" height="732" alt="image" src="https://github.com/user-attachments/assets/4360a098-1858-47ef-8774-d5bd71051138" />


# X-ray 肺炎影像分類系統

使用深度學習進行胸部 X 光影像的肺炎檢測。本專案採用卷積神經網路 (CNN) 對 X 光影像進行二元分類（正常 vs 肺炎）。

---

## 專案簡介

本專案實現了基於深度學習的肺炎檢測系統，能夠自動分析胸部 X 光影像並判斷是否患有肺炎。

### 主要功能

- 二元分類：NORMAL vs PNEUMONIA
- CNN 深度學習模型
- 數據增強 (Data Augmentation)
- 模型評估與視覺化
- 單張影像預測

### 技術特點

- 深度學習框架: TensorFlow/Keras
- 優化器: Adam Optimizer
- 數據增強: 旋轉、平移、縮放
- 評估指標: Accuracy, Precision, Recall, F1-Score

---

## 資料集

### 資料結構

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 資料統計

| 類別 | 訓練集 | 測試集 | 驗證集 |
|------|--------|--------|--------|
| NORMAL | 1,341 | 234 | 8 |
| PNEUMONIA | 3,875 | 390 | 8 |
| **總計** | **5,216** | **624** | **16** |

---

## 環境需求

### Python 版本
- Python 3.8+

### 必要套件

```
tensorflow>=2.10.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

---

## 安裝步驟

### 1. Clone 專案

```bash
git clone https://github.com/你的用戶名/chest-xray-pneumonia.git
cd chest-xray-pneumonia
```

### 2. 建立虛擬環境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. 安裝依賴套件

```bash
pip install -r requirements.txt
```

---

## 使用方法

### 訓練模型

```bash
python train_model.py
```

可調整參數:
- `--epochs`: 訓練輪數 (預設 30)
- `--batch_size`: 批次大小 (預設 64)
- `--learning_rate`: 學習率 (預設 0.001)

### 評估模型

```bash
python evaluate_model.py
```

輸出結果:
- 分類報告 (Precision, Recall, F1-Score)
- 混淆矩陣
- 準確率曲線
- 損失曲線

### 單張影像預測

```bash
python predict.py --image path/to/xray_image.jpg
```

---

## 模型架構

### CNN 架構摘要

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Conv2D (64 filters)         (None, 32, 32, 64)        1,792     
BatchNormalization          (None, 32, 32, 64)        256       
MaxPooling2D (2x2)          (None, 16, 16, 64)        0         
_________________________________________________________________
Conv2D (128 filters)        (None, 16, 16, 128)       73,856    
BatchNormalization          (None, 16, 16, 128)       512       
MaxPooling2D (2x2)          (None, 8, 8, 128)         0         
_________________________________________________________________
Conv2D (256 filters)        (None, 8, 8, 256)         295,168   
BatchNormalization          (None, 8, 8, 256)         1,024     
MaxPooling2D (2x2)          (None, 4, 4, 256)         0         
_________________________________________________________________
Flatten                     (None, 4096)              0         
_________________________________________________________________
Dense (500 neurons)         (None, 500)               2,048,500 
BatchNormalization          (None, 500)               2,000     
Dense (100 neurons)         (None, 100)               50,100    
BatchNormalization          (None, 100)               400       
Dense (2 neurons)           (None, 2)                 202       
=================================================================
Total params: 2,473,810
Trainable params: 2,471,714
Non-trainable params: 2,096
```

### 訓練配置

| 項目 | 設定 |
|------|------|
| 優化器 | Adam (lr=0.001) |
| 損失函數 | Categorical Crossentropy |
| 評估指標 | Accuracy |
| 訓練輪數 | 30 epochs |
| 批次大小 | 64 |

### 數據增強

- 隨機旋轉: ±25°
- 水平平移: ±3 pixels
- 垂直平移: ±3 pixels
- 隨機縮放: 0.3

---

## 訓練結果

### 模型表現

| 指標 | 測試集 |
|------|--------|
| Accuracy | 97.0% |
| Precision (平均) | 0.95 |
| Recall (平均) | 0.95 |
| F1-Score (平均) | 0.95 |

### 分類報告

```
              precision    recall  f1-score   support

   PNEUMONIA       0.98      0.98      0.98       203
      NORMAL       0.93      0.91      0.92        58

    accuracy                           0.97       261
   macro avg       0.95      0.95      0.95       261
weighted avg       0.97      0.97      0.97       261
```

### 混淆矩陣

|  | 預測 PNEUMONIA | 預測 NORMAL |
|---|----------------|-------------|
| **實際 PNEUMONIA** | 199 | 4 |
| **實際 NORMAL** | 5 | 53 |

### 詳細分析

**PNEUMONIA (肺炎) 類別**
- Precision: 0.98 (98%)
- Recall: 0.98 (98%)
- F1-Score: 0.98
- Support: 203 張

**NORMAL (正常) 類別**
- Precision: 0.93 (93%)
- Recall: 0.91 (91%)
- F1-Score: 0.92
- Support: 58 張

**模型準確率**: 97% (253/261 正確預測)

---

## 專案結構

```
chest-xray-pneumonia/
├── chest_xray/                 # 資料集目錄
│   ├── train/
│   ├── test/
│   └── val/
├── models/                     # 訓練好的模型
│   ├── my_model.h5
│   ├── my_model.keras
│   └── best_model.keras
├── docs/                       # 文檔與圖片
│   ├── training_curves.png
│   └── confusion_matrix.png
├── notebooks/                  # Jupyter Notebooks
│   └── X-ray-1.ipynb
├── train_model.py              # 訓練腳本
├── evaluate_model.py           # 評估腳本
├── predict.py                  # 預測腳本
├── requirements.txt            # 套件需求
└── README.md                   # 說明文件
```

---

## 使用範例

### Python 預測腳本

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 載入模型
model = load_model('models/my_model.h5')

# 讀取並預處理影像
img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (32, 32))
img = img.astype('float32') / 255.0
img = img.reshape(1, 32, 32, 3)

# 預測
prediction = model.predict(img)
class_idx = np.argmax(prediction[0])
classes = ['NORMAL', 'PNEUMONIA']

print(f"預測結果: {classes[class_idx]}")
print(f"信心度: {prediction[0][class_idx]:.2%}")
```

---

## 評估指標說明

### Precision (精確率)
在所有預測為肺炎的病例中，真正患有肺炎的比例。

```
Precision = TP / (TP + FP)
```

### Recall (召回率)
在所有實際肺炎病例中，模型正確識別出的比例。

```
Recall = TP / (TP + FN)
```

### F1-Score
精確率和召回率的調和平均數。

```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 模型改進建議

### 1. 使用遷移學習
- 採用預訓練模型: VGG16, ResNet50, DenseNet
- 凍結前層，微調後層

### 2. 增加訓練數據
- 擴充數據集
- 使用 GAN 生成合成影像
- 跨資料集訓練

### 3. 處理類別不平衡
- 使用加權損失函數
- 過採樣少數類別
- 欠採樣多數類別

### 4. 優化模型架構
- 增加 Dropout 層防止過擬合
- 使用 Learning Rate Scheduler
- 實驗不同的優化器

---

## 醫療免責聲明

**重要提示**: 本系統僅供學術研究和教育用途，不應用於實際臨床診斷。任何醫療決策應由專業醫療人員根據完整的臨床資訊做出。

---

## 貢獻指南

歡迎提交 Pull Request 或 Issue 來改進本專案。

### 貢獻方式
1. Fork 本專案
2. 創建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

---

## 參考資料

### 資料集來源
- Kaggle: Chest X-Ray Images (Pneumonia)
- NIH Chest X-ray Dataset

### 相關資源
- [TensorFlow 官方文檔](https://www.tensorflow.org/)
- [Keras 應用指南](https://keras.io/)
- [OpenCV 文檔](https://docs.opencv.org/)

---

## 授權

MIT License

---

## 作者

廖子婷

---

**最後更新**: 2024-12-02

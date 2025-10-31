
#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd  # 使用 pandas 來讀取 Excel 文件

# 使用 pandas 讀取 Excel 文件
df = pd.read_excel('11_banana_quality.xlsx')
"""
Size	Weight	Sweetness	Softness	HarvestTime	Ripeness	Acidity

"""
 
# 提取需要的列數據
t1 = df["Size"].values.astype(np.float64)  # 提取 "Humidity9am" 列，轉為 numpy array
len = t1.shape[0]
X = np.reshape(t1, (len, 1))

# 依次追加其他需要的列
X = np.append(X, np.reshape(df["Weight"].values.astype(np.float64), (len, 1)), axis=1)
X = np.append(X, np.reshape(df["Sweetness"].values.astype(np.float64), (len, 1)), axis=1)
X = np.append(X, np.reshape(df["Softness"].values.astype(np.float64), (len, 1)), axis=1)
X = np.append(X, np.reshape(df["HarvestTime"].values.astype(np.float64), (len, 1)), axis=1)
X = np.append(X, np.reshape(df["Ripeness"].values.astype(np.float64), (len, 1)), axis=1)
X = np.append(X, np.reshape(df["Acidity"].values.astype(np.float64), (len, 1)), axis=1) 

# 提取標籤數據
Y = df["Label"].values.astype(np.int8)   # 3~8  -> 0~5

# 分割數據集為訓練集與測試集
category = 2
dim = X.shape[1]   # 輸入維度
print("dim=", dim)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05)

# 將標籤轉換為 One-Hot 編碼格式
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=(category))
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=(category))

print("x_train[:11]", x_train[:11])
print("y_train[:11]", y_train[:11])
print("y_train2[:11]", y_train2[:11])

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation=tf.nn.relu, input_dim=dim))
model.add(tf.keras.layers.Dense(units=100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=category, activation=tf.nn.softmax))

# 編譯模型
model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train2, epochs=200, batch_size=64)

# 測試模型
model.summary()

# 評估模型
score = model.evaluate(x_test, y_test2, batch_size=64)
print("score:", score)

# 預測並顯示結果
predict = model.predict(x_test)
print("predict:", predict)
predict2 = np.argmax(predict, axis=1)  # 取得每一個��本的����類別
print("predict_classes:", predict2)
print("y_test", y_test[:])

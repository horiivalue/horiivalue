import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Keras
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation

# データの分割
from sklearn.model_selection import train_test_split

# MSE
from sklearn.metrics import mean_squared_error

# JupyterNotebook上でグラフを表示する設定
# DataFrameで全ての列を表示する設定
pd.options.display.max_columns = None

dataset_tenki = pd.read_csv("data4.csv")
dataset_tenki.head()

# 列の抜き出し
dataset = dataset_tenki[['平均気温(℃)', '最高気温(℃)','最低気温(℃)', '平均風速(m/s)','最大風速(m/s)', '平均現地気圧(hPa)', '平均雲量(10分比)', '天気概況(昼：06時～18時)']]

# 列名をリネームする
dataset = dataset.rename(columns={'平均気温(℃)': 'kion1', '最高気温(℃)': 'kion2','最低気温(℃)': 'kion3','平均風速(m/s)': 'wind1','最大風速(m/s)': 'wind2', \
    '平均現地気圧(hPa)': 'kiatsu', '平均雲量(10分比)': 'cloud', '天気概況(昼：06時～18時)': 'tenki1'})

dataset.head()

# 文字列に雨が含まれていたら-1を返す
for i in range(366):
    s = dataset.iloc[i,7].find('雨')
    if s == -1:
        dataset.iloc[i,7] = 0
    else:
        dataset.iloc[i,7] = 1
dataset.head()


# 二値分類モデルを使って分析しやすくするため
is_tenki1 = (dataset['tenki1']==0).astype(np.int64)
dataset['is_tenki1'] = is_tenki1
dataset.head()


# 目的関数と説明変数を設定
Y = np.array(dataset['is_tenki1'])
X = np.array(dataset[['kion1','kion2','kion3','wind1','wind2','kiatsu','cloud']])
print("Y=", Y.shape, ", X=", X.shape)


# 数値にばらつきがあるので整える。
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)

print("Y_train=", Y_train.shape, ", X_train=", X_train.shape)
print("Y_valid=", Y_valid.shape, ", X_valid=", X_valid.shape)
print("Y_test=", Y_test.shape, ", X_test=", X_test.shape)

# モデルの初期化
model = keras.Sequential()

# 入力層
model.add(Dense(8, activation='relu', input_shape=(7,)))
# 隠れ層
model.add(Dense(8, activation='relu'))
# 出力層
model.add(Dense(1, activation='sigmoid'))

# モデルの構築
model.compile(optimizer = "rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# 学習の実施
log = model.fit(X_train, Y_train, epochs=5000, batch_size=32, verbose=True,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         min_delta=0, patience=100,
                                                         verbose=1)],
         validation_data=(X_valid, Y_valid))


plt.plot(log.history['loss'], label='loss')
plt.plot(log.history['val_loss'], label='val_loss')
plt.legend(frameon=False) # 凡例の表示
plt.xlabel("epochs")
plt.ylabel("crossentropy")
plt.show()

# predictで予測を行う
Y_pred = model.predict(X_test)

# 二値分類は予測結果の確率が0.5以下なら0,
# それより大きければ1となる計算で求める
Y_pred_cls = (Y_pred > 0.5).astype("int32")

Y_pred_ = Y_pred_cls.reshape(-1)

Y_pred

Y_pred_

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred_))

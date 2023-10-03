#各種のインポート
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
%matplotlib inline
# DataFrameで全ての列を表示する設定
pd.options.display.max_columns = None

#データをPandasで読み込み
df = pd.read_csv("ritu-jk2322.csv")

#データを5行だけ表示
df.head()

# 列の抜き出し
dataset = dataset_keizai[['国内総生産(支出側)', '家計最終消費支出','輸出', '民間企業設備','輸入']]

# 列名をリネームする
dataset = dataset.rename(columns={'国内総生産(支出側)': 'keizai', '家計最終消費支出': 'kakei','輸出': 'export','民間企業設備': 'minkan','輸入':'import'})

dataset.head()

for i in range(117):
    s = dataset.iloc[i,0]
    if s > 0:
        dataset.iloc[i,0] = 1
    else:
        dataset.iloc[i,0] = 0
dataset.head()

Y = np.array(dataset['keizai'])
X = np.array(dataset[['kakei','export','minkan','import']])

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)
X

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)

# モデルの初期化
model = keras.Sequential()

# 入力層
model.add(Dense(8, activation='relu', input_shape=(4,)))
# 隠れ層
model.add(Dense(8, activation='relu'))
# 出力層
model.add(Dense(1, activation='sigmoid'))

# モデルの構築
model.compile(optimizer = "rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

%%time
# 学習の実施
log = model.fit(X_train, Y_train, epochs=5000, batch_size=24, verbose=True,
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
from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred_))


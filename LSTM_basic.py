import os
os.chdir('D:\\GitHub\\DL4\\DL_basic')
os.getcwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.metrics import mean_squared_error


#####################
# データセット表示用コード
#####################
# 1949年1月から1960年12月までの月次航空会社乗客数
# このデータセットには、1列目に年月、2列目にその月の航空会社乗客数が千人単位で記録されている
dataframe = pd.read_csv('international-airline-passengers.csv', index_col=0, engine='python', skipfooter=3)
dataframe.columns = ['passengers (k)']
dataframe.index = pd.to_datetime(dataframe.index)
dataframe.plot()

# データの読み込み（データセット中のYのみロード） 2列目の乗客数のみ読み込み
dataframe = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values.astype('float32')


#####################
# データの正規化
#####################
# データが人数という単位を持っているので、この単位への依存性をなくすため（また予測性能向上のため）スケーリングを行う
# スケーリングには、ライブラリscikit-learnのMinMaxScalerクラスを使用
# MinMaxScalerは初期化時にfeature_range引数として、どの範囲にスケーリングするかを指定すると、
# fit_trasformメソッドでデータセットをスケーリングし、またスケーリングに用いたパラメータを記憶する
####### [0,1]に正規化 #######
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


#####################
# 訓練データとテストデータ作成
#####################
# 訓練データとテストデータを分割（時系列の予測が目的のため、ある時点から先をテストデータとする）
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
x_whole = dataset[:,:]

# 時系列の予測の場合、ある時点のデータのみ参照して同じ時点の予測を行うことは困難
# そこで、数回前のデータまで参照できるようにする
# RNNでは、この数回前までのデータを1系列とみなして入力に使用する
# X=[data[t-look_back],...,data[t-1]], Y=data[t]となるデータセットに変換
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 5
x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)
x_whole, _ = create_dataset(x_whole, look_back)

# [samples, time steps, features]へ変形
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_whole = np.reshape(x_whole, (x_whole.shape[0], x_whole.shape[1], 1))


#####################
# モデル構築
#####################
# LSTM（4ユニット）を使用してモデルを構築
# 最終的に必要な出力はスカラー値なので、最後にDenseレイヤーを使用し、損失関数はMSEを使用
model = Sequential()

model.add(LSTM(4, input_shape=(look_back, 1))) # input_shape=(系列長T, x_tの次元), output_shape=(units,)
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)
loss = history.history['loss']

plt.plot(np.arange(len(loss)), loss, label="LSTM")
plt.legend()
plt.show()


#####################
# モデルの可視化
#####################
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')


#####################
# モデル評価
#####################
# 予測の精度をRMSEで評価、イメージが湧くように、正規化を解いて人数の単位で出力
# テストデータに対する予測（評価のため訓練データも）
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
y_pred = model.predict(x_whole)

# 正規化を元に戻す
# inverse_transformメソッドで逆変換、データのスケーリングを元に戻す
train_pred = scaler.inverse_transform(train_pred)
y_train = scaler.inverse_transform([y_train])
test_pred = scaler.inverse_transform(test_pred)
y_test = scaler.inverse_transform([y_test])
y_pred = scaler.inverse_transform(y_pred)

# 平均二乗誤差のルートで評価
train_score = math.sqrt(mean_squared_error(y_train[0], train_pred[:,0]))
print('Train RMSE: %.2f' % (train_score))
test_score = math.sqrt(mean_squared_error(y_test[0], test_pred[:,0]))
print('Test RMSE: %.2f' % (test_score))

# 予測データのプロットを行って視覚的に評価
# 訓練データ部分の予測データ
train_pred_plot = np.empty_like(dataset)
train_pred_plot[:, :] = np.nan
train_pred_plot[look_back:len(train_pred)+look_back, :] = train_pred

# テストデータ部分の予測データ
test_pred_plot = np.empty_like(dataset)
test_pred_plot[:, :] = np.nan
test_pred_plot[len(train_pred)+(look_back*2)+1:len(dataset)-1, :] = test_pred

# 全体の予測データ
y_pred_plot = np.empty_like(dataset)
y_pred_plot[:, :] = np.nan
y_pred_plot[look_back:len(dataset)-1, :] = y_pred

# プロット
plt.plot(scaler.inverse_transform(dataset), label='Dataset')
plt.plot(train_pred_plot, label='Prediction under train data')
plt.plot(test_pred_plot, label='Prediction under test data')
plt.legend()
plt.grid(True)
plt.show()

# プロット
plt.plot(scaler.inverse_transform(dataset), label='Dataset')
plt.plot(y_pred_plot, label='Prediction')
plt.legend()
plt.grid(True)
plt.show()























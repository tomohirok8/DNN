import os
os.chdir('D:\\GitHub\\DL4\\DL_basic')
os.getcwd()
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist       # データ読み込み用
from keras.utils import to_categorical # データ読み込み用

# Inputレイヤーからスタート（返り値はテンソル）
inputs = Input(shape=(784,))

# レイヤークラスのインスタンスはテンソルを引数に取れる（返り値はテンソル）
x = Dense(128, activation='relu')(inputs)      # InputレイヤーとDenseレイヤー(1層目)を接続　（隠れ層インスタンス生成＋コール）
x = Dense(64, activation='relu')(x)            # Denseレイヤー(1層目)とDenseレイヤー(2層目)を接続　（隠れ層インスタンス生成＋コール）
output_layer = Dense(10, activation='softmax') # レイヤーのインスタンス化を切り分けることももちろん可能　（出力層インスタンス生成）
                                               # (別のモデル構成時にこのレイヤーを指定・再利用することも可能になる)
predictions = output_layer(x)                  # Denseレイヤー(2層目)とDenseレイヤー(3層目)を接続 　（隠れ層コール）

# Modelクラスを作成（入力テンソルと出力テンソルを指定すればよい）
model = Model(inputs=inputs, outputs=predictions) # これで、「(784,)のInputを持つDense3層」構成のモデルが指定される

# 以降はSequentialと同じ
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train)

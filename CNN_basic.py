import os
os.chdir('D:\\GitHub\\DL4\\DL_basic')
os.getcwd()
import keras
#import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.datasets import cifar10
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
#from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.datasets import fashion_mnist
#from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

random_state = 42


##########################################
####### Fasion MNISTをCNNでクラス分類 #######
##########################################
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05,
                    wspace=0.05)
# このとき読み込んだ画像は(バッチサイズ、縦の画素数、 横の画素数)の次元で表されています。
for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap='gray')
#x_train.shape

# CNNでは2次元の画像として処理していくために4次元テンソル (バッチサイズ、縦の画素数、横の画素数、チャンネル数)として扱います。 
# チャンネル数は白黒画像の場合は1、 カラー画像の場合はRGBで3です
# Fashion MNISTの画像は白黒データですのでチャンネル数を1に設定しています。
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#x_train.shape


####### モデルを実装 #######
model = Sequential()

# 入力画像 28x28x1 (縦の画素数)x(横の画素数)x(チャンネル数)
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                 kernel_initializer='he_normal', input_shape=(28, 28, 1)))  # 28x28x1 -> 24x24x16
model.add(MaxPooling2D(pool_size=(2, 2)))  # 24x24x16 -> 12x12x16
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',
                 kernel_initializer='he_normal'))  # 12x12x16 -> 8x8x64
model.add(MaxPooling2D(pool_size=(2, 2)))  # 8x8x64 -> 4x4x64

model.add(Flatten())  # 4x4x64-> 1024
model.add(Dense(10, activation='softmax'))  # 1024 -> 10

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)

####### モデルを可視化 #######
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

early_stopping = EarlyStopping(patience=1, verbose=1)
model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, verbose=1,
          validation_data=(x_test, y_test), callbacks=[early_stopping])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, 1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('submission.csv', header=True, index_label='id')


##########################################
####### CIFAR10のデータをCNNでクラス分類 #######
##########################################
(x2_train, y2_train), (x2_test, y2_test) = cifar10.load_data()

# 画像はRGBデータなのでFashion MNISTとは異なり、チャンネル数は3になります。
x2_train = x2_train.astype('float32') / 255
y2_train = np.eye(10)[y2_train.astype('int32').flatten()]
x2_test = x2_test.astype('float32') / 255
y2_test = np.eye(10)[y2_test.astype('int32').flatten()]

x2_train, x2_valid, y2_train, y2_valid = train_test_split(x2_train, y2_train, test_size=10000)
#x2_train.shape

# CIFAR-10の画像の例を表示してみます。この画像ひとつひとつに10のカテゴリのうちひとつが付与されています。
fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05,
                    wspace=0.05)

for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.imshow(x2_train[i])


####### モデルを実装 #######
model2 = Sequential()

model2.add(Conv2D(6, kernel_size=(5, 5), activation='relu',
                 kernel_initializer='he_normal', input_shape=(32, 32, 3)))  # 32x32x3 -> 28x28x6
model2.add(MaxPooling2D(pool_size=(2, 2)))  # 28x28x6 -> 14x14x6
model2.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                 kernel_initializer='he_normal'))  # 14x14x6 -> 10x10x16
model2.add(MaxPooling2D(pool_size=(2, 2)))  # 10x10x16 -> 5x5x16

model2.add(Flatten())  # 5x5x16 -> 400
model2.add(Dense(120, activation='relu',
                kernel_initializer='he_normal'))  # 400 ->120
model2.add(Dense(84, activation='relu', kernel_initializer='he_normal'))  # 120 ->84
model2.add(Dense(10, activation='softmax'))  # 84 ->10

model2.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)


####### モデルを可視化 #######
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

# Jupyter用
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model2, show_shapes=True).create(prog='dot', format='svg'))

early_stopping = EarlyStopping(patience=1, verbose=1)
model2.fit(x=x2_train, y=y2_train, batch_size=128, epochs=100, verbose=1,
          validation_data=(x2_valid, y2_valid), callbacks=[early_stopping])

y2_pred = model2.predict(x2_test)
y2_pred = np.argmax(y2_pred, 1)

submission2 = pd.Series(y2_pred, name='label')
submission2.to_csv('submission2.csv', header=True, index_label='id')


# DataAugumentationや画像データの正規化を学習に反映させてみます

datagen = ImageDataGenerator(
    width_shift_range=0.2,  # 3.1.1 左右にずらす
    height_shift_range=0.2,  # 3.1.2 上下にずらす
    horizontal_flip=True,  # 3.1.3 左右反転
    # 3.2.1 Global Contrast Normalization (GCN) (Falseに設定しているのでここでは使用していない)
    samplewise_center=False,
    samplewise_std_normalization=False,
    zca_whitening=False)  # 3.2.2 Zero-phase Component Analysis (ZCA) Whitening (Falseに設定しているのでここでは使用していない)


model2.fit_generator(datagen.flow(x2_train, y2_train, batch_size=100),
                    steps_per_epoch=x2_train.shape[0] // 100, epochs=30, validation_data=(x2_valid, y2_valid))

y3_pred = model2.predict(x2_test)
y3_pred = np.argmax(y3_pred, 1)

submission3 = pd.Series(y3_pred, name='label')
submission3.to_csv('submission3.csv', header=True, index_label='id')





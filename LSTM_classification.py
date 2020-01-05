import os
os.chdir('D:/GitHub/DL3')
os.getcwd()
import tensorflow as tf
print('tensorflow version', tf.__version__)
if int(tf.__version__.split('.')[0]) >= 2:
    from tensorflow import keras
else:
    import keras
from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Flatten, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
from sklearn import metrics

from read_data import read_imdb



#####################
# データの読み込み
#####################
train_data, test_data, train_labels, test_labels, reverse_word_index = read_imdb()

print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

x_vali = train_data[:10000]
x_train = train_data[10000:]
 
y_vali = train_labels[:10000]
y_train = train_labels[10000:]
	

#####################
# モデル構築
#####################
seqX_len = len(x_train[0])

vocab_dim = 256

vocab_size = len(reverse_word_index)

# Inputレイヤー（返り値としてテンソルを受け取る）
inputs = Input(shape=(seqX_len,), name='input')

# モデルの層構成（手前の層の返り値テンソルを次の接続したい層に引数として与える）
# InputレイヤーとEmbeddingレイヤーを接続、Embeddingレイヤーをインスタンス化
embedded = Embedding(vocab_size, vocab_dim, mask_zero=True, name='embedding')(inputs)
# shape: (seqX_len,)->(seqX_len, vocab_size)

# EmbeddingレイヤーとLSTMレイヤーを接続、LSTMレイヤーをインスタンス化
encoded = Bidirectional(LSTM(256, return_sequences=True))(embedded)
# shape: (seqX_len, vocab_size)->(256, 512)

# LSTMレイヤーとSeqSelfAttentionレイヤーとLSTMレイヤーを接続
encoded = SeqSelfAttention(name='attention')(encoded)

# SeqSelfAttentionレイヤーを接続
encoded = Bidirectional(LSTM(128))(encoded)
# shape: (256, 512)->(256, 256)

outputs = Dense(1, activation='sigmoid')(encoded)

# モデル構築（入力はEncoderとDecoder、出力はDecoderのみ）
model = Model(inputs=inputs, outputs=outputs)

# parse_categorical_crossentropy（正解ラベルとしてone_hot表現のベクトルでなく数値を受け取るcategorical_crossentropy）を使用
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 二値分類            :binary_crossentropy
# 多クラス単一ラベル分類  :categorical_crossentropy
# 多クラス多ラベル分類    :binary_crossentropy
# 回帰問題（任意の値）   :mse
# 回帰問題（０～１の値）  :mse / binary_crossentropy


#####################
# モデルの可視化
#####################
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')


model.summary()

#####################
# モデルの学習
#####################
history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_vali, y_vali),
                    verbose=1,
                    callbacks=[EarlyStopping(patience=5, monitor='val_acc', mode='max')])



history_dict = history.history
history_dict.keys()
 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(acc) + 1)
 
plt.figure(figsize=(16,9))
plt.subplots_adjust(hspace=0.3)
plt.subplot(2,1,1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlim(0,len(epochs))
plt.ylim(0,max([max(loss), max(val_loss)]))
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlim(0,len(epochs))
plt.ylim(0,1)
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()


#####################
# モデルによる生成
#####################
predict = model.predict(test_data, verbose=1)

predict_labels = [0] * predict.shape[0]
for i in range(predict.shape[0]):
    if predict[i] >=0.5: predict_labels[i]=1



print('accuracy =', metrics.accuracy_score(test_labels, predict_labels))
print('precision =', metrics.precision_score(test_labels, predict_labels))
print('recall =', metrics.recall_score(test_labels, predict_labels))
print('F1 =', metrics.f1_score(test_labels, predict_labels))





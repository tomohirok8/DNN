import os
os.chdir('D:\\GitHub\\DL4\\DL_basic')
os.getcwd()

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense


# 機械学習で頻繁に用いられる、MNISTの手書き数字データセットを用います。
# MNISTのデータセットも、全てのデータを使用するのではなく、学習用と評価用に予め分割してあるわけです。
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train[0].shape


# MNISTの手書き数字のデータを下記で表示
fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
# 各MNIST画像の上に（タイトルとして）対応するラベルを表示
for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.set_title(str(y_train[i]))
    ax.imshow(x_train[i], cmap='gray')


# 分類タスクの時の出力データはラベルですが、ラベルは数字としての大小には意味がないということです。
# というのも、グループの名前として数字を割り振っているだけであるためです。こうした数字を名義尺度と呼びます。
# 機械学習のアルゴリズムでは数字の大小に意味があるものとして扱ってしまうため、名義尺度をうまく変換しなければなりません。
# この名義尺度を変換する表現として使用されるのが、one-hot表現と呼ばれるものです。
# 入力画像を行列(28x28)からベクトル(長さ784)に変換
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 名義尺度の値をone-hot表現へ変換
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


####### モデル構築 #######
# optimizer（最適化手法）
# loss（損失関数）
# metrics（評価関数（任意））

# モデルの「容器」を作成
model = Sequential()

# ↓ 多層パーセプトロンを構成する
# 「容器」へ各layer（Dense, Activation）を積み重ねていく（追加した順に配置されるので注意）
# 最初のlayerはinput_shapeを指定して、入力するデータの次元を与える必要がある
model.add(Dense(units=256, input_shape=(784,),activation='relu'))
# ↑ Denseが隠れ層を作る、input_shapeが入力層の次元、unitsが次の層の次元
model.add(Dense(units=100,activation='relu'))
# ↑ 次の隠れ層を作っている、input_shapeは書かなくていい（書くなら256）
model.add(Dense(units=10,activation='softmax'))
# ↑ 出力層を作っている
# ↑ 分類問題のときは出力層の関数はsoftmaxになる（他クラス版のsigmoid関数、2クラスのときはsigmoidになる）

# モデルの学習方法について指定しておく
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# ↑ loss：誤差関数の指定（他クラスのときはこれ、交差誤差エントロピー）
# ↑ optimizer：sgd＝確率的的勾配降下法（データを小さい塊に分割する、メモリを節約）
# ↑ metrics：評価方法


# モデルの学習
# x：学習に使用する入力データ
# y：学習に使用する出力データ
# batch_size：学習中のパラメータ更新を1回行うにあたって用いるサンプル数（ミニバッチのサイズ）
# epochs：学習のエポック数
# verbose：学習のログを出力するか（0:しない、1：バーで出力、2:エポックごとに出力）
# validation_split/validation_data：検証用に用いるデータの割合（0～１の実数）、または検証用データそのもの（いずれかのみ指定可能）
# shuffle：各エポックごとにデータをシャッフルするか
# callbacks：訓練中のモデルの挙動を監視できるcallback関数を指定できます
model.fit(x_train, y_train,
          batch_size=1000, epochs=10, verbose=1,
          validation_data=(x_test, y_test))
####### epochs：パラメータの更新を何回やるか（パラメータの更新を何回やるか）


#####################
# モデルの可視化
#####################
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')


####### モデルの評価 #######
# 固定長のバッチごとに損失関数値または評価関数値を出力
# x：評価に使用する入力データ
# y：評価に使用する出力データ
# batch_size：1回の評価を行うにあたって用いるサンプル数
# verbose：評価のログを出力するか（0:しない、1：する(デフォルト)）
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


####### モデルによる予測 #######
# x：予測に使用する入力データ
# batch_size：まとめて1度に予測を行うサンプル数
# verbose：評価のログを出力するか（0:しない(デフォルト)、1：する）
classes = model.predict(x_test, batch_size=128)
print(classes[0])
print(y_test[0])


####### モデルの可視化 #######
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))



#####################
# 前処理
#####################
# 学習の高速化や性能向上などのために入力データを変換する前処理を行うことがあります。
# そのひとつに、データを学習において取り扱いやすいよう、特定の範囲にデータが収まるように変換するスケーリングがあります。
# 例えば、mnistの入力画像の各ピクセルの値は0~255の範囲の整数値ですが、それぞれ255で割ることで値を0~1の浮動小数点数に収めます。
x_train, x_test = x_train/255, x_test/255


#####################
# 最適化アルゴリズム (optimizer)
#####################
####### SGD #######
from keras.optimizers import SGD
SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# lr: 学習率、0以上の実数
# momentum: モーメンタム、0以上の実数（前回のパラメータ更新量を反映させる比率（3.2.2で説明））
# decay: 更新毎の学習率の減衰率、0以上の実数
# nesterov: Nesterov momentumを適用するかどうか（Trueならモーメンタム項の計算を1ステップ先読みして評価します（3.2.2で説明））
optimizer='sgd'
# Momentum
# 前ステップの更新を加味することで、 勾配の変化を滑らかにします。
# これによって、「通り過ぎる」ことによる行ったり来たりの振動を抑制することができます。
# また、勾配変化の少ない（パラメータ空間内の）通常の斜面においては、
# 他の勾配変化の大きい斜面と比較して学習率が上昇し、加速的に学習が進むという効果を持っています。
# Momentumでは前ステップの更新量を加味する割合を'momentum'として指定します。
# なお、momentum=0では通常のSGDと一致することもあり、KerasではSGDクラスのmomentum引数を調整することで使用できます。
# 通常はmomentum=0.9程度に設定します。
optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True)


####### Adagrad #######
# これまでは全パラメータに対して一様な学習率が設定されていましたが、各方向に対して勾配が異なることを考慮すれば、
# 各パラメータごとに学習率を変化させることができると、より効率的な最適化ができそうです。
# AdaGradでは、全体の学習率を各方向ごとに過去の勾配の累積で割り引くことで、
# 勾配が大きかった方向の学習率を下げ、小さかった方向の学習率を上げる工夫を導入しています。
# これによって、たとえ鞍点のようなある方向には勾配が小さいような状況でも、学習が進みやすくなります。
# Kerasでは,keras.optimizers.Adagradクラスを使用し、通常は学習率としてlr=0.01程度を用います。
# なお、AdaGradは学習の初期に勾配が大きいとすぐに更新量が小さくなってしまい、学習がストップしてしまうという欠点があるため、
# 学習率の選択、また重みの初期値の選択は慎重に行う必要があるという欠点をもっていることは気にしておくとよいでしょう。
from keras.optimizers import Adagrad
model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0))


####### RMSprop #######
# AdaGradでは勾配の蓄積が大きくなり、更新量が小さくなると二度と大きくなることがないという欠点がありました。
# RMSpropでは、この点に対処するため、勾配の情報が指数的な減衰によって次第に忘却されるように更新式を変更したことが特徴的になっています。
# （＝勾配の2乗の指数移動平均を使用）
# Kerasでは、keras.optimizers.RMSpropクラスを用います。勾配の指数移動平均を制御するパラメータとしてrhoが新たに指定できる点が特徴的です。
# rhoはどれだけ過去の勾配を重視するかを表し、通常はrho=0.9程度とすることが多いです。
from keras.optimizers import RMSprop
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0))


####### AdaDelta #######
# RMSpropによって学習率が不可逆的に悪化することを防ぐことができましたが、AdaGradの全体の学習率に鋭敏であるという性質はそのままです。
# この全体の学習率への鋭敏性、つまり問題設定毎に適切な学習率が変化してしまうという問題は、
# 実は更新量と勾配の次元の不一致を学習率で調整していることによるものです。（ここでの次元は物理的な次元のことで、いわゆる単位に相当するものです）
# そこで、AdaDeltaではそうした次元の不一致を加味して自動的に適切な学習率が設定されるようにしています。
# 具体的には、勾配の2乗の指数移動平均に加えて、更新量の2乗の指数移動平均をもちい、両者の比を学習率として設定しています。
# （なぜこれで次元の不一致に対処可能かは詳しく扱いませんが、Newton法が次元に対してロバストである＋Hessian逆行列の近似を利用して導出されます）
# Kerasでは、keras.optimizers.Adadeltaクラスを用います。RMSpropと同様に、更新量と勾配の指数移動平均を制御するパラメータrhoを設定できます。
# 通常rho=0.95とすることが推奨されています。
# なお、Kerasの実装では一応学習率lrを設定できるようになっていますが、AdaDeltaの提案論文では学習率は自動的に決定されるものとしている上、
# Kerasの公式HPでもlrはデフォルトのままとすることを推奨しているため、学習率の設定は基本的に不要です。
from keras.optimizers import Adadelta
model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0))


####### Adam #######
# AdaDeltaとは異なるRMSpropの改良法としてAdamが挙げられます。
# Adamでは、各方向への勾配の2乗に加えて勾配自身も、指数移動平均による推定値に置き換えています。
# これにより、ある種Momentumと似た効果が期待できます。
# Kerasでは、keras.optimizers.Adamクラスを使用します。
# パラメータとしては、勾配、勾配の2乗それぞれの指数移動平均を制御するパラメータとしてbeta_1,beta_2が新たに指定可能です。
# といっても、ほとんどの場合はデフォルトのパラメータが推奨され、実際に使用されています。
from keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

#####################
# 初期化 (initializer)
#####################
# 勾配に関するテクニックの3つめはパラメータの初期化についてです。
# 各層のパラメータは0を中心とした乱数で初期化しますが、大きすぎる値で初期化すれば学習の初期段階での勾配が過大になり、
# 逆に小さすぎる値だと勾配自体も過小になってしまい、いずれにしても学習はうまく進みません。
# そこで、初期化にあたっては、その値のスケール（分散）を適切に設定する必要があります。
# このパラメータの初期化にあたって比較的頻繁に用いられる手法として、LeCunによる手法、Glorotによる手法、Heによる手法が挙げられます。
# 特にGlorotの初期化法は活性化関数が全て線形な場合の解析結果であり、中央付近が線形とみなせるsigmoid関数やtanh関数に適していると言えます。
# また、Heの初期化法は活性化関数がReLUであるときに適しています。

####### LeCunの初期化 #######
model.add(Dense(128, activation='relu', kernel_initializer='lucun_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='lucun_normal'))


####### Glorotの初期化（Xavierの初期化） #######
model.add(Dense(128, activation='sigmoid', kernel_initializer='glorot_uniform'))
model.add(Dense(128, activation='sigmoid', kernel_initializer='glorot_normal'))


####### Heの初期化 #######
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))


#####################
# 過学習に対するテクニック
#####################

# 正則化 (regularization)
# 過学習が発生する一つの理由は、MLPのモデルは特に多くのパラメータ・自由度を持つために、
# 訓練データに対して、その本質的な部分以上に統計的ばらつきまで含めて完全にフィットしようとしてしまうことにあります。
# そこで過学習を回避するには、学習過程でいくつかのパラメータが自動的に機能しなくなると良いわけですが、これを実現するのが正則化です。
# 具体的には、損失関数にパラメータの大きさに対するペナルティ項（正則化項）を含めます。
# これにより, パラメータを自由に動けないように制限し、なるべく少ないパラメータでデータにフィットするようにできます。
# （実際の挙動としては、パラメータの値が0に近づくようにするということです。0であればパラメータは実質的に機能しません。）
# なお、正則化には様々な種類が存在し、主に次のL2,L1正則化またそれらを組み合わせたElasticNetが用いられます。
# 実際に学習したモデルのコストをチェックする際は、正則化項を含めない値が他のモデルに比べて減っているかを見る必要があります。
# というのも、正則化項はあくまで学習の都合上導入されたもので、予測の意味では、正則化項を含めない目的関数で評価すべきであるためです。
####### L2正則化 #######
# L2正則化では、全パラメータの2乗和を正則化項として損失関数に加えます。
# L2正則化では、パラメータを完全に0にすることは少ないものの、パラメータを滑らかにすることで予測精度のより良いモデルを構築します。
# Kerasではkeras.regularizers.l2として定義されていますが、実際には各layerにregularizerとして引数で指定して用います。
# なお、keras.regularizers.l2は引数として、正則化項に掛かる係数を指定できます。
from keras.layers import Dense
from keras import regularizers
model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))

####### L1正則化 #######
# L1正則化では、全パラメータの絶対値の和を正則化項として損失関数に加えます。
# L1正則化ではL2正則化よりもパラメータが0になりやすいという特徴（スパース性）があります。
# Kerasではkeras.regularizers.l1として定義されていますが、実際には各layerにregularizerとして引数で指定して用います。
# なお、keras.regularizers.l1は引数として、正則化項に掛かる係数を指定できます。
from keras.layers import Dense
from keras import regularizers
model.add(Dense(128, kernel_regularizer=regularizers.l1(0.01)))

####### ElasticNet #######
# L1正則化とL2正則化の組み合わせです。
# Kerasではkeras.regularizers.l1_l2として定義されていますが、実際には各layerにregularizerとして引数で指定して用います。
# なお、keras.regularizers.l1_l2は引数として、各々の正則化項に掛かる係数を指定できます。
from keras.layers import Dense
from keras import regularizers
model.add(Dense(128, kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.01)))


#####################
# 早期終了 (early stopping)
#####################
# 過学習に突入してからの学習は、未知のデータに対する予測力に寄与しない、（統計的なばらつき等の）本質的でない部分の学習なわけです。
# そこで、いっそのこと早々に学習を止めてしまうことで過学習を回避する手もあります。それが早期終了(early stopping)です。
# といっても、具体的にいつ学習を止めるかが重要です。
# 先述の通り、訓練データに対する誤差は最適化によってほとんどの場合低下しますが、検証データの誤差は過学習時には上昇します。
# そこで、検証データの誤差が大きくなってきた（或いは評価関数値が下がってきた）ところで学習をストップさせます。
# Kerasではcallbackという機能を使って各エポック毎のモデルにパラメータを保存し、検証データのコストが大きくなったら前のエポックのパラメータを使用するようにします。
# model.fitの引数に以下のように設定します。
model.fit(x=x_train, y=y_train, ..., callbacks=keras.callbacks.EarlyStopping(patience=0, verbose=1))


#####################
# ドロップアウト (dropout)
#####################
# 過学習で余計に学習している部分に着目すると、それは統計的なばらつきと言えるような部分でした。
# こうした確率的なばらつきは、一般に大量に足し合わせると相殺されます。
# そこで、訓練データセットから部分訓練データセットを大量に作成し、各モデルの予測結果を平均するアンサンブルという手法が用いられることがあります。
# このアンサンブル法は大変魅力的な手法なのですが、とてつもない計算量を要するためそのまま用いることは難しいものでした。
# そこで出てきたものが、ドロップアウト (dropout)と呼ばれる手法で、これは近似的にアンサンブル法を実現するものになっています。
# 具体的には、ドロップアウトは入力の一部をランダムに0にして出力するlayerの一種です。要するに一部のユニットを取り除いた状況を再現します。
# このユニットの除去を確率的に行い、一部のユニットが除去された部分ネットワークに対して学習することを繰り返すことで、
# 多数のモデルを同時に訓練することと同じ効果を再現しているわけです。
# Kerasでは、keras.layers.core.Dropoutクラスを用いて実装できます。
# rate: 入力を0にする確率、0～1の実数値
# seed: 乱数のシード値
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)




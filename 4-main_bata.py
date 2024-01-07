import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils  import np_utils
import numpy as np
import matplot as plt

physical_devices = tensorflow.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tensorflow.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tensorflow.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

#クラスラベル
labels = ['Lv26','Lv27','Lv28','Lv29',"Lv30",'Lv31','Lv32','Lv33','Lv34','Lv35','Lv36','Lv37']
#ディレクトリ
dataset_dir = "j:/musicgame/analyze_data/" # 前処理済みデータ
model_dir   = "j:/musicgame/analyze_data/cnn_h5"      # 学習済みモデル


#リサイズ
resize_setting = (300,300)


#メイン関数
def main():


  ### 1. データの前処理 ###

  #numpyデータの読み込み
  X_train = np.load(dataset_dir+'x_train.npy', allow_pickle=True)
  X_test = np.load(dataset_dir+'x_test.npy', allow_pickle=True)
  l_train = np.load(dataset_dir+'l_train.npy', allow_pickle=True)
  l_test = np.load(dataset_dir+'l_test.npy', allow_pickle=True)
  '''
  print(X_train.shape)
  '''
  '''
  #正規化（０〜１の範囲に丸める）
  X_train = X_train.astype('float16')/X_train.max()
  X_test = X_test.astype('float16')/X_test.max()

  #モノクロデータを4次元データに変換
  X_train_newarr = (X_train.shape[0],X_train.shape[1],X_train.shape[1],1)
  X_train = np.reshape(X_train,X_train_newarr)
  X_test_newarr = (X_test.shape[0],X_test.shape[1],X_test.shape[1],1)
  X_test = np.reshape(X_test,X_test_newarr)
  '''



  #クラスラベルの正解値の変更
  l_train = np_utils.to_categorical(l_train,len(labels))
  l_test = np_utils.to_categorical(l_test,len(labels))

  ### 2. モデル学習、評価　###

  #モデル学習(実行）
  model = model_train(X_train,l_train,X_test,l_test)
  #モデル評価
  model.evaluate(X_test,l_test,verbose=1)


#モデル学習関数
def model_train(X_train,l_train,X_test,l_test):

  #インスタンス（データのフレームワーク？大きい場所的な感じ）
  model = Sequential()

  #1層目(畳み込み層)
  model.add(Conv2D(32,3,3,padding='same',input_shape=X_train.shape[1:]))
  model.add(Activation('relu'))
  #2層目(畳み込み層)
  
  model.add(Conv2D(32,(3,3)))
  model.add(Activation('relu'))
  
  #3層目
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.3))
  # 6層目 (Max Pooling)

  model.add(MaxPooling2D(pool_size=(2,2)))
  
  # データを1列に並べる
  model.add(Flatten())
  # 7層目 (全結合層)
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(12))
  model.add(Activation('softmax'))
  # 最適化の手法
  opt = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.2, amsgrad=False)
  #opt = tensorflow.keras.optimizers.SGD(learning_rate=0.1)
  # 損失関数
  model.compile(loss="categorical_crossentropy",
                optimizer='Adam',
                metrics=["accuracy"]
                )
  #モデル学習
  
  log = model.fit(X_train,l_train,
            batch_size=8
            ,epochs=100
            ,verbose=1
            ,validation_data=(X_test,l_test))
  model.save(model_dir+'epoch+100')
  print(log)
  import matplotlib.pyplot as plt
  import datetime
  # グラフ表示
  now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
  
  #plt.figure(tight_layout=True)
 
  plt.title(now)

  
  plt.xlabel=("epochs")
  plt.ylabel=("loss")
  plt.plot(log.history['loss'], label='loss')
  plt.plot(log.history['val_loss'],label='val_loss')
  plt.legend(frameon=False) # 凡例の表示

  plt.show()

  plt.title(now)
  plt.xlabel=("epochs")
  plt.ylabel=("accuracy")
  plt.plot(log.history['accuracy'], label='accuracy')
  plt. plot(log.history['val_accuracy'],label='val_accuracy')
  plt.legend(frameon=False) # 凡例の表示
  
  plt.show()



  #モデル読み込み
  # model = keras.models.load_model('/content/music_game/cnn_h5')
  return model


model = main()

import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils  import np_utils
import sys
import numpy as np
from PIL import Image
from tensorflow import keras
import cv2

# クラスラベル
labels = ['Lv26','Lv27','Lv28','Lv29',"Lv30",'Lv31','Lv32','Lv33','Lv34','Lv35','Lv36','Lv37']
# ディレクトリ
dataset_dir = "j:/musicgame/analyze_data/" # 前処理済みデータ
model_dir   = "j:/musicgame/analyze_data/cnn_h5"      # 学習済みモデル


X_train = np.load(dataset_dir+'x_train.npy', allow_pickle=True)
X_test = np.load(dataset_dir+'x_test.npy', allow_pickle=True)
y_train = np.load(dataset_dir+'l_train.npy', allow_pickle=True)
y_test = np.load(dataset_dir+'l_test.npy', allow_pickle=True)
print(y_test)
"""
# 0~255の整数範囲になっているため、0~1間に数値が収まるよう正規化
X_train = X_train.astype('float16')/X_train.max()
X_test  = X_test.astype("float16") /  X_train.max()
"""
print(X_train.shape)
'''
X_train_newarr = (X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_train = np.reshape(X_train,X_train_newarr)
X_test_newarr = (X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_test = np.reshape(X_test,X_test_newarr)
'''

    # クラスラベルの正解値は1、他は0になるようワンホット表現を適用
y_train = np_utils.to_categorical(y_train,len(labels))
y_test  = np_utils.to_categorical(y_test,len(labels))
print(X_train.shape)
# リサイズ設定
resize_settings = (300,300)

# 推論用モデル
def predict():

    #インスタンス
    model = Sequential()
    # 1層目 (畳み込み）
    model.add(Conv2D(32,(3,3),padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    # 2層目（Max Pooling)
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    # 3層目 (Max Pooling)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    # 4層目 (畳み込み)
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation('relu'))
    # 5層目 (畳み込み)
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    # 6層目 (Max Pooling)
    model.add(MaxPooling2D(pool_size=(2,2)))
    # データを1列に並べる
    model.add(Flatten())
    # 7層目 (全結合層)
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # 出力層(softmaxで確率を渡す：当てはまるものを1で返す)
    model.add(Dense(12))
    model.add(Activation('softmax'))
    # 最適化の手法
    opt = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # 損失関数
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"]
                 )


    
    # モデル学習(推論では不要のためコメントアウト)
    # model.fit(X_train,y_train,batch_size=10,epochs=150)

    # モデルを読み込み
    model = keras.models.load_model("J:/musicgame/analyze_data/cnn_h5")

    return model

# 変換用関数
def pil2cv(image):
    import numpy as np
    import cv2
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    from PIL import Image
    import cv2
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

# 実行関数
def main(path):
    import statistics
    import math
    import gc
    i = 0
    moment_Lv_ave=[]
    moment_Lv_represent=[]
    import glob
    files = glob.glob(path+'/*')
    print(files)
    model = predict()
    for i in range(len(files)):

        X     = []                               # 推論データ格納
        image = Image.open(files[i]) 
        image = image.convert('L')               # 画像読み込み
        image = image.convert('RGB')             # GRAY変換
        image = image.resize(resize_settings)    # リサイズ
        data  = np.asarray(image)
        #print(data)

        #print(data.shape)
                 # 数値の配列変換
        X.append(data)
        X = np.array(X)
    
        #print(X.shape)
 

    # モデル呼び出し
        #model = predict()
        
        pre_Lv=[]
        l = 0
    # numpy形式のデータXを与えて予測値を得る
        model_output = model.predict([X])[0]
        for l in range(len(model_output)):
            a = model_output[l]*(l+26)
            pre_Lv.append(a)
            l+=1
            print(pre_Lv)
    #モデルの推定値（リスト形式）
        print(model_output)
        moment_Lv_ave.append(sum(pre_Lv))
        print(moment_Lv_ave)
        print('瞬間のレベル（平均値）')
        print('{:.2f}'.format(moment_Lv_ave[i]))

    # 推定値 argmax()を指定しmodel_outputの配列にある推定値が一番高いインデックスを渡す
        predicted = model_output.argmax()
    # アウトプット正答率
        accuracy = int(model_output[predicted] *100)
        print(accuracy)
        #print("{0} ({1} %)".format(labels[predicted],accuracy))
        print(labels[predicted])
        moment_Lv_represent.append(26+predicted)

        gc.collect()
        i+=1
    
    ave_Lv_ave = sum(moment_Lv_ave)/(len(moment_Lv_ave))
    ave_Lv_represent = sum(moment_Lv_represent)/(len(moment_Lv_represent))
    moment_Lv_ave = sorted(moment_Lv_ave)
    moment_Lv_represent = sorted(moment_Lv_represent)
    median_Lv_ave = statistics.median(moment_Lv_ave)
    median_Lv_represent = statistics.median(moment_Lv_represent)
    return ave_Lv_ave,ave_Lv_represent,median_Lv_ave,median_Lv_represent
path = 'J:/musicgame/predict_photo'
# ala = ave_Lv_ave | alr = ave_Lv_represent
ala, alr,mla,mlr = main(path)
print('瞬間の平均値による全体平均')
print(ala)
print('瞬間の代表値による平均値')
print(alr)
print('瞬間の平均値の中央値')
print(mla)
print('瞬間の代表値の中央値')
print(mlr)


from PIL import Image
import numpy as np
import os
import glob
import cv2
import random
import gc


"""
クラスラベル(テスト用)
labels = ['Lv36','Lv37','nothing']
"""

#クラスラベル（本番用）
labels = ['Lv26','Lv27','Lv28','Lv29',"Lv30",'Lv31','Lv32','Lv33','Lv34','Lv35','Lv36','Lv37']
 
#####直接操作する層、データの入力層#####

#ディレクトリ（ファイル構造）
dataset_dir = "J:/musicgame/analyze_data/"
model_dir = 'j:/musicgame/analyze_data/cnn_h5'


#リサイズ設定
resize_setting = (300,300)

#画像データ
X_train = []
l_train = []
X_test = []
l_test = []

#レベルごとの処理（種類ごと）
for class_num ,label in enumerate(labels):
  gc.collect()

  #写真のディレクトリ
  photos_dir = 'J:/musicgame/photo_mono/'+label


  #画像データの取得
  files = glob.glob(photos_dir+'/*.png')

  n = 1600
  #写真を順に取得
  for i in range(n):

  ###前処理層###
    
    num = random.randint(1,len(files))
    file = photos_dir+"/"+str(num)+'.png'
    print(file)
      #画像の読み込み
    try: 
      image = cv2.imread(file)
      print(image)
      #画像を数字配列に変換
      data = np.asarray(image,'bool')
    except TypeError:
      print('型のエラー')
      pass





      
    '''
    print(data.shape)
    '''


      #テストデータ追加
    if i%5 == 0:
      X_test.append(data)
      l_test.append(class_num)

      #学習データ追加
    else:
      X_train.append(data)
      l_train.append(class_num)

    i += 1




#numpy配列に変更
X_train = np.array(X_train)
X_test = np.array(X_test)
l_train = np.array(l_train)
l_test = np.array(l_test)

print(X_train.shape)
print(l_train.shape)
print(X_test.shape)
print(l_test.shape)

print(l_train)
print(len(X_train))

print(len(l_train))

print(len(X_test))
print(len(l_test))


#前処理データを保存
dataset = (X_train,X_test,l_train,l_test)
print(type(dataset))
np.save(dataset_dir+'x_train',X_train)
np.save(dataset_dir+'l_train',l_train)
np.save(dataset_dir+'x_test',X_test)
np.save(dataset_dir+'l_test',l_test)
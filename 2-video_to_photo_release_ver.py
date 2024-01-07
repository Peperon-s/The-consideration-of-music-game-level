def video_path(folder_path):
    import os
    import glob
    #動画を取得するためのパスの取得
    file = os.path.join(folder_path,'*.mp4')
    all_file_path = glob.glob(file)

    return(all_file_path)

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


def video_to_photo(video_path_list,encode_path,bairitu):
    import numpy as np
    import cv2
    import os
    from PIL import Image, ImageOps
    i = 0
    name_num = 0
    for i in range(len(video_path_list)):
        print(i)
        #count = 0 #フレーム用カウント
        video_path = video_path_list[i] #i番目のビデオのパスの取得(video_path関数による入力を考慮)
        video = cv2.VideoCapture(video_path)

        # フレーム数を取得
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # フレームレート(1フレームの時間単位はミリ秒)の取得
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))

        # n秒ごと
        n = 1
        # read間隔の設定
        read_interval = int((frame_rate * n) - 1)
        for l in range(frame_count):  # フレーム数分回す
            ret = video.grab()
            if ret is False:
                break
            if l % read_interval == 0:
                ret, work_frame = video.read()
                if ret is False:
                    break
                # ここにフレームへの処理が入る
                name_num +=1
                image = cv2pil(work_frame)
                size = (300,300)
                new_size = 1600
                center_x = int(image.width / 2)
                center_y = int(image.height / 2)
                image = image.crop((center_x - new_size / 2, center_y - new_size / 2, center_x + new_size / 2, center_y + new_size / 2))
                #image = image.resize(size)
                image = ImageOps.fit(image, size)
                image = pil2cv(image)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                ret ,image = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
                cv2.imwrite(encode_path +f"/{name_num}.png",image) 
        video.release()
    i+=1
       
            
"""
            image = cv2.GaussianBlur(image,(3,3),3)
            med_val = np.median(image)
            sigma = 0.33  # 0.33
            min_val = int(max(0, (1.0 - sigma) * med_val))
            max_val = int(max(255, (1.0 + sigma) * med_val))
            image = cv2.Canny(image, threshold1 = min_val, threshold2 = max_val)
 """
        
    
    

#実行層
import os
j = 29
while j <= 37:
    print(j)
    j_str = str(j)
    a = video_path("j:/musicgame/video_data/L"+j_str)
    print(a)
    try:
      os.mkdir("j:/musicgame/photo_mono/Lv"+j_str)
    except FileExistsError:
        pass
    video_to_photo(a,'j:/musicgame/photo_mono/Lv'+j_str,1.0)
    j += 1
    print(j)
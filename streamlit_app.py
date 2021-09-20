import streamlit as st
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image,ImageOps
import numpy as np

# 画像を正方形にする
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def main():
    # タイトル表示
    st.title('マスク有無判定')
    st.write('## 読み込んだ画像の人物がマスクを着用しているか判定します。')

    # file upload
    uploaded_file = st.file_uploader('Choose a image file')
    if uploaded_file is not None:
        try:
            # 画像を読み込む
            uploaded_img = Image.open(uploaded_file)
            uploaded_img = ImageOps.exif_transpose(uploaded_img)  # 画像を適切な向きに補正する
            st.image(uploaded_img)

            # 画像を256x256の正方形にする
            uploaded_img=expand2square(uploaded_img,(0, 0, 0))
            uploaded_img = uploaded_img.resize((256,256))
            st.image(uploaded_img)
            img_array = np.array(uploaded_img)

            # 判定
            model = tf.keras.models.load_model('mask-model_cnn.hdf5')
            pred = tf.nn.sigmoid(model.predict(img_array[None, ...]))
            prediction=1 - pred.numpy()[0][0]

            # 結果表示
            if prediction<0.5:  # マスクなし
                st.info(f"マスクを着けてください")
            else:  # マスクあり
                st.info(f"マスクを着けています")

        except:
            st.error("判定できませんでした・・・適切な画像をアップロードしてください！")



if __name__ == "__main__":
    main()
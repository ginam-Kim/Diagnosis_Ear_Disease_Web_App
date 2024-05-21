import streamlit as st
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import matplotlib.pyplot as plt 
import keras
from PIL import Image
import io
import matplotlib


st.set_page_config(page_title="귀 질환 진단 서비스", page_icon="📸")

st.image('./image/main.png', use_column_width=True)

st.markdown('''
<style>
    .big-header {
        text-align: center;
        padding: 30px 0;
        margin-bottom: 10px;
        border-bottom: 3px solid #ffd700;
    }
    .result {
        font-size: 20px; /* 결과 텍스트 크기 설정 */
    }
    .result2 {
        font-size: 15px; /* 작은 글꼴 크기 설정 */
        text-align: center;
        padding: 20px 0;
        margin-bottom: 10px;
        border-top: 3px solid #ffd700;
    }
            
    [data-testid="StyledFullScreenButton"]{
            visibility: hidden
    }
                    
</style>
''', unsafe_allow_html=True)


st.markdown('''
📌 가정용 귀 내시경을 통해 **직접** 찍은 사진에 대해 **정상/비정상 유무 및 확률**을 알려드립니다.
''')

st.markdown('''
📌 **이미지를 업로드**하거나 **예시 이미지를 선택**해 주세요.
''')



def recall(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + keras.backend.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + keras.backend.epsilon())
    return precision

def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+keras.backend.epsilon()))

# @st.cache_resource -> 사용해서 inference time 15초에서 5초정도로 감소
@st.cache_resource  
def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'precision': precision, 'recall': recall, 'f1': f1})


def predict_image(model, img_array):
    return model.predict(img_array)

def inference_crop(img_path, model, img_height=299, img_width=299):
    img_ori = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.image.resize(img_ori, [img_height, img_width])
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255

    predictions = predict_image(model, img_array)  # 모델 예측 호출

    # 클래스 및 클래스 확률 출력
    class_labels = ["Abnormal", "Normal"]  # 클래스 라벨 설정
    predicted_class = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class]
    Abnormal_probability = predictions[0][0]
    normal_probability = predictions[0][1]

    if predicted_class_label == "Abnormal":
        prob = Abnormal_probability
    else:
        prob = normal_probability

    return img_path, predicted_class_label, prob

###############################################################################################


def show_heatmap(image_path, model):
    original_image = Image.open(image_path)
    if original_image.mode == 'RGBA':
        original_image = original_image.convert('RGB')  # 알파 채널 제거
    cropped = tf.image.central_crop(original_image, central_fraction=0.8)
    encoded_image = tf.image.encode_png(cropped)

    # EagerTensor를 numpy 배열로 변환
    encoded_image_np = encoded_image.numpy()
    cropped_img_410 = Image.open(io.BytesIO(encoded_image_np))
    cropped_img = cropped_img_410.resize((299, 299))

    # tf.keras.preprocessing.image
    x = tf.keras.preprocessing.image.img_to_array(cropped_img)
    x = np.expand_dims(x, axis=0)
    # 이미지 데이터를 전처리하기 전에 복사하여 새로운 배열을 생성
    x = x.copy()
    x = tf.keras.applications.xception.preprocess_input(x)

    last_conv_layer = model.get_layer("block14_sepconv2_act")
    model_1 = tf.keras.Model(model.inputs, last_conv_layer.output)

    input_2 = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x_2 = model.get_layer("global_average_pooling2d_1")(input_2)
    x_2 = model.get_layer("dense_2")(x_2)
    x_2 = model.get_layer("dense_3")(x_2)
    model_2 = tf.keras.Model(input_2, x_2)

    with tf.GradientTape() as tape:
        output_1 = model_1(x)
        tape.watch(output_1)  # 마지막 층으로 미분하기 위한 준비
        preds = model_2(output_1)
        class_id = tf.argmax(preds[0])
        output_2 = preds[:, class_id]

    grads = tape.gradient(output_1, output_1)  # 그레디언트 계산
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # 식5 적용

    output_1 = output_1.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        output_1[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(output_1, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # 정규화

    img = tf.keras.preprocessing.image.load_img(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')  # 알파 채널 제거
    img = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = tf.convert_to_tensor(img)

    cropped = tf.image.central_crop(img_tensor, central_fraction=0.8)
    cropped_img = cropped.numpy().astype(np.uint8)

    heatmap = np.uint8(255 * heatmap)  # [0,255]로 변환

    jet = matplotlib.colormaps.get_cmap("jet")  # jet 컬러맵으로 표시
    color = jet(np.arange(256))[:, :3]
    color_heatmap = color[heatmap]

    color_heatmap = tf.keras.preprocessing.image.array_to_img(color_heatmap)
    color_heatmap = color_heatmap.resize((cropped_img.shape[1], cropped_img.shape[0]))
    color_heatmap = tf.keras.preprocessing.image.img_to_array(color_heatmap)

    overlay_img = color_heatmap * 0.4 + cropped_img  # 덧씌움
    overlay_img = tf.keras.preprocessing.image.array_to_img(overlay_img)

    # 수정 시도
    color_heatmap = tf.keras.preprocessing.image.array_to_img(color_heatmap)

    return color_heatmap, overlay_img


best_model = load_model("./model/crop_model.h5")


# 기본 이미지 경로
default_image_path1 = "./image/otitexterna_27.png" 
default_image_path2 = "./image/정상_1.png"


with st.spinner("### ⏳ 잠시만 기다려주세요."):


    col1, col2 = st.columns(2)
    with col1:
        use_default_image1 = st.toggle("예시 이미지(외이도염)", value=False)
    with col2:
        use_default_image2 = st.toggle("예시 이미지(정상)", value=False)

    # 파일 업로더
    uploaded_file = st.file_uploader("📸 이미지를 업로드해주시길 바랍니다!", type=['png', 'jpg', 'jpeg'])

    # 이미지 로드
    if uploaded_file is not None: # 이미지를 업로드한 경우
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # 모델 inference -> label 및 확률 출력
        path, label, prob = inference_crop(uploaded_file, best_model)
        # 히트맵, 히트맵+원본 출력
        hm, overlay = show_heatmap(uploaded_file, best_model)
        
        if label == 'Normal':
            emoji = '🙂'
            color = '#0000FF'
            label_kor = '정상'
        else:
            emoji = '😫'
            color = '#FF0000'
            label_kor = '비정상'

        st.markdown(f'''
        <div class="header">
            <h2><span style="color: {color};">{label_kor}{emoji}으로 판별되었습니다.</span></h2>
            <h2 class='result'><span style="color: {color};">{label_kor}일 확률이 {prob:.4f}입니다.</span></h2>
        </div>
        ''', unsafe_allow_html=True)
        
    elif use_default_image1 and not(use_default_image2): # 예시 이미지1을 선택한 경우
        image = Image.open(default_image_path1)
        st.image(image, caption='Default Image1.', use_column_width=True)

        path, label, prob = inference_crop(default_image_path1, best_model)
        hm, overlay = show_heatmap(default_image_path1, best_model)
        
        if label == 'Normal':
            emoji = '🙂'
            color = '#0000FF'
            label_kor = '정상'
        else:
            emoji = '😫'
            color = '#FF0000'
            label_kor = '비정상'

        st.markdown(f'''
        <div class="header">
            <h2><span style="color: {color};">{label_kor}{emoji}으로 판별되었습니다.</span></h2>
            <h2 class='result'><span style="color: {color};">{label_kor}일 확률이 {prob:.4f}입니다.</span></h2>
        </div>
        ''', unsafe_allow_html=True)


        
    elif use_default_image2 and not(use_default_image1): # 예시 이미지2를 선택한 경우
        image = Image.open(default_image_path2)
        st.image(image, caption='Default Image2.', use_column_width=True)    

        path, label, prob = inference_crop(default_image_path2, best_model)
        hm, overlay = show_heatmap(default_image_path2, best_model)
        
        if label == 'Normal':
            emoji = '🙂'
            color = '#0000FF'
            label_kor = '정상'
        else:
            emoji = '😫'
            color = '#FF0000'
            label_kor = '비정상'

        st.markdown(f'''
        <div class="header">
            <h2><span style="color: {color};">{label_kor}{emoji}으로 판별되었습니다.</span></h2>
            <h2 class='result'><span style="color: {color};">{label_kor}일 확률이 {prob:.4f}입니다.</span></h2>
        </div>
        ''', unsafe_allow_html=True)    



    if (uploaded_file and not use_default_image1 and not use_default_image2) or \
    (not uploaded_file and use_default_image1 and not use_default_image2) or \
    (not uploaded_file and not use_default_image1 and use_default_image2):
        with st.expander("🔍 모델이 어느 부분을 보고 특정 클래스로 분류하였는지 시각적으로 확인해보아요!", expanded=True):

            col1, col2 = st.columns([4.5, 4.5])
            with col1: # 히트맵 이미지 들어갈 곳
                image = Image.open(default_image_path1)
                st.image(hm, caption='Heatmap', use_column_width=True)
                
            with col2: # 원본 + 히트맵 이미지 들어갈 곳
                image = Image.open(default_image_path2)
                st.image(overlay, caption='Overlapped Image', use_column_width=True)

            # with col3:
            #     st.image('./image/color_bar.png', caption='', use_column_width=True)
                

            st.image('./image/color_bar_가로.png', caption='', use_column_width=True)

            st.markdown('''
                ❗ **빨간 부분**일수록, 모델이 판단의 근거로써 중요하게 고려한 부분입니다.
            ''')            



st.markdown('''
<div>
    <h2 class='result2'><span style="color:#000000	;">CLEAR(강하람, 김기남, 좌대현)</span></h2>
</div>
''', unsafe_allow_html=True)
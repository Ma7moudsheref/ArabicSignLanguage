import streamlit as st
import cv2
import pickle
import mediapipe as mp
import numpy as np
import requests
import os
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display

st.set_page_config(page_title="مترجم لغة الإشارة العربي", layout="centered")

# --- دالة تحميل الموديل من رابط MediaFire المباشر ---
@st.cache_resource
def download_and_load_model():
    # هذا الرابط المباشر للملف اللي أنت رفعته
    file_url = "https://www.mediafire.com/file/slwpbp2cqiw9gp8/arabic_model.p/file"
    model_path = "arabic_model.p"
    
    if not os.path.exists(model_path):
        with st.spinner('جاري تحميل الموديل لأول مرة، انتظر لحظة...'):
            # ملاحظة: ميديا فاير يحتاج أحياناً ضغطة يدوية، لكن سنحاول تحميله برمجياً
            # إذا فشل التحميل البرمجي، سنطلب من المستخدم التأكد
            r = requests.get(file_url, allow_redirects=True)
            with open(model_path, 'wb') as f:
                f.write(r.content)
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# محاولة تحميل الموديل
try:
    data = download_and_load_model()
    model = data['model']
    label_encoder = data['label_encoder']
    st.success("✅ تم اتصال السيرفر بالموديل بنجاح!")
except Exception as e:
    st.error("⚠️ فشل في تحميل الموديل تلقائياً. تأكد من رفع الملف arabic_model.p بجانب الكود أو تحديث الرابط.")

st.title("مترجم لغة الإشارة العربي 🖐️")

# --- إعدادات الكاميرا ---
img_file_buffer = st.camera_input("التقط صورة لإشارة يدك للترجمة")

if img_file_buffer is not None:
    # تحويل الصورة إلى تنسيق OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # تحويل الألوان لـ MediaPipe
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # إعداد MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # استخراج النقاط (نفس منطق بايتشارم)
            landmarks = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
            landmarks = landmarks - landmarks[0]
            max_v = np.abs(landmarks).max()
            if max_v > 0: landmarks /= max_v
            
            distances = np.linalg.norm(landmarks, axis=1)
            angle = np.arctan2(landmarks[8][1], landmarks[8][0])
            data_in = np.hstack([landmarks.flatten(), distances, [angle]])
            
            # التوقع
            prediction = model.predict([data_in])[0]
            char = label_encoder.inverse_transform([prediction])[0]
            
            # عرض النتيجة بشكل شيك
            st.balloons()
            st.markdown(f"<h1 style='text-align: center; color: #00ffcc;'>الحرف المتوقع: {char}</h1>", unsafe_allow_html=True)
    else:
        st.warning("لم يتم رصد يد في الصورة، حاول مرة أخرى.")

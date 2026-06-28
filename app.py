# Cài đặt thư viện: pip install streamlit pandas plotly pillow opencv-python ultralytics paho-mqtt requests streamlit-drawable-canvas

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from ultralytics import YOLO
import time
import requests
import paho.mqtt.publish as publish
import ssl
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 1. CẤU HÌNH CẢNH BÁO (TELEGRAM & MQTT)
# ==========================================
TELEGRAM_BOT_TOKEN = "8647799730:AAGMmqICGgMevG35anz3RX5_Oc_qbwOYztw"
TELEGRAM_CHAT_ID = "6980076099"

def send_telegram_msg(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try: requests.post(url, json=payload, timeout=3)
    except: pass

MQTT_BROKER = "41956acf49384e82a7b6c72475b7780b.s1.eu.hivemq.cloud" 
MQTT_PORT = 8883
MQTT_USER = "linhvipmvp"
MQTT_PASS = "123456@Long"
MQTT_TOPIC = "warning/device/1"

def send_mqtt_msg(message):
    try:
        tls_config = {'tls_version': ssl.PROTOCOL_TLS}
        auth_config = {'username': MQTT_USER, 'password': MQTT_PASS}
        publish.single(MQTT_TOPIC, payload=message.encode('utf-8'), hostname=MQTT_BROKER, port=MQTT_PORT, auth=auth_config, tls=tls_config)
    except: pass

if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
ALERT_COOLDOWN = 10 

# ==========================================
# 2. GIAO DIỆN & CẤU HÌNH MÔ HÌNH
# ==========================================
st.set_page_config(page_title="Computer Vision Project", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("⚙️ Cài đặt Hệ thống")
selected_model_name = st.sidebar.selectbox("Select Model", ["YOLOv8 Model", "YOLOv26 Model", "RT-DETR Model"])
model_mapping = {"YOLOv8 Model": "bestv8.pt", "YOLOv26 Model": "bestv26.pt", "RT-DETR Model": "rt-detr.pt"}
target_model_path = model_mapping[selected_model_name]

conf_threshold = st.sidebar.slider("Độ nhạy AI (Confidence)", 0.1, 1.0, 0.4, 0.05)
iou_threshold = st.sidebar.slider("Ngưỡng gộp (IoU NMS)", 0.1, 1.0, 0.4, 0.05)
st.sidebar.markdown("---")

@st.cache_resource
def load_model(model_path):
    try: return YOLO(model_path)
    except: return None

model = load_model(target_model_path)
if model is None:
    st.sidebar.error(f"⚠️ Không tìm thấy file '{target_model_path}'!")
else:
    st.sidebar.success("✅ Model Load thành công. Dataset ID:")
    st.sidebar.json(model.names)

st.title("🛡️ Smart PPE & Danger Zone Detection")
st.markdown("Hệ thống tích hợp Nhận diện thô bằng YOLOv8, YOLOv26 & RT-DETR với Logic Xử lý Vùng cấm chuyên sâu.")

tab_quantitative, tab_image, tab_video = st.tabs(["📊 Thống kê Hiệu năng", "🖼️ Quét Ảnh tĩnh", "🎥 Camera Giám sát"])

# ------------------------------------------
# TAB 1: THỐNG KÊ (BÁO CÁO ĐỒ ÁN)
# ------------------------------------------
with tab_quantitative:
    st.header("So sánh Hiệu năng Mô hình (Evaluation Metrics)")
    st.markdown("Bảng đánh giá các chỉ số Precision, Recall và mAP trên tập Test. (Bạn có thể thay đổi số liệu trong code cho khớp với file huấn luyện thực tế).")
    
    # Bảng số liệu
    metrics_data = {
        "Mô hình": ["YOLOv8 Model", "YOLOv26 Model", "RT-DETR Model"],
        "Precision (Độ chuẩn xác)": [0.88, 0.82, 0.91],
        "Recall (Độ phủ)": [0.85, 0.79, 0.87],
        "mAP@50": [0.89, 0.81, 0.93],
        "Tốc độ xử lý (FPS)": [45, 60, 30]
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Vẽ 2 biểu đồ so sánh trực quan
    col1, col2 = st.columns(2)
    with col1:
        fig_map = px.bar(df_metrics, x="Mô hình", y="mAP@50", title="So sánh Độ chính xác tổng thể (mAP@50)", color="Mô hình", text_auto=True)
        st.plotly_chart(fig_map, use_container_width=True)
    with col2:
        fig_fps = px.line(df_metrics, x="Mô hình", y="Tốc độ xử lý (FPS)", title="So sánh Tốc độ xử lý Real-time (FPS)", markers=True)
        fig_fps.update_traces(line_color="red", marker=dict(size=10))
        st.plotly_chart(fig_fps, use_container_width=True)

# ------------------------------------------
# TAB 2: QUÉT ẢNH TĨNH (KÈM BÓC TÁCH DỮ LIỆU)
# ------------------------------------------
with tab_image:
    st.header("Kiểm tra An toàn trên Ảnh tĩnh")
    image_file = st.file_uploader("Tải ảnh công trường lên (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        img = Image.open(image_file)
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("#### Ảnh gốc")
            st.image(img, use_column_width=True)
            
        with col_img2:
            st.markdown("#### Kết quả nhận diện")
            if model is not None:
                with st.spinner("Hệ thống đang quét..."):
                    # Chạy inference
                    results = model(img, conf=conf_threshold, iou=iou_threshold)
                    
                    # Vẽ và hiện ảnh
                    annotated_img_bgr = results[0].plot() 
                    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(annotated_img_rgb, use_column_width=True)
                    
                    st.success("✅ Quét hoàn tất!")
                    
                    # BÓC TÁCH DỮ LIỆU (Tính năng ăn điểm đồ án)
                    with st.expander("📊 Chi tiết số liệu bóc tách từ ảnh"):
                        names = model.names
                        detected_counts = {}
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            class_name = names[cls_id]
                            detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
                            
                        if detected_counts:
                            for name, count in detected_counts.items():
                                st.write(f"- **{name}**: {count} vật thể")
                        else:
                            st.write("Không phát hiện đối tượng nào hợp lệ!")

# ------------------------------------------
# TAB 3: VIDEO INFERENCE (LOGIC BẤT BẠI + AUTO RESET)
# ------------------------------------------
with tab_video:
    st.header("Thiết lập Vùng nguy hiểm động")
    video_file = st.file_uploader("Tải Video lên", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        
        # LOGIC TỰ ĐỘNG RESET VÙNG CẤM KHI UPLOAD VIDEO KHÁC
        if 'current_video_name' not in st.session_state or st.session_state['current_video_name'] != video_file.name:
            st.session_state['current_video_name'] = video_file.name
            if 'saved_polygon' in st.session_state:
                del st.session_state['saved_polygon']
            for key in list(st.session_state.keys()):
                if key.startswith('canvas'):
                    del st.session_state[key]

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
        tfile.close()
        
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if ret:
            MAX_WIDTH = 800
            h, w = first_frame.shape[:2]
            if w > MAX_WIDTH:
                ratio = MAX_WIDTH / w
                first_frame = cv2.resize(first_frame, (MAX_WIDTH, int(h * ratio)))
                h, w = first_frame.shape[:2]
            
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            st.markdown("### 1. Vẽ Vùng cấm (Kéo Thả Hình Chữ Nhật)")
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)", stroke_width=2, stroke_color="#FF0000",
                background_image=Image.fromarray(first_frame_rgb),
                update_streamlit=True, height=h, width=w, drawing_mode="rect", key="canvas",
            )
            
            if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    obj = objects[-1]
                    if obj["type"] == "rect":
                        scale_x = obj.get("scaleX", 1)
                        scale_y = obj.get("scaleY", 1)
                        left, top = int(obj["left"]), int(obj["top"])
                        rect_w, rect_h = int(obj["width"] * scale_x), int(obj["height"] * scale_y)
                        points = [[left, top], [left + rect_w, top], [left + rect_w, top + rect_h], [left, top + rect_h]]
                        
                        if st.button("✅ Xác nhận Tọa độ Vùng cấm", type="primary"):
                            st.session_state['saved_polygon'] = np.array(points, np.int32)
                            st.success("Đã khóa Vùng cấm thành công!")
            
            if 'saved_polygon' in st.session_state:
                st.markdown("---")
                if st.button("🚀 BẮT ĐẦU CHẠY HỆ THỐNG", type="secondary"):
                    cap = cv2.VideoCapture(video_path)
                    stframe = st.empty()
                    zone_polygon = st.session_state['saved_polygon']
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame = cv2.resize(frame, (w, h))
                        
                        # CHẠY NHẬN DIỆN THẦN THÁNH BẢN GỐC
                        results = model(frame, conf=conf_threshold, iou=iou_threshold)
                        annotated_frame = results[0].plot()
                        
                        # Vẽ khung vùng cấm
                        cv2.polylines(annotated_frame, [zone_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
                        
                        intrusion_flag = False
                        ppe_violation_flag = False
                        
                        person_count = 0
                        helmet_count = 0
                        goggles_count = 0
                        missing_items_explicit = []
                        
                        # VÒNG LẶP CÀN QUÉT VẬT THỂ
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # TÌM KIẾM XÂM NHẬP 
                            cx = int((x1 + x2) / 2) 
                            cy = int((y1 + y2) / 2)
                            foot_y = y2           
                            
                            is_center_in = cv2.pointPolygonTest(zone_polygon, (float(cx), float(cy)), False)
                            is_foot_in = cv2.pointPolygonTest(zone_polygon, (float(cx), float(foot_y)), False)
                            
                            if is_center_in >= 0 or is_foot_in >= 0:
                                intrusion_flag = True
                                cv2.circle(annotated_frame, (cx, cy), 10, (0, 0, 255), -1)
                                cv2.putText(annotated_frame, "!!! INTRUSION !!!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                            # LOGIC 7 NHÃN DATASET
                            if cls_id == 6: person_count += 1
                            elif cls_id == 2: helmet_count += 1
                            elif cls_id == 1: goggles_count += 1
                            elif cls_id == 5: missing_items_explicit.append("mũ")
                            elif cls_id == 4: missing_items_explicit.append("kính")
                            elif cls_id == 3: missing_items_explicit.append("găng tay")

                        # HIỆU ỨNG KÍNH ĐỎ X-RAY KHI XÂM NHẬP
                        if intrusion_flag:
                            overlay = annotated_frame.copy()
                            cv2.fillPoly(overlay, [zone_polygon], (0, 0, 255))
                            annotated_frame = cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0)
                            
                        # TỔNG HỢP LỖI THIẾU PPE
                        current_missing = set(missing_items_explicit)
                        if person_count > helmet_count: current_missing.add("mũ")
                        if person_count > goggles_count: current_missing.add("kính")
                        
                        if len(current_missing) > 0 and (person_count > 0 or len(missing_items_explicit) > 0):
                            ppe_violation_flag = True

                        # GỬI CẢNH BÁO MQTT & TELEGRAM
                        current_time = time.time()
                        if (intrusion_flag or ppe_violation_flag) and (current_time - st.session_state.last_alert_time > ALERT_COOLDOWN):
                            if intrusion_flag:
                                send_telegram_msg(f"🚨 BÁO ĐỘNG ĐỎ: Phát hiện xâm nhập lúc {time.strftime('%H:%M:%S')}!")
                                send_mqtt_msg("Báo động đỏ. Có người xâm nhập khu vực cấm.")
                            elif ppe_violation_flag:
                                items_str = " và ".join(current_missing)
                                send_telegram_msg(f"⚠️ CẢNH BÁO AN TOÀN: Công nhân thiếu {items_str} lúc {time.strftime('%H:%M:%S')}!")
                                send_mqtt_msg(f"Nhắc nhở. Yêu cầu trang bị đầy đủ {items_str} bảo hộ.")
                                
                            st.session_state.last_alert_time = current_time 
                        
                        # Show frame ra màn hình
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        stframe.image(annotated_frame, channels="RGB", use_column_width=True)
                    
                    cap.release()
                    try: os.unlink(video_path)
                    except: pass
                    st.success("✅ Chạy Video Hoàn tất!")
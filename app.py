import streamlit as st
import torch
from PIL import Image

# Muat model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

st.title('YoloV5 Object Detection')

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Mengkonversi uploaded_file ke format PIL.Image
    im = Image.open(uploaded_file).convert("RGB")

    # Deteksi objek dalam gambar
    results = model(im)

    # Gunakan method render untuk mendapatkan gambar dengan bounding boxes
    img_with_boxes = Image.fromarray(results.render()[0])

    # Menampilkan gambar dengan bounding boxes
    st.image(img_with_boxes, caption="Detected Image.", use_column_width=True)

    # Tampilkan daftar objek yang terdeteksi
    st.subheader("Detected Objects:")
    for item in results.pred[0]:
        confidence = item[4]
        class_id = int(item[5])
        class_name = model.names[class_id]
        st.write(f"Deteksi {class_name} dengan confidence {confidence:.2f}")

if __name__ == '__main__':
    st.sidebar.info('This app is created using YoloV5 and Streamlit')
    st.sidebar.info('For any queries/suggestions please reach out at [your_email@domain.com]')

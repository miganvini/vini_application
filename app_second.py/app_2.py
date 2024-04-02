import cv2 
from ultralytics import YOLO
import streamlit as st 


model_path="weights/yolov8n.pt"
st.set_page_config (
    page_title='Object détection using YOLOV8',
    page_icon = '😉',
    layout = "wide",
    initial_sidebar_state = 'expanded'
)
with st.sidebar:
    st.header('Video/webcam Config')
    uploaded_file = st.sidebar.file_uploader("choose the video file", type= ["mp4"]) # ficgier et type
    source_vid = st.sidebar._selectbox (
        "Or select webcam", ["Webcam"]
    )
    # Mise en place du model
    model = YOLO(model_path)

    if uploaded_file is not None:
        source_vid = uploaded_file.name

    elif source_vid == "webcam" :
        source_vid = 0

    if source_vid is not None: 
        vid_cap = cv2.VideoCapture(str(source_vid))
        # Si detectObject n'est pas lancer la fonction ne s'execute pas 
        if st.sidebar.button('Detect objects') :
            st_frame = st.empty()
            while vid_cap.isOpened() :
                sucess, image = vid_cap.read()
                if sucess:
                    image = cv2.resize( 720, 405) # 720*9/16 ,16 la largeur de notre page et 9 la hauteur 
                    # L'image aura une forme rectangulaire et non carrée (720, 720)

                    # Une prédiction de l'image obtenue
                    res = model.predict(image)

                    resul_tensor = res[0].boxes

                    result_ploted = res[0].plot()

                    st.frame.image(result_ploted,caption = "Dectection video ", channels= "BGR" ,use_column_vidth= True )


                else:
                    vid_cap.release()
                    break


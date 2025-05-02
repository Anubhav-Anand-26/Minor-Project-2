import streamlit as st
from ultralytics import YOLO
import cv2
import os
import tempfile
from PIL import Image
import pandas as pd
import plotly.express as px
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Custom metric card styling function
def style_metric_cards(
    border_size_px: int = 1,
    border_color: str = "#CCC",
    border_radius_px: int = 5,
    background_color: str = "#FFF",
    box_shadow: bool = True,
):
    """
    Applies styling to metric cards
    """
    box_shadow_str = (
        "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
        if box_shadow
        else "box-shadow: none !important;"
    )
    st.markdown(
        f"""
        <style>
            div[data-testid="metric-container"] {{
                border: {border_size_px}px solid {border_color};
                padding: 5% 5% 5% 10%;
                border-radius: {border_radius_px}px;
                background-color: {background_color};
                {box_shadow_str}
            }}
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {{
                justify-content: center;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# App Config
st.set_page_config(
    page_title="🚔 Traffic Violation Detector",
    layout="wide",
    page_icon="🚔",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(162deg, #1a2980 0%, #26d0ce 100%) !important;
    }
    .st-b7, .st-c0, .st-cj, .st-ck, .st-cl, .st-cm {
        color: #000000 !important;
    }
    .css-1aumxhk {
        background-color: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .css-1v0mbdj {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff4d4d 0%, #f9cb28 100%);
    }
    .stRadio > div {
        flex-direction: row;
        gap: 15px;
    }
    .stRadio > div > label {
        background: rgba(255,255,255,0.9);
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s;
        color: #000000 !important;
    }
    .stRadio > div > label:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stRadio > div > label > div {
        color: #000000 !important;
    }
    .stFileUploader > div > div > div > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #000000 !important;
    }
    .stMarkdown p {
        color: #000000 !important;
    }
    /* Fix for file uploader text color */
    .stFileUploader label {
        color: #000000 !important;
    }
    .stFileUploader section {
        color: #000000 !important;
    }
    .stFileUploader section div {
        color: #000000 !important;
    }
    /* Fix for radio button text */
    .stRadio label div {
        color: #000000 !important;
    }
    /* Fix for upload instructions */
    .uploadedFileName {
        color: #000000 !important;
    }
    .fileUploaderInstructions {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 30px;">
    <h1 style="color: white; text-align: center; margin: 0;">🚔 AI Traffic Violation Detection System</h1>
    <p style="color: white; text-align: center; margin: 5px 0 0 0;">Upload images/videos to detect vehicles (Ambulance, Bus, Car, Motorcycle, Truck)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Settings
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: white; margin-bottom: 5px;">⚙️ Settings</h2>
        <div style="height: 2px; background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.7) 50%, rgba(255,255,255,0) 100%); margin-bottom: 20px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, help="Adjust the minimum confidence level for detection")
    model_select = st.selectbox("Model", ["YOLOv8n", "YOLOv8s", "Custom (best.pt)"], index=2, help="Select the model for detection")
    
    st.markdown("""
    <div style="height: 2px; background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.7) 50%, rgba(255,255,255,0) 100%); margin: 20px 0;"></div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
        <h4 style="color: white; margin-bottom: 10px;">🔍 Classes Detected</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            <span style="background: rgba(255,99,71,0.7); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">Ambulance</span>
            <span style="background: rgba(65,105,225,0.7); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">Bus</span>
            <span style="background: rgba(34,139,34,0.7); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">Car</span>
            <span style="background: rgba(218,165,32,0.7); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">Motorcycle</span>
            <span style="background: rgba(147,112,219,0.7); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">Truck</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model(model_name):
    if model_name == "Custom (best.pt)":
        model_path = "models/best.pt"
    else:
        model_size = model_name[-1].lower()  # 'n', 's', 'm', etc.
        model_path = f"yolov8{model_size}.pt"
    return YOLO(model_path)

model = load_model(model_select)

# File Upload Section
st.markdown("""
<div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
    <h3 style="margin-top: 0; color: #000000;">📤 Upload Media</h3>
""", unsafe_allow_html=True)

upload_type = st.radio("Input Type:", ["Image", "Video"], horizontal=True, label_visibility="collapsed")
uploaded_file = st.file_uploader(f"Upload {upload_type}", type=["jpg", "jpeg", "png", "mp4", "mov"], 
                               help="Upload an image or video for traffic violation detection")

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    if upload_type == "Image":
        # Display Image
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📤 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        if st.button("🚀 Detect Violations", key="detect_img"):
            with st.spinner("🔍 Detecting vehicles..."):
                # Save temp file
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                image.save(temp_path)
                
                # Run YOLO detection
                results = model.predict(temp_path, conf=confidence, save=True)
                
                # Display Results
                st.success("✅ Detection Complete!")
                
                with col2:
                    st.markdown("### 🎯 Detected Objects")
                    res_img = results[0].plot()[:, :, ::-1]  # Convert BGR to RGB
                    st.image(res_img, use_column_width=True)
                
                # Show Detection Data
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.markdown("""
                    <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px;">
                        <h3 style="margin-top: 0; color: #000000;">📊 Detection Metrics</h3>
                    """, unsafe_allow_html=True)
                    
                    # Metrics Cards
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Detections", len(boxes))
                    with col2:
                        unique_classes = len(set([model.names[int(cls)] for cls in boxes.cls]))
                        st.metric("Unique Classes", unique_classes)
                    with col3:
                        avg_conf = sum([float(conf) for conf in boxes.conf])/len(boxes)
                        st.metric("Avg Confidence", f"{avg_conf:.2f}")
                    
                    style_metric_cards()
                    
                    # Dataframe
                    data = {
                        "Class": [model.names[int(cls)] for cls in boxes.cls],
                        "Confidence": [f"{float(conf):.2f}" for conf in boxes.conf],
                        "Box Coordinates": [str(box.xyxy[0].tolist()) for box in boxes]
                    }
                    df = pd.DataFrame(data)
                    
                    st.dataframe(df.style.background_gradient(cmap='Blues'), 
                                use_container_width=True)
                    
                    # Visualizations
                    fig_col1, fig_col2 = st.columns(2)
                    with fig_col1:
                        fig = px.pie(df, names="Class", title="🚗 Vehicle Distribution", 
                                    color_discrete_sequence=px.colors.sequential.RdBu)
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with fig_col2:
                        fig2 = px.bar(df, x="Class", y="Confidence", color="Class",
                                     title="📈 Confidence by Vehicle Type",
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    else:  # Video Processing
        st.markdown("### 📹 Original Video")
        st.video(uploaded_file)
        
        if st.button("🚀 Detect Violations", key="detect_vid"):
            with st.spinner("Processing video..."):
                # Save temp file
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process video
                cap = cv2.VideoCapture(temp_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Output video
                output_path = os.path.join(temp_dir, "output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                # Progress bar with custom styling
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.markdown("""
                <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px;">
                    <p style="margin: 0; font-weight: bold; color: #000000;">⏳ Processing frames...</p>
                </div>
                """, unsafe_allow_html=True)
                
                detection_data = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detect objects
                    results = model.predict(frame, conf=confidence, verbose=False)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                    
                    # Collect detection data
                    boxes = results[0].boxes
                    for box in boxes:
                        detection_data.append({
                            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                            "class": model.names[int(box.cls)],
                            "confidence": float(box.conf)
                        })
                    
                    # Update progress
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress = current_frame / total_frames
                    progress_bar.progress(progress)
                    status_text.markdown(f"""
                    <div style="background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px;">
                        <p style="margin: 0; font-weight: bold; color: #000000;">⏳ Processing frame {current_frame}/{total_frames} ({progress:.1%})</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                cap.release()
                out.release()
                
                # Show output
                st.success("✅ Video Processing Complete!")
                st.markdown("### 🎯 Processed Video with Detections")
                st.video(output_path)
                
                # Show detection statistics if we have data
                if detection_data:
                    st.markdown("""
                    <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px;">
                        <h3 style="margin-top: 0; color: #000000;">📊 Video Detection Statistics</h3>
                    """, unsafe_allow_html=True)
                    
                    df = pd.DataFrame(detection_data)
                    
                    # Metrics
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("Total Detections", len(df))
                    with mcol2:
                        st.metric("Unique Classes", df['class'].nunique())
                    with mcol3:
                        st.metric("Avg Confidence", f"{df['confidence'].mean():.2f}")
                    
                    style_metric_cards()
                    
                    # Visualizations
                    fig_col1, fig_col2 = st.columns(2)
                    with fig_col1:
                        fig = px.histogram(df, x="class", color="class", 
                                         title="🚦 Detections by Vehicle Type",
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with fig_col2:
                        fig2 = px.box(df, x="class", y="confidence", color="class",
                                    title="📦 Confidence Distribution by Class",
                                    color_discrete_sequence=px.colors.qualitative.Set3)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Timeline visualization
                    st.markdown("### 📅 Detection Timeline")
                    timeline_df = df.groupby(['frame', 'class']).size().unstack().fillna(0)
                    fig3 = px.line(timeline_df, title="⏱️ Detections Over Time",
                                 labels={"value": "Detections", "frame": "Frame Number"},
                                 color_discrete_sequence=px.colors.sequential.Plasma)
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
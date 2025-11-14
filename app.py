import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import time
from collections import Counter, deque
import pandas as pd
import numpy as np
import os

# -----------------------------------
# Streamlit Page Config
# -----------------------------------
st.set_page_config(
    page_title="YOLOv8 Vehicle Counter", 
    layout="wide",
    page_icon="🚗"
)

# -----------------------------------
# Global Variables for Tracking
# -----------------------------------
class VehicleCounter:
    def __init__(self):
        self.data_deque = {}
        self.entering_count = Counter()
        self.leaving_count = Counter()
        self.unique_ids = set()
        self.line_coords = None
        
    def reset(self):
        self.data_deque = {}
        self.entering_count = Counter()
        self.leaving_count = Counter()
        self.unique_ids = set()

# -----------------------------------
# Helper Functions
# -----------------------------------
def get_color_for_class(class_name):
    """Assign colors to vehicle classes"""
    colors = {
        'bus': (255, 100, 50),
        'truck': (50, 200, 255),
        'bike': (255, 255, 50),
        'car': (100, 255, 100),
        'auto': (255, 50, 255)
    }
    return colors.get(class_name.lower(), (255, 255, 255))

def ccw(A, B, C):
    """Check counter-clockwise orientation"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    """Check if line segments AB and CD intersect"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def get_direction(point1, point2, line_orientation):
    """Determine movement direction based on line orientation"""
    if line_orientation == "Horizontal":
        if point1[1] > point2[1]:
            return "South"
        elif point1[1] < point2[1]:
            return "North"
    else:
        if point1[0] > point2[0]:
            return "East"
        elif point1[0] < point2[0]:
            return "West"
    return ""

def draw_counting_line(frame, line_coords, color=(46, 162, 112), thickness=3):
    """Draw the virtual counting line"""
    cv2.line(frame, line_coords[0], line_coords[1], color, thickness)
    cv2.circle(frame, line_coords[0], 6, (0, 255, 0), -1)
    cv2.circle(frame, line_coords[1], 6, (0, 255, 0), -1)
    return frame

def process_tracking(frame, results, model, counter, line_coords, line_orientation, show_trails=True):
    """Process tracking results and count line crossings"""
    h, w = frame.shape[:2]
    
    frame = draw_counting_line(frame, line_coords)
    
    if len(results[0].boxes) == 0:
        return frame, counter
    
    boxes = results[0].boxes
    
    if hasattr(boxes, 'id') and boxes.id is not None:
        current_ids = boxes.id.cpu().numpy().astype(int).tolist()
        lost_ids = set(counter.data_deque.keys()) - set(current_ids)
        for lost_id in lost_ids:
            counter.data_deque.pop(lost_id, None)
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        
        if hasattr(boxes, 'id') and boxes.id is not None:
            track_id = int(boxes.id[i])
        else:
            track_id = i
        
        counter.unique_ids.add(track_id)
        center = (int((x1 + x2) / 2), int(y2))
        
        if track_id not in counter.data_deque:
            counter.data_deque[track_id] = deque(maxlen=64)
        
        counter.data_deque[track_id].appendleft(center)
        
        if len(counter.data_deque[track_id]) >= 2:
            direction = get_direction(counter.data_deque[track_id][0], 
                                     counter.data_deque[track_id][1],
                                     line_orientation)
            
            if intersect(counter.data_deque[track_id][0], 
                        counter.data_deque[track_id][1],
                        line_coords[0], line_coords[1]):
                
                cv2.line(frame, line_coords[0], line_coords[1], (255, 255, 255), 5)
                
                if line_orientation == "Horizontal":
                    if "South" in direction:
                        counter.leaving_count[cls_name] += 1
                    elif "North" in direction:
                        counter.entering_count[cls_name] += 1
                else:
                    if "East" in direction:
                        counter.leaving_count[cls_name] += 1
                    elif "West" in direction:
                        counter.entering_count[cls_name] += 1
        
        color = get_color_for_class(cls_name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID:{track_id} {cls_name} {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        if show_trails and len(counter.data_deque[track_id]) > 1:
            for j in range(1, len(counter.data_deque[track_id])):
                if counter.data_deque[track_id][j - 1] is None or counter.data_deque[track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 1.5)
                cv2.line(frame, counter.data_deque[track_id][j - 1], 
                        counter.data_deque[track_id][j], color, thickness)
    
    return frame, counter

def draw_count_overlay(frame, counter, fps=0):
    """Draw count information overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (w - 280, 10), (w - 10, 140), (0, 0, 0), -1)
    cv2.putText(overlay, "ENTERING", (w - 270, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    y_offset = 60
    for idx, (cls, count) in enumerate(counter.entering_count.items()):
        text = f"{cls}: {count}"
        cv2.putText(overlay, text, (w - 270, y_offset + idx * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.rectangle(overlay, (10, 10), (280, 140), (0, 0, 0), -1)
    cv2.putText(overlay, "LEAVING", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
    
    y_offset = 60
    for idx, (cls, count) in enumerate(counter.leaving_count.items()):
        text = f"{cls}: {count}"
        cv2.putText(overlay, text, (20, y_offset + idx * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if fps > 0:
        cv2.rectangle(overlay, (10, h - 50), (280, h - 10), (0, 0, 0), -1)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (20, h - 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay, f"Unique: {len(counter.unique_ids)}", (150, h - 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    return frame

# -----------------------------------
# Sidebar Navigation
# -----------------------------------
st.sidebar.title("🚗 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🎯 Tracking & Counting", "📊 YOLO Models Info", "👨‍💻 About Me"]
)

st.sidebar.markdown("---")

# -----------------------------------
# HOME PAGE
# -----------------------------------
if page == "🏠 Home":
    st.markdown("""
    <h1 style='text-align:center; color: #1E88E5;'>🚗 YOLOv8 Vehicle Tracking & Counting System</h1>
    <h3 style='text-align:center; color: #666;'>Computer Vision and DeepSort for Traffic Analysis</h3>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ###  Welcome to the Vehicle Counting Application!
    
    This powerful application uses state-of-the-art **YOLOv8** object detection and tracking to count vehicles 
    crossing a virtual line in videos or live webcam feeds. Perfect for traffic analysis, surveillance, and 
    transportation management.
    """)
    
    # Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ###  Key Features
        
        -  **Video & Webcam Support** - Process recorded videos or live streams
        -  **Real-time Tracking** - Track multiple vehicles simultaneously
        -  **Line Crossing Detection** - Count vehicles entering and leaving
        -  **Visual Trails** - See vehicle movement paths
        -  **Live Statistics** - Real-time counting and analytics
        -  **Multiple Models** - Choose from Nano, Small, Medium, or Large models
        -  **Customizable Settings** - Adjust line position, confidence, and more
        """)
    
    with col2:
        st.markdown("""
        ###  Technologies Used
        
        - **YOLOv8** - Ultralytics object detection
        - **DeepSort** - tracking and counting
        - **OpenCV** - Computer vision processing
        - **Streamlit** - Interactive web interface
        - **Python** - Core programming language
        - **Deep Learning** - Neural network models
        - **PyTorch** - Deep learning framework
                    
        """)
    
    st.markdown("---")
    
    # How to Use
    st.markdown("""
    ###  How to Use This Application
    
    #### **Step 1: Navigate to Tracking & Counting**
    Click on **" Tracking & Counting"** in the sidebar to access the main application.
    
    #### **Step 2: Select Your Model**
    Choose a YOLO model based on your needs:
    - **Nano (n)** - Fastest, best for real-time applications
    - **Small (s)** - Balanced speed and accuracy (recommended)
    - **Medium (m)** - Higher accuracy, moderate speed
    - **Large (l)** - Best accuracy, slower processing
    
    #### **Step 3: Choose Input Source**
    - **Video**: Upload an MP4, AVI, MOV, or MKV file
    - **Webcam**: Use your device's camera for live detection
    
    #### **Step 4: Configure Settings**
    - Adjust confidence threshold (default: 0.4)
    - Set line orientation (Horizontal or Vertical)
    - Position the counting line using sliders
    - Enable/disable tracking trails and FPS display
    
    #### **Step 5: Start Processing**
    - Upload your video and processing starts automatically
    - For webcam, check the "Run Webcam" box
    - Watch real-time detection and counting!
    
    #### **Step 6: View Results**
    - See entering and leaving vehicle counts
    - View breakdown by vehicle class (car, bus, truck, bike, auto)
    - Export data for further analysis
    """)
    
    st.markdown("---")
    
    # Supported Vehicles
    st.markdown("""
    ### 🚙 Supported Vehicle Types
    
    This application can detect and count the following vehicle types:
    """)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(" Car")
        
    
    with col2:
        st.markdown(" Bus")
        
    
    with col3:
        st.markdown(" Truck")
        
    
    with col4:
        st.markdown(" Bike")
        
    
    with col5:
        st.markdown(" Auto")
        

    

    

    


# -----------------------------------
# TRACKING & COUNTING PAGE
# -----------------------------------
elif page == "🎯 Tracking & Counting":
    st.markdown("""
    <h1 style='text-align:center;'>🎯 Vehicle Tracking & Counting</h1>
    <p style='text-align:center;'>Track vehicles crossing a virtual line - Count Entering & Leaving</p>
    """, unsafe_allow_html=True)
    
    # Model Selection
    st.markdown("### 🤖 Select YOLO Model")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_size = st.selectbox(
            "Choose Model Size",
            ["Nano (n)", "Small (s)", "Medium (m)", "Large (l)"],
            index=1,
            help="Select the YOLO model size: Nano (fastest), Small (balanced), Medium (accurate), Large (most accurate)"
        )
    
    with col2:
        use_custom_path = st.checkbox("Custom Path", help="Enable to use a custom model file path")
    
    model_mapping = {
        "Nano (n)": "yolo_nano.pt",
        "Small (s)": "yolo_small.pt",
        "Medium (m)": "yolo_medium.pt",
        "Large (l)": "yolo_large.pt"
    }
    
    model_path = model_mapping[model_size]
    
    if use_custom_path:
        model_path = st.text_input("Custom YOLO Model Path", value="yolo_medium.pt")
    
    st.markdown("---")
    
    # Sidebar Settings
    st.sidebar.header("⚙️ Settings")
    source_type = st.sidebar.radio("Select Source", ["Video", "Webcam"])
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    show_trails = st.sidebar.checkbox("Show Tracking Trails", True)
    show_fps = st.sidebar.checkbox("Show FPS", True)
    
    st.sidebar.subheader("📏 Counting Line Settings")
    line_orientation = st.sidebar.radio("Line Orientation", ["Horizontal", "Vertical"])
    
    if line_orientation == "Horizontal":
        line_y = st.sidebar.slider("Line Y Position (%)", 10, 90, 50, 5)
        line_x_start = st.sidebar.slider("Line Start X (%)", 0, 50, 10, 5)
        line_x_end = st.sidebar.slider("Line End X (%)", 50, 100, 90, 5)
    else:
        line_x = st.sidebar.slider("Line X Position (%)", 10, 90, 50, 5)
        line_y_start = st.sidebar.slider("Line Start Y (%)", 0, 50, 10, 5)
        line_y_end = st.sidebar.slider("Line End Y (%)", 50, 100, 90, 5)
    
    st.sidebar.subheader("🖼️ Output Display Settings")
    output_width = st.sidebar.slider("Display Width (pixels)", 400, 1200, 800, 50)
    
    # Load Model
    if not os.path.exists(model_path):
        st.error("❌ Model path not found. Please provide a valid YOLO model file.")
        st.stop()
    
    try:
        model = YOLO(model_path)
        st.success(f"✅ Model loaded: {model_path} ({model_size})")
        st.info(f"📦 Classes: {', '.join(model.names.values())}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Video Processing
    if source_type == "Video":
        uploaded_video = st.file_uploader("📹 Upload Video", type=["mp4", "mov", "avi", "mkv"])
        
        if uploaded_video:
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(video_path)
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            
            if line_orientation == "Horizontal":
                line_coords = (
                    (int(frame_width * line_x_start / 100), int(frame_height * line_y / 100)),
                    (int(frame_width * line_x_end / 100), int(frame_height * line_y / 100))
                )
            else:
                line_coords = (
                    (int(frame_width * line_x / 100), int(frame_height * line_y_start / 100)),
                    (int(frame_width * line_x / 100), int(frame_height * line_y_end / 100))
                )
            
            st.info(f"📐 Video: {frame_width}x{frame_height} | {total_frames} frames @ {fps_video:.1f} FPS | Line: {line_orientation}")
            
            counter = VehicleCounter()
            counter.line_coords = line_coords
            
            stframe = st.empty()
            progress_bar = st.progress(0)
            stats_placeholder = st.empty()
            
            frame_count = 0
            prev_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                results = model.track(frame, persist=True, conf=conf_thresh, verbose=False)
                frame, counter = process_tracking(frame, results, model, counter, 
                                                 line_coords, line_orientation, show_trails)
                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time else 0
                prev_time = curr_time
                
                if show_fps:
                    frame = draw_count_overlay(frame, counter, fps)
                else:
                    frame = draw_count_overlay(frame, counter)
                
                aspect_ratio = frame.shape[0] / frame.shape[1]
                display_height = int(output_width * aspect_ratio)
                frame_resized = cv2.resize(frame, (output_width, display_height))
                
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", width=output_width)
                
                progress_bar.progress(frame_count / total_frames)
                
                if frame_count % 10 == 0:
                    with stats_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔼 Total Entering", sum(counter.entering_count.values()))
                        with col2:
                            st.metric("🔽 Total Leaving", sum(counter.leaving_count.values()))
                        with col3:
                            st.metric("🆔 Unique Vehicles", len(counter.unique_ids))
            
            cap.release()
            progress_bar.empty()
            
            st.success("✅ Video processing completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔼 Entering Vehicles")
                if counter.entering_count:
                    df_entering = pd.DataFrame(counter.entering_count.items(), 
                                              columns=["Class", "Count"])
                    st.dataframe(df_entering, use_container_width=True)
                    st.bar_chart(df_entering.set_index("Class"))
                else:
                    st.info("No vehicles entered")
            
            with col2:
                st.subheader("🔽 Leaving Vehicles")
                if counter.leaving_count:
                    df_leaving = pd.DataFrame(counter.leaving_count.items(), 
                                             columns=["Class", "Count"])
                    st.dataframe(df_leaving, use_container_width=True)
                    st.bar_chart(df_leaving.set_index("Class"))
                else:
                    st.info("No vehicles left")
    
    # Webcam Processing
    elif source_type == "Webcam":
        st.info("📷 Starting webcam... Toggle 'Run Webcam' to start/stop.")
        
        run_webcam = st.checkbox("▶️ Run Webcam")
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            
            ret, test_frame = cap.read()
            if ret:
                frame_height, frame_width = test_frame.shape[:2]
                
                if line_orientation == "Horizontal":
                    line_coords = (
                        (int(frame_width * line_x_start / 100), int(frame_height * line_y / 100)),
                        (int(frame_width * line_x_end / 100), int(frame_height * line_y / 100))
                    )
                else:
                    line_coords = (
                        (int(frame_width * line_x / 100), int(frame_height * line_y_start / 100)),
                        (int(frame_width * line_x / 100), int(frame_height * line_y_end / 100))
                    )
            else:
                st.error("❌ Cannot access webcam")
                st.stop()
            
            counter = VehicleCounter()
            counter.line_coords = line_coords
            
            stframe = st.empty()
            stats_placeholder = st.empty()
            prev_time = time.time()
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.track(frame, persist=True, conf=conf_thresh, verbose=False)
                frame, counter = process_tracking(frame, results, model, counter, 
                                                 line_coords, line_orientation, show_trails)
                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time else 0
                prev_time = curr_time
                
                if show_fps:
                    frame = draw_count_overlay(frame, counter, fps)
                else:
                    frame = draw_count_overlay(frame, counter)
                
                aspect_ratio = frame.shape[0] / frame.shape[1]
                display_height = int(output_width * aspect_ratio)
                frame_resized = cv2.resize(frame, (output_width, display_height))
                
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", width=output_width)
                
                with stats_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔼 Entering", sum(counter.entering_count.values()))
                    with col2:
                        st.metric("🔽 Leaving", sum(counter.leaving_count.values()))
                    with col3:
                        st.metric("🆔 Unique IDs", len(counter.unique_ids))
            
            cap.release()

# -----------------------------------
# YOLO MODELS INFO PAGE
# -----------------------------------
elif page == "📊 YOLO Models Info":
    st.markdown("""
    <h1 style='text-align:center;'> YOLOv8 Models Information</h1>
    <p style='text-align:center;'>Understanding Different YOLO Model Sizes</p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ###  What is YOLO?
    
    **YOLO (You Only Look Once)** is a state-of-the-art, real-time object detection system. YOLOv8 is the latest version 
    developed by Ultralytics, offering improved accuracy and speed compared to previous versions.
    """)
    
    st.markdown("---")
    
    # Model Comparison
    st.markdown("###  Model Comparison")
    
    model_data = {
        "Model": ["YOLOv8n (Nano)", "YOLOv8s (Small)", "YOLOv8m (Medium)", "YOLOv8l (Large)"],
        "Parameters": ["3.2M", "11.2M", "25.9M", "43.7M"],
        "Speed (ms)": ["~8ms", "~12ms", "~18ms", "~25ms"],
        "Accuracy (mAP)": ["37.3%", "44.9%", "50.2%", "52.9%"],
        "Best For": [
            "Real-time applications, embedded devices",
            "Balanced performance, recommended for most use cases",
            "High accuracy requirements with moderate speed",
            "Maximum accuracy, research applications"
        ]
    }
    
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Detailed Model Info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ###  YOLOv8n (Nano)
        
        **Best for:** Real-time processing, webcam applications
        
        **Characteristics:**
        -  Fastest inference speed (~80 FPS)
        -  Smallest model size (~6 MB)
        -  Moderate accuracy
        -  Perfect for mobile and edge devices
        -  Low computational requirements
        
 
        """)
        
        st.markdown("""
        ###  YOLOv8m (Medium)
    
        **Best for:** High accuracy requirements
        
        **Characteristics:**
        -  Balance between speed and accuracy
        -  Medium model size (~50 MB)
        -  High accuracy
        -  Suitable for most desktop applications
        -  Moderate computational requirements
        

        """)
    
    with col2:
        st.markdown("""
        ###  YOLOv8s (Small)
        
        **Best for:** Most general applications (Recommended)
        
        **Characteristics:**
        -  Great balance of speed and accuracy
        -  Small model size (~22 MB)
        -  Good accuracy
        -  Works well on standard computers
        -  Reasonable computational requirements
        

        """)
        
        st.markdown("""
        ###  YOLOv8l (Large)
        
        **Best for:** Maximum accuracy, research
        
        **Characteristics:**
        -  Highest accuracy
        -  Largest model size (~87 MB)
        -  Slower inference speed (~40 FPS)
        -  Requires powerful hardware
        -  High computational requirements
        

        """)
    
    st.markdown("---")
    
    # Performance Metrics
    st.markdown("###  Performance Metrics Explained")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Parameters**
        
        Number of learnable weights in the neural network. 
        More parameters = higher capacity but slower speed.
        """)
    
    with col2:
        st.markdown("""
        **Speed (Inference Time)**
        
        Time taken to process one image. Lower is better. 
        Measured in milliseconds (ms) or FPS.
        """)
    
    with col3:
        st.markdown("""
        **mAP (Mean Average Precision)**
        
        Accuracy metric for object detection. Higher is better. 
        Ranges from 0-100%.
        """)
    
    st.markdown("---")
    
    # Choosing the Right Model
    st.markdown("###  How to Choose the Right Model?")
    
    st.info("""
    **Quick Guide:**
    
    -  **Need Speed?** → Choose **Nano (n)**
    -  **Need Balance?** → Choose **Small (s)**  Recommended
    -  **Need Accuracy?** → Choose **Medium (m)** or **Large (l)**
    -  **Mobile/Embedded?** → Choose **Nano (n)**
    -  **Powerful PC?** → Choose **Large (l)**
    """)
    
    st.markdown("---")
    
    # Technical Details

    


# -----------------------------------
# ABOUT ME PAGE
# -----------------------------------
elif page == "👨‍💻 About Me":
    st.markdown("""
    <h1 style='text-align:center;'> AHMED PASHA</h1>
    <p style='text-align:center;'>Machine Learning Engineer | Computer Vision Specialist</p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    # with col1:
    #     st.markdown("""
    #     <div style='text-align:center; padding: 20px;'>
    #         <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    #                     width: 200px; height: 200px; border-radius: 50%; 
    #                     margin: 0 auto; display: flex; align-items: center; 
    #                     justify-content: center; font-size: 80px;'>
    #             👨‍💻
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # with col2:
    #     st.markdown("""
    #     # AHMED PASHA
    #     ### Machine Learning Engineer | Computer Vision | MLOps
        
    #     Passionate about developing cutting-edge AI solutions for real-world problems. 
    #     Specializing in computer vision, object detection, and deep learning applications.
    #     """)
    
    # st.markdown("---")
    
    # Professional Summary
    st.markdown("###  Professional Summary")
    
    st.markdown("""
    I'm a dedicated **Machine Learning Engineer** with expertise in computer vision and deep learning. 
    My work focuses on developing practical AI solutions that solve real-world problems, particularly 
    in the areas of object detection, tracking, and video analytics.
    
    This Vehicle Counting System demonstrates my ability to integrate state-of-the-art deep learning 
    models (YOLOv8) with user-friendly interfaces to create production-ready applications.
    """)
    
    st.markdown("---")
    
    # Skills & Expertise
    st.markdown("###  Skills & Expertise")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ** Machine Learning**
        - Deep Learning
        - Computer Vision
        - Object Detection
        - Image Segmentation
        - Neural Networks
        - Transfer Learning
        """)
    
    with col2:
        st.markdown("""
        ** Programming & Tools**
        - Python
        - PyTorch
        - TensorFlow
        - OpenCV
        - YOLO (v5, v8, v11)
        - Streamlit
        """)
    
    with col3:
        st.markdown("""
        ** MLOps & Deployment**
        - Model Optimization
        - Docker
        - Git/GitHub
        - Cloud Deployment
        - CI/CD Pipelines
        - Model Monitoring
        """)
    
    st.markdown("---")
    
    # Projects & Applications
    
    st.markdown("""
    ####  Vehicle Counting System (Current Project)
    Real-time vehicle detection and counting system using YOLOv8 with line crossing detection. 
    Features multiple model support, customizable settings, and comprehensive analytics.
    
    **Technologies:** YOLOv8, OpenCV, Streamlit, Python
    
    **Key Features:**
    - Multi-model support (Nano to Large)
    - Real-time and video processing
    - Line crossing detection
    - Vehicle classification
    - Interactive web interface
    """)
    
    st.markdown("---")
    
    # Technical Specializations
    st.markdown("###  Technical Specializations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Computer Vision Applications:**
        - Object Detection & Tracking
        - Image Classification
        - Video Analytics
        - Real-time Processing
        - Traffic Analysis
        - Surveillance Systems
        """)
    
    with col2:
        st.markdown("""
        **YOLO Expertise:**
        - Custom Model Training
        - Model Fine-tuning
        - Performance Optimization
        - Multi-class Detection
        - Real-time Inference
        - Edge Deployment
        """)
    
    st.markdown("---")
    
    
    # Contact & Connect
    st.markdown("### 📬 Connect With Me")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
         Email
        
        ahmedpasha99999@gmail.com
        """)
    
    with col2:
        st.markdown("""
         LinkedIn
        
        www.linkedin.com/in/ahmed-pasha-221990298
        """)
    
    with col3:
        st.markdown("""
        GitHub
        
        https://github.com/ahmedpasha746666
        """)

    


# -----------------------------------
# Footer (Appears on all pages)
# -----------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding: 10px;'>
    <p style='color: #888; font-size: 14px;'>
        Developed with  by <b>AHMED</b> | Machine Learning Engineer | YOLO Specialist
    </p>
</div>
""", unsafe_allow_html=True)
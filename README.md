# 🚀 Object Detection, Tracking & Counting System

This project is a complete real-time **Object Detection + Object Tracking + Object Counting** application built using **YOLO and DeepSORT**.
The model is trained on a **custom manually annotated dataset**, delivering high accuracy across multiple object classes.


| Feature | Status |
|--------|--------|
| Multi-class Object Detection 
| DeepSORT Object Tracking 
| Unique ID-based Counting 
| Supports Image, Video & Webcam 
| Real-time Processing 


##  Demo Preview
During execution, the output displays:
-  Bounding box around detected object  
-  Class label with confidence score  
-  Tracking ID (e.g., **ID-07**)  
-  Real-time object count overlay  



##  Tech Stack
| Component | Technology |
|----------|------------|
| Object Detection | YOLO |
| Object Tracking | DeepSORT |
| Programming Language | Python |
| Annotation Tool | Manual labeling (YOLO format) |
| Libraries | OpenCV, NumPy, Pandas, Ultralytics |
| Deployment | Streamlit 

## 📂 Dataset Details
- Fully **custom dataset manually annotated**
- Supports multiple object categories
- Annotation Format:

class x_center y_center width height


##  Installation & Execution

git clone <repo-link>
cd <project-folder>
pip install -r requirements.txt
python app.py



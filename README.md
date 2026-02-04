![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![TFLite](https://img.shields.io/badge/Model-TFLite-green)
![Audio](https://img.shields.io/badge/Input-Microphone-purple)
![GUI](https://img.shields.io/badge/UI-Tkinter-red)
![ML](https://img.shields.io/badge/Type-Keyword%20Spotting-success)
![Offline](https://img.shields.io/badge/Mode-Offline-important)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

# Intrusion Detection System using YOLOv8 & OpenCV

This project implements a real-time intrusion detection system using YOLOv8 for person detection and OpenCV for video processing.

Users can draw custom regions on a video or webcam feed. When a person enters any defined region, the system:

- Triggers a Windows alert beep  
- Displays an on-screen intrusion message  
- Draws bounding boxes  
- Logs entry/exit timestamps to a CSV file  
- Saves the processed output video  

This project is intended for learning, demos, and academic use.

---

## 🚀 Features

- YOLOv8 person detection  
- Interactive region drawing using mouse  
- Real-time intrusion alerts (Windows native beep)  
- On-screen alert messages  
- CSV logging:
  - Person ID  
  - Region ID  
  - Entry time  
  - Exit time  
  - Duration  
- Output video recording  
- Works with video files or webcam  

---

## 📁 Project Structure

Intrusion_Detection/
│
├── Intrusion.py # Main script
├── yolov8n.pt # YOLOv8 weights
├── test.mp4 # Sample input video
├── README.md
├── requirements.txt
└── .gitignore

---

## 🧰 Requirements

- Python 3.9+
- Windows OS (uses winsound for alert)

Python packages:

- opencv-python  
- numpy  
- ultralytics  

---

## 🔧 Installation

### 1. Clone repository

```bash
git clone https://github.com/yourusername/Intrusion_Detection.git
cd Intrusion_Detection
```

### 2. Create conda environment (recommended)

```bash
conda create -n intrusion_env python=3.10
conda activate intrusion_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Using video file

Edit in `Intrusion.py`:

```python
video_source = "test.mp4"
```

Run:

```bash
python Intrusion.py
```

### Using webcam

Change:

```python
video_source = "0"
```

Then run:

```bash
python Intrusion.py
```

---

## 🖱️ How It Works

1. Program starts and opens video/webcam  
2. You get 30 seconds to draw up to 5 regions using the mouse  
3. After regions are set, detection begins  
4. When a person enters a region:
   - Beep alert plays  
   - Red box is drawn  
   - Alert message appears  
   - Entry/exit data is saved to CSV  
5. Press `q` in the video window to quit  

---

## 📄 Output

- `output.mp4` → processed video  
- `output.csv` → intrusion logs  

CSV columns:

```
Person | Region | Start Time | End Time | Duration | Timestamp
```

---

## ⚠️ Notes

- Person tracking uses simple IoU + center distance logic  
- This is a demo / academic project, not production-grade surveillance  
- Alert sound uses Windows native beep (no ffmpeg required)  

---

## 📌 Future Improvements

- DeepSORT tracking  
- Region labels  
- GPU acceleration  

---

## 👩‍💻 Author

**Shreya Sidabache**

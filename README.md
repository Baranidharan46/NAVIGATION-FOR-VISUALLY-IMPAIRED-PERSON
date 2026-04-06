# 🦯 YOLOv8 Navigation System for Visually Impaired

A real-time obstacle detection and navigation assistance system built using YOLOv8 and computer vision. The system detects objects and vehicles through a webcam, determines their position relative to the user, and provides clear audio direction cues to help visually impaired individuals navigate safely.

---

## 🎯 Features

- 🔍 **Real-time object detection** using YOLOv8m
- 🗣️ **Threaded Text-to-Speech** alerts — voice runs in background without freezing video
- 📐 **Dynamic zone detection** — Left / Center / Right zones adapt to any screen resolution
- 🚗 **Multi-class detection** — Person, Bicycle, Car, Motorcycle, Bus, Truck, Traffic Light, Stop Sign
- ⏱️ **Cooldown system** — prevents repeated voice alerts (4 second gap)
- 🎯 **Confidence filtering** — only detects objects with 50%+ confidence to reduce false positives
- ⚡ **Frame skipping** — YOLO runs every 3rd frame for smooth real-time performance

---

## 🗣️ Voice Alert Examples

| Situation | Audio Output |
|---|---|
| Person on the left | *"Caution! Person detected on your left. Please move to the right."* |
| Car on the right | *"Caution! Car detected on your right. Please move to the left."* |
| Bus directly ahead | *"Warning! Bus directly ahead. Please stop or slow down."* |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8m (Ultralytics) | Real-time object detection |
| OpenCV | Webcam capture and frame rendering |
| pyttsx3 | Offline text-to-speech engine |
| Threading | Non-blocking audio playback |

---

## 📁 Project Structure

```
├── Predict.py       # Main detection + navigation logic
├── main.py          # Haar Cascade based detection (alternate version)
├── yolov8m.pt       # YOLOv8 medium model weights (auto-downloaded)
└── README.md        # Project documentation
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Baranidharan46/NAVIGATION-FOR-VISUALLY-IMPAIRED-PERSON.git
cd NAVIGATION-FOR-VISUALLY-IMPAIRED-PERSON

# Install dependencies
pip install ultralytics opencv-python pyttsx3
```

---

## ▶️ How to Run

```bash
python Predict.py
```

> **Note:** First run will automatically download `yolov8m.pt` (~25MB). Make sure you have an active internet connection.

Press **`Q`** to quit the application.

---

## 🧠 How It Works

1. **Webcam captures** live video frame by frame
2. **YOLOv8m detects** objects and returns bounding box coordinates
3. **Zone classifier** divides the frame into Left | Center | Right based on object's x-position
4. **Voice alert** is triggered with direction instruction (runs in a separate thread)
5. **Cooldown system** ensures alerts are not repeated within 4 seconds
6. Loop continues in real-time until user exits

### Zone Logic
```
|── LEFT ──|── CENTER ──|── RIGHT ──|
   Turn Right   Stop!     Turn Left
```

---

## 🔍 Detectable Objects

| Class | COCO ID |
|---|---|
| Person | 0 |
| Bicycle | 1 |
| Car | 2 |
| Motorcycle | 3 |
| Bus | 5 |
| Truck | 7 |
| Traffic Light | 9 |
| Stop Sign | 11 |

---

## 🚀 Future Improvements

- [ ] GPS integration for outdoor navigation
- [ ] Distance estimation using bounding box size
- [ ] Mobile app version using Flutter
- [ ] Support for multiple languages in voice alerts
- [ ] CUDA/GPU acceleration for faster inference

---

## 👨‍💻 Author

**Baranidharan**
B.Tech — Artificial Intelligence & Machine Learning
K. Ramakrishnan College of Technology, Trichy

[![GitHub](https://img.shields.io/badge/GitHub-Baranidharan46-black?logo=github)](https://github.com/Baranidharan46)

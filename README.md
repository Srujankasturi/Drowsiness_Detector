# Real-Time Drowsiness Detection System

A computer vision system that detects driver/operator drowsiness in real-time using eye aspect ratio (EAR) analysis and head pose estimation. Built with OpenCV and MediaPipe.

## Demo

> Run the system and close your eyes for 2 seconds — the screen flashes red with a DROWSY alert.

## How It Works

The system uses two independent signals to detect drowsiness:

**1. Eye Aspect Ratio (EAR)**  
MediaPipe's 468-point face mesh tracks 6 landmarks around each eye. The EAR is computed as the ratio of vertical to horizontal eye distances. When both eyes remain below an EAR threshold of 0.25 for more than 2 seconds, a drowsiness alert fires.

**2. Head Pose / Nod Detection**  
The vertical distance between the nose tip and chin landmarks is tracked each frame. When this distance exceeds a calibrated threshold (indicating the head drooping forward), a separate nod alert triggers — catching drowsiness even when eyes are open.

Both signals log timestamped events to `alerts.csv` for post-session analysis.

## Features

- Real-time webcam-based detection (CPU only, no GPU required)
- Dual-signal detection: EAR + head nod
- False alert prevention — both eyes must be closed (winking does not trigger)
- Timestamped CSV event logging
- Live EAR and tilt values displayed on screen

## Tech Stack

- Python
- OpenCV — video capture and frame processing
- MediaPipe — 468-point face mesh landmark detection
- NumPy — EAR geometry calculations

## Installation

```bash
git clone https://github.com/Srujankasturi/Drowsiness_Detector.git
cd Drowsiness_Detector
pip install -r requirements.txt
python drowsiness_detector.py
```

## Requirements

```
opencv-python
mediapipe==0.10.33
numpy
```

## Use Case

This system is directly applicable to:
- Driver monitoring systems
- Factory operator attention tracking
- Remote worker fatigue detection

These are core use cases for AI-powered Operations Insight Platforms.

## Project Structure

```
Drowsiness_Detector/
├── drowsiness_detector.py   # main detection script
├── alerts.csv               # auto-generated event log (gitignored)
├── requirements.txt
└── README.md
```

## Author

Srujan Kasturi — [GitHub](https://github.com/Srujankasturi) 
B.Tech CSE, SRM University AP

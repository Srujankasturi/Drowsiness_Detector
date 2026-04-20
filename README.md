# Drowsiness Detection System

Detects driver or operator drowsiness in real time using eye tracking and head pose estimation. Runs on CPU — no GPU needed.

## How it works

Two signals run simultaneously. The first tracks eye closure: MediaPipe's face mesh gives 6 landmarks around each eye, and the Eye Aspect Ratio (EAR) measures how open the eye is based on vertical vs horizontal distances. If both eyes stay below an EAR of 0.25 for more than 3 seconds, the alert fires. One eye closed doesn't count — winking or looking sideways won't trigger it.

The second signal catches head nodding. It measures the vertical distance between the nose tip and chin each frame. When that distance crosses a calibrated threshold (head drooping forward), a separate alert triggers — useful for cases where someone's eyes are technically open but their head is already going down.

Both signals log to `alerts.csv` with timestamps.

**Pipeline:**
1. MediaPipe face mesh detects 468 landmarks per frame
2. EAR computed for both eyes independently
3. Both eyes must be below threshold for 3+ seconds to trigger eye alert
4. Nose-to-chin distance tracked for head pose
5. Either signal triggers the DROWSY alert and red screen flash

## Stack

- Python
- OpenCV — video capture, frame display
- MediaPipe — face mesh landmark detection
- NumPy — EAR geometry

## Setup

```bash
git clone https://github.com/Srujankasturi/Drowsiness_Detector.git
cd Drowsiness_Detector
pip install opencv-python mediapipe==0.10.33 numpy
```

```bash
python drowsiness_detector.py
```

Close your eyes for 2 seconds — screen goes red. Open them — back to normal. Check `alerts.csv` for the log.

## Tunable parameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `EAR_THRESHOLD` | 0.25 | How closed eyes need to be |
| `CLOSED_EYES_TIME` | 2s | How long before alert fires |
| `TILT_THRESHOLD` | 100 | Head nod sensitivity |

The tilt threshold was calibrated by printing raw values at upright vs. nodding positions — upright was 85–92, nodding went above 100.

## Use case

Driver monitoring, factory operator attention tracking, remote worker fatigue detection — anywhere you need to know if someone's losing focus in front of a camera.

## Author

Srujan Kasturi — [GitHub](https://github.com/Srujankasturi)  
B.Tech CSE, SRM University AP
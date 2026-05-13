# Camera-Based Anomaly Monitor

A lightweight computer vision and AI pose-estimation proof of concept for detecting unusual activity in camera or video feeds.

This project uses OpenCV for motion detection and MediaPipe for pre-trained human skeleton detection. It monitors motion patterns, scores anomalous activity, detects human pose behavior, displays live detections, and saves alert evidence for review. It is designed as a practical demo that can be shown quickly, extended later, and shared on GitHub as a clean computer vision project.

## What It Does

- Reads from a webcam, IP camera stream, or video file
- Detects significant movement using OpenCV background subtraction
- Detects human skeletons using MediaPipe pose estimation
- Draws skeletons for multiple people at once
- Recognizes known people from a local face image folder
- Scores rapid hand movement and tamper-like body motion
- Tracks detected people across frames with stable short-lived track IDs
- Scores loitering, repeated movement, restricted-zone dwell, and fast ROI motion
- Scores each frame with motion and pose-based anomaly scores
- Flags unusual activity when the score crosses a threshold
- Focuses alerts on person behavior instead of every moving object
- Draws pose skeleton landmarks over people
- Labels recognized people on screen and in alert logs
- Keeps anomalous people marked for a few seconds instead of flashing briefly
- Saves alert images and short MP4 event clips
- Supports optional restricted zones with region-of-interest monitoring
- Clips restricted zones to the visible frame so edge ROIs score correctly
- Saves alert snapshots
- Writes alert events to a JSONL audit log
- Runs locally with a simple command

## Why This Is Useful

This is a proof of concept for a camera-based anomaly detection system. It does not require training data, cloud services, or a custom model. It uses pre-trained pose estimation plus explainable scoring logic to prove the workflow:

```text
camera feed -> motion detection + pose estimation -> anomaly score -> alert -> evidence log
```

Later versions can add object detection, stronger multi-camera tracking, a custom action classifier, or a learned anomaly detection model.

## Example Use Cases

- After-hours movement
- Restricted-zone activity
- Tamper-like hand movement near a machine or restricted zone
- Smash-like rapid arm movement
- Known-person behavior review, for example `Person A` triggered `tamper_like_motion`
- Loitering-like behavior
- Unusual motion in normally quiet areas
- Crowd or activity spikes
- General surveillance anomaly review

## Project Structure

```text
camera-based-anomaly-monitor/
  src/
    anomaly_monitor/
      __init__.py
      config.py
      detector.py
      enroll.py
      events.py
      faces.py
      main.py
      menu.py
      names.py
      pose.py
      tracking.py
  tests/
    test_config.py
    test_names.py
    test_tracking.py
  data/
    alerts/
    known_faces/
    models/
  requirements.txt
  .gitignore
  README.md
```

## Setup

Create a virtual environment:

```bash
py -3.12 -m venv .venv312-run
```

Activate it on Windows:

```bash
.venv312-run\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the lightweight unit tests:

```bash
python -m unittest discover -s tests
```

## Add Known Faces

Option 1: capture face samples from your webcam:

```bash
enroll-face --name "Person A" --count 8
```

Look at the camera and press `s` to save each face sample. Press `q` when finished.

Option 2: create one folder per person inside `data/known_faces` and put a few clear face photos in each folder:

```text
data/
  known_faces/
    Person A/
      face1.jpg
      face2.jpg
    Person B/
      face1.jpg
```

Use simple demo names like `Person A`, `Person B`, or real names only if you have permission to use those photos. The recognizer trains locally when the app starts.

## Run With Webcam

For the easiest terminal workflow, use the interactive menu:

```bash
anomaly-menu
```

Menu options:

```text
1. Begin monitoring and logging
2. Begin monitoring with custom settings
3. Enroll a new face
4. Edit known faces
5. View recent alerts
6. Show project folders
0. Exit
```

You can still run the monitor directly:

```bash
python -m anomaly_monitor.main --source 0
```

Or use the installed command:

```bash
anomaly-monitor --source 0
```

## Run With a Video File

```bash
python -m anomaly_monitor.main --source path/to/video.mp4
```

## Optional Arguments

```bash
python -m anomaly_monitor.main --source 0 --threshold 0.08 --cooldown 5
```

Monitor only a restricted zone:

```bash
python -m anomaly_monitor.main --source 0 --roi 100,80,500,300
```

Make pose behavior alerts more sensitive:

```bash
python -m anomaly_monitor.main --source 0 --pose-threshold 0.5 --wrist-speed-threshold 1.0
```

Tune person tracking and motion-history behavior:

```bash
python -m anomaly_monitor.main --source 0 --loitering-seconds 8 --roi-dwell-seconds 2
```

Run with known-face recognition:

```bash
python -m anomaly_monitor.main --source 0 --known-faces-dir data/known_faces
```

Detect up to six people:

```bash
python -m anomaly_monitor.main --source 0 --max-poses 6
```

Keep anomaly labels visible longer and save longer alert clips:

```bash
python -m anomaly_monitor.main --source 0 --alert-hold-seconds 8 --event-video-seconds 6
```

Disable skeleton/pose analysis and use motion only:

```bash
python -m anomaly_monitor.main --source 0 --no-pose --motion-alerts --show-motion-boxes
```

Arguments:

- `--source`: webcam index, video file path, or RTSP/HTTP camera URL
- `--known-faces-dir`: folder containing known face images by person
- `--face-confidence-threshold`: face recognition threshold, where lower is stricter
- `--threshold`: anomaly score needed to trigger an alert
- `--pose-threshold`: pose behavior score needed to trigger an alert
- `--wrist-speed-threshold`: sensitivity for rapid wrist/hand movement
- `--loitering-seconds`: seconds a tracked person can stay near the same spot before loitering is flagged
- `--roi-dwell-seconds`: seconds a tracked person can stay inside the ROI before dwell is flagged
- `--motion-history-seconds`: seconds of per-person movement history to keep
- `--rapid-body-speed-threshold`: sensitivity for fast full-body movement
- `--repeated-motion-distance`: recent path length needed to flag repeated back-and-forth motion
- `--max-poses`: maximum number of people/skeletons to detect at once
- `--cooldown`: seconds to wait before creating another alert
- `--alert-hold-seconds`: seconds to keep a person marked after an anomaly
- `--event-video-seconds`: seconds of recent annotated frames to save as an MP4 clip
- `--event-video-fps`: FPS used for saved alert clips
- `--warmup-frames`: frames to skip before saving alerts while the baseline warms up
- `--output-dir`: folder for alert snapshots and event logs
- `--min-area`: ignore motion regions smaller than this many pixels
- `--roi`: optional restricted zone in `x,y,width,height` format
- `--pose-model`: path to the MediaPipe pose model file
- `--show-mask`: show the foreground mask window
- `--show-motion-boxes`: draw boxes around moving regions
- `--motion-alerts`: allow motion-only alerts
- `--no-pose`: turn off pose estimation
- `--no-tracking`: turn off person tracking and motion-history scoring
- `--no-face-recognition`: turn off known-face recognition

## How Detection Works

The app combines two detection paths.

First, it uses a background subtraction model to learn what the scene normally looks like. When new motion appears, it extracts moving regions, filters out tiny noise, and calculates a motion score. Motion boxes and motion-only alerts are disabled by default so the demo stays focused on people.

Second, it uses a pre-trained MediaPipe pose landmarker to draw a human skeleton and estimate body behavior. It currently looks for:

- rapid hand/wrist movement
- hands entering a restricted zone
- extended-arm posture
- combined tamper-like motion patterns

Third, it tracks pose detections across frames with a lightweight centroid tracker. Each tracked person gets a temporary track ID such as `T1`, and the app keeps recent normalized movement history. That history is used to flag:

- loitering near the same spot
- staying inside a restricted zone
- repeated back-and-forth movement
- fast full-body movement near a restricted zone

Fourth, it uses OpenCV face detection and LBPH face recognition to label people from `data/known_faces`. Alert logs include the recognized identity when available.

When a person triggers an anomaly, the app keeps that person marked as `ALERT` for a few seconds, saves a JPEG snapshot, saves a short MP4 clip, and writes the evidence paths to `data/alerts/events.jsonl`.

The motion score is based on how much of the frame changed:

```text
motion_score = moving_pixels / total_pixels
```

The final alert score uses the highest motion, pose, or tracking behavior score. This is simple, explainable, and good enough for a first proof of concept.

On first run, the project downloads the lightweight MediaPipe pose model to:

```text
data/models/pose_landmarker_lite.task
```

## Roadmap

- Add object detection with YOLO
- Add stronger person tracking with ByteTrack or DeepSORT
- Add trained action recognition for specific tampering/smashing examples
- Add a Streamlit or web dashboard
- Add alert notifications through email or Teams
- Add model-based anomaly detection for learned behavior patterns

## Notes

This project is a demo, not a production surveillance system. Human review should always be used before acting on alerts.

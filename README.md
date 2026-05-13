# OpenCV Based Anomaly Monitor

A lightweight computer vision and AI pose-estimation proof of concept for detecting unusual activity in camera or video feeds.

This project uses OpenCV for motion detection and MediaPipe for pre-trained human skeleton detection. It monitors motion patterns, scores anomalous activity, detects human pose behavior, displays live detections, and saves alert evidence for review. It is designed as a practical demo that can be shown quickly, extended later, and shared on GitHub as a clean computer vision project.

## What It Does

- Reads from a webcam, IP camera stream, or video file
- Detects significant movement using OpenCV background subtraction
- Detects human skeletons using MediaPipe pose estimation
- Draws skeletons for multiple people at once
- Recognizes known people from a local face image folder
- Gives session labels to unknown faces, for example `Unknown A`
- Remembers flagged identities so a flagged unknown person stays flagged if they return
- Defaults to a simple T-pose test alert so normal movement stays quiet
- Can track detected people across frames with stable short-lived track IDs
- Can score loitering, repeated movement, restricted-zone dwell, and fast ROI motion
- Scores each frame with motion and pose-based anomaly scores
- Flags unusual activity when the score crosses a threshold
- Focuses alerts on person behavior instead of every moving object
- Draws pose skeleton landmarks over people
- Draws a red box around flagged people
- Labels recognized people on screen and in alert logs
- Keeps anomalous people marked for a few seconds instead of flashing briefly
- Saves alert images and MP4 event clips with pre-roll and post-roll
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
    test_detector_identity_memory.py
    test_faces.py
    test_names.py
    test_pose.py
    test_tracking.py
  data/
    alerts/
    known_faces/
    models/
  start-cli.cmd
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
add-face --name "Person A" --count 8
```

Look at the camera and press `s` to save each face sample. Press `q` when finished.

The older command still works too:

```bash
enroll-face --name "Person A" --count 8
```

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

On Windows, you can also double-click `start-cli.cmd` or run:

```powershell
.\start-cli.cmd
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
python -m anomaly_monitor.main --source 0 --pose-threshold 0.8
```

Tune person tracking and motion-history behavior:

```bash
python -m anomaly_monitor.main --source 0 --tracking --loitering-seconds 20 --roi-dwell-seconds 6
```

Re-enable the older rapid-hand and tamper-style behavior rules:

```bash
python -m anomaly_monitor.main --source 0 --full-behavior --tracking
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
python -m anomaly_monitor.main --source 0 --alert-hold-seconds 8 --pre-alert-seconds 2 --post-alert-seconds 3
```

Disable skeleton/pose analysis and use motion only:

```bash
python -m anomaly_monitor.main --source 0 --no-pose --motion-alerts --show-motion-boxes
```

Arguments:

- `--source`: webcam index, video file path, or RTSP/HTTP camera URL
- `--known-faces-dir`: folder containing known face images by person
- `--face-confidence-threshold`: face recognition threshold, where lower is stricter
- `--unknown-face-match-threshold`: session unknown-face threshold, where lower is stricter
- `--identity-alert-hold-seconds`: seconds a flagged identity stays remembered
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
- `--pre-alert-seconds`: seconds before an alert to include in saved MP4 clips
- `--post-alert-seconds`: seconds after an alert to include in saved MP4 clips
- `--event-video-seconds`: deprecated alias for `--post-alert-seconds`
- `--event-video-fps`: FPS used for saved alert clips
- `--warmup-frames`: frames to skip before saving alerts while the baseline warms up
- `--output-dir`: folder for alert snapshots and event logs
- `--min-area`: ignore motion regions smaller than this many pixels
- `--roi`: optional restricted zone in `x,y,width,height` format
- `--pose-model`: path to the MediaPipe pose model file
- `--show-mask`: show the foreground mask window
- `--show-motion-boxes`: draw boxes around moving regions
- `--motion-alerts`: allow motion-only alerts
- `--full-behavior`: enable rapid-hand, ROI, and extended-arm behavior rules
- `--no-pose`: turn off pose estimation
- `--tracking`: turn on person tracking and motion-history scoring
- `--no-tracking`: force person tracking off
- `--no-face-recognition`: turn off known-face recognition

## How Detection Works

The app combines two detection paths.

First, it uses a background subtraction model to learn what the scene normally looks like. When new motion appears, it extracts moving regions, filters out tiny noise, and calculates a motion score. Motion boxes and motion-only alerts are disabled by default so the demo stays focused on people.

Second, it uses a pre-trained MediaPipe pose landmarker to draw a human skeleton and estimate body behavior. By default, the app is in a quiet test mode where only a T-pose is considered anomalous. This makes it easy to verify the alert path without normal movement causing alerts.

When `--full-behavior` is enabled, it also looks for:

- rapid hand/wrist movement
- hands entering a restricted zone
- extended-arm posture
- combined tamper-like motion patterns

Third, it can track pose detections across frames with a lightweight centroid tracker when `--tracking` is enabled. Each tracked person gets a temporary track ID such as `T1`, and the app keeps recent normalized movement history. That history is used to flag:

- loitering near the same spot
- staying inside a restricted zone
- repeated back-and-forth movement
- fast full-body movement near a restricted zone

Fourth, it uses OpenCV face detection and LBPH face recognition to label people from `data/known_faces`. If a face is not recognized, the app assigns a session label such as `Unknown A`. When an identity triggers an alert, that identity is remembered for a while, so `Unknown A` is marked again if they leave and come back during the same run.

When a person triggers an anomaly, the app keeps that person marked as `ALERT`, draws a red box around them, saves a JPEG snapshot, records the configured post-alert frames, then saves an MP4 clip. By default each clip includes about 2 seconds before the alert and 3 seconds after it. Evidence paths are written to `data/alerts/events.jsonl`.

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

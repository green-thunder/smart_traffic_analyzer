# Smart Traffic Analyzer

Real-time vehicle detection, tracking and zone-based counting from a camera feed, built with **YOLOv8**, **ByteTrack** and **Supervision**.

Built in collaboration with Tashkent State University of Railway Engineering to turn raw camera video into traffic data for transport planning.

## How it works

1. Every frame from the camera goes through a YOLO model trained on vehicles (`vehicles.pt`).
2. Only the vehicle classes listed in `selected_classes` are kept.
3. **ByteTrack** gives each vehicle a stable ID across frames.
4. You draw polygon **zones** on the road. Each vehicle ID that enters a zone is counted once, and the count is drawn in the middle of the zone.

## Project structure

| Path | What it does |
|---|---|
| `main.py` | Opens the camera, runs the per-frame callback and shows the annotated video (press `Esc` to quit) |
| `utils/callback.py` | Model, tracker, zones and counting logic |
| `example/traffic_analyser.ipynb` | Example notebook |

## Setup

```bash
pip install -r requirements.txt

# download the trained vehicle model from Google Drive
gdown '1bqFNJx4n6HVIsRwTu7kG3hbNIwFUHV9j'
```

The model is loaded from `../model/vehicles.pt`; change the path in `utils/callback.py` if you keep it elsewhere.

## Configure zones

Set your polygons in `zones_coor` in `utils/callback.py`. Coordinates are in pixels for a 1280×720 frame:

```python
zones_coor = [
    np.array([[185, 473], [213, 548], [56, 576], [39, 493]], np.int32),
    # add more polygons here
]
```

## Run

```bash
python main.py
```

`main.py` reads camera `0`. To use an IP camera, pass its stream URL to `cv2.VideoCapture` instead.

## Tech

Python · Ultralytics YOLOv8 · Supervision · ByteTrack · OpenCV · PyTorch

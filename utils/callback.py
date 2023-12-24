from ultralytics import YOLO
import numpy as np
import supervision as sv
from collections import defaultdict
from typing import List

frame_width, frame_height = 1280, 720

selected_classes = [2, 3, 5, 6, 7] 
model = YOLO("../model/vehicles.pt")
tracker = sv.ByteTrack()
colors = sv.ColorPalette.default()

counts = defaultdict(set)

zones_coor = [
    # np.array([[185,473],[213,548],[56,576],[39,493]], np.int32), #example
    np.array(['your coordinates for polygon'], np.int32)
    #your other polygones
    #...
]

zones = [
    sv.PolygonZone(
        polygon=zone_coor,
        frame_resolution_wh=(frame_width, frame_height),
        triggering_position=sv.Position.CENTER
    )
    for zone_coor in zones_coor
]


box_annotator = sv.BoxAnnotator(
        thickness=2, 
        text_thickness=1, 
        text_scale=0.5, 
        color=sv.Color.red()
    )

def callback_per_frame(frame: np.ndarray) -> np.ndarray:
        
    results = model(frame, agnostic_nms=True, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, selected_classes)]
    detections = tracker.update_with_detections(detections)

    frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
        skip_label=True,
    )

    for i, zone in enumerate(zones):
        polygon = zone.polygon
        frame = sv.draw_polygon(
        frame,
        polygon,
        colors.by_idx(i)
    )
        
    map = zone.trigger(detections=detections)

    detections_in_zone: List[sv.Detections] = detections[map]

    for tracker_id in detections_in_zone.tracker_id:
        counts[i].add(tracker_id)
    count = len(counts[i])

    zone_center = sv.get_polygon_center(polygon=polygon)
    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y)

    frame = sv.draw_text(
        frame,
        str(count),
        text_anchor,
        background_color=colors.by_idx(i)
    )
    
    return frame

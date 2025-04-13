import numpy as np
import cv2
import json
from ultralytics import YOLO
from collections import defaultdict
import torch
from boxmot.trackers.strongsort.strongsort import StrongSort
from pathlib import Path
import matplotlib.pyplot as plt
import random
import supervision as sv
import time
# from . import draw_all

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * (yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def clip_bbox_change(curr_box, last_box, scale_limit=0.5):
    w1, h1 = curr_box[2] - curr_box[0], curr_box[3] - curr_box[1]
    w2, h2 = last_box[2] - last_box[0], last_box[3] - last_box[1]
    if abs(w1 - w2) / (w2 + 1e-5) > scale_limit or abs(h1 - h2) / (h2 + 1e-5) > scale_limit:
        return last_box
    return curr_box

def is_jump(curr_box, last_box, threshold=50):
    cx1 = (curr_box[0] + curr_box[2]) / 2
    cy1 = (curr_box[1] + curr_box[3]) / 2
    cx2 = (last_box[0] + last_box[2]) / 2
    cy2 = (last_box[1] + last_box[3]) / 2
    return np.linalg.norm([cx1 - cx2, cy1 - cy2]) > threshold

def crossed_line(p1, p2, line_start, line_end):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end)) and \
           (ccw(p1, p2, line_start) != ccw(p1, p2, line_end))

def point_in_polygon(point, polygon):
    point = (float(point[0]), float(point[1]))
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def get_color_for_id(track_id):
    random.seed(track_id)
    return tuple(random.randint(100, 255) for _ in range(3))
def load_config(video_path, mask_path, annotation_path, model_path):
    cap = cv2.VideoCapture(video_path)
    mask = cv2.imread(mask_path)
    
    with open(annotation_path, "r") as f:
        regions = json.load(f)

    lines = [tuple(map(tuple, line)) for line in regions.get("lines", [])]
    polygons = [np.array(poly, dtype=np.int32) for poly in regions.get("polygons", [])]

    model = YOLO(model_path)
    model.conf = 0.4
    model.iou = 0.45

    return cap, mask, lines, polygons, model


def initialize_tracker(weights_path="model/osnet_x1_0_market1501.pt", max_cosine_distance=0.2):
    return StrongSort(
        reid_weights=Path(weights_path),
        device='0' if torch.cuda.is_available() else 'cpu',
        max_cos_dist=max_cosine_distance,
        half=False,
        max_age=30
    )


def process_frame(frame, model, tracker, last_boxes, bbox_buffer, bbox_history,
                  last_centers, track_colors, counted_ids,
                  lines, polygons, line_counts, polygon_counts,
                  box_annotator, line_zone,
                  buffer_size=5, max_jump_distance=100, scale_limit=0.5,
                  enable_debug=False, frame_count=0):

    detections = sv.Detections.from_ultralytics(model(frame, verbose=False)[0])
    detections = detections[detections.class_id == 0]
    
    if len(detections) == 0:
        return frame

    detection_array = np.hstack((
        detections.xyxy,
        detections.confidence.reshape(-1, 1),
        detections.class_id.reshape(-1, 1)
    ))

    tracks = tracker.update(detection_array, frame)
    if len(tracks) == 0:
        return frame

    annotated_detections = sv.Detections(
        xyxy=tracks[:, :4],
        class_id=tracks[:, -1].astype(int),
        tracker_id=tracks[:, 4].astype(int)
    )

    crossed_in, crossed_out = line_zone.trigger(annotated_detections)

    labels = [f"ID: {track_id}" for track_id in annotated_detections.tracker_id]
    frame = box_annotator.annotate(scene=frame, detections=annotated_detections)

    all_anchors = np.array([
        annotated_detections.get_anchors_coordinates(anchor)
        for anchor in line_zone.triggering_anchors
    ])

    for anchors_per_detection in all_anchors.transpose(1, 0, 2):
        for x, y in anchors_per_detection:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)

    for bbox, track_id in zip(annotated_detections.xyxy, annotated_detections.tracker_id):
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bbox_arr = np.array([x1, y1, x2, y2])

        if track_id in last_boxes:
            last_box = last_boxes[track_id]
            if is_jump(bbox_arr, last_box, max_jump_distance):
                bbox_arr = last_box
            else:
                bbox_arr = clip_bbox_change(bbox_arr, last_box, scale_limit)
            bbox_arr = 0.6 * bbox_arr + 0.4 * last_box

        bbox_buffer[track_id].append(bbox_arr)
        if len(bbox_buffer[track_id]) > buffer_size:
            bbox_buffer[track_id].pop(0)
        bbox_arr = np.mean(bbox_buffer[track_id], axis=0).astype(int)
        last_boxes[track_id] = bbox_arr
        track_colors.setdefault(track_id, get_color_for_id(track_id))

        prev_center = last_centers.get(track_id)
        curr_center = ((bbox_arr[0] + bbox_arr[2]) // 2, (bbox_arr[1] + bbox_arr[3]) // 2)

        if prev_center and track_id not in counted_ids:
            for i, (line_start, line_end) in enumerate(lines):
                if crossed_line(prev_center, curr_center, line_start, line_end):
                    line_counts[i] += 1
                    counted_ids.add(track_id)
                    break
            else:
                for j, poly in enumerate(polygons):
                    if point_in_polygon(curr_center, poly):
                        polygon_counts[j] += 1
                        counted_ids.add(track_id)
                        break

        last_centers[track_id] = curr_center
        color = track_colors[track_id]
        cv2.rectangle(frame, (bbox_arr[0], bbox_arr[1]), (bbox_arr[2], bbox_arr[3]), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (bbox_arr[0], max(0, bbox_arr[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, curr_center, 4, color, -1)

        if enable_debug:
            area = (bbox_arr[2] - bbox_arr[0]) * (bbox_arr[3] - bbox_arr[1])
            bbox_history[track_id].append((frame_count, area))

    return frame


def draw_overlays(frame, lines, polygons, line_counts, polygon_counts, line_zone):
    for i, poly in enumerate(polygons):
        cv2.polylines(frame, [poly], isClosed=True, color=(255, 0, 0), thickness=2)
        M = cv2.moments(poly)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, f"Poly {i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for i, (start, end) in enumerate(lines):
        cv2.line(frame, start, end, (0, 255, 0), 2)
        mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(frame, f"Line {i}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    y_offset = 30
    for i in range(len(lines)):
        cv2.putText(frame, f"Line {i} in Count : {line_zone.in_count}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame, f"Line {i} OUT Count: {line_zone.out_count}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    for j, count in polygon_counts.items():
        cv2.putText(frame, f"Poly {j} Count : {count}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_offset += 30

    return frame


def plot_debug_metrics(bbox_history):
    plt.figure(figsize=(12, 6))
    for track_id, area_list in bbox_history.items():
        if len(area_list) >= 2:
            frames, areas = zip(*area_list)
            plt.plot(frames, areas, label=f'ID {track_id}')
    plt.xlabel("Frame")
    plt.ylabel("Bounding Box Area")
    plt.title("Bounding Box Area Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    lifespans = {track_id: len(areas) for track_id, areas in bbox_history.items()}
    plt.figure(figsize=(10, 5))
    plt.bar(lifespans.keys(), lifespans.values(), color='skyblue')
    plt.xlabel("Track ID")
    plt.ylabel("Frames Tracked")
    plt.title("Track ID Lifespan")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_people_counter(video_path, mask_path, annotation_path, model_path, output_path, enable_debug=True):
    cap, mask, lines, polygons, model = load_config(video_path, mask_path, annotation_path, model_path)
    tracker = initialize_tracker()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    line_counts = {i: 0 for i in range(len(lines))}
    polygon_counts = {i: 0 for i in range(len(polygons))}
    last_boxes, last_centers, track_colors = {}, {}, {}
    bbox_buffer, bbox_history = defaultdict(list), defaultdict(list)
    counted_ids = set()
    box_annotator = sv.BoxAnnotator(thickness=2)
    line_zone = sv.LineZone(start=sv.Point(*lines[0][0]), end=sv.Point(*lines[0][1]))

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        frame = process_frame(
            frame, model, tracker, last_boxes, bbox_buffer, bbox_history,
            last_centers, track_colors, counted_ids, lines, polygons,
            line_counts, polygon_counts, box_annotator, line_zone,
            enable_debug=enable_debug, frame_count=frame_count
        )
        frame = draw_overlays(frame, lines, polygons, line_counts, polygon_counts, line_zone)
        out.write(frame)
        cv2.imshow("People Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if enable_debug:
        plot_debug_metrics(bbox_history)
# run_people_counter(
#     video_path="C:/Work/CarbonZAP/visitor_count/people-counting/shopping.mp4",
#     mask_path="C:/Work/CarbonZAP/visitor_count/people-counting/images/mask_ppl.png",
#     annotation_path="C:/Work/CarbonZAP/visitor_count/people-counting/images/annotations.json",
#     model_path="yolov8l.pt",
#     output_path="output_video.mp4",
#     enable_debug=True
# )

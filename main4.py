import logging
import numpy as np
import cv2
import time
import supervision as sv
import mediapipe as mp
from deepface import DeepFace
import os
from collections import defaultdict

from people_counting.people_counting8 import *
from face_recognition.face_rec2 import (is_folder_empty, detect_faces, save_unknown_face)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

"""General Configurations"""
video_path = r"C:\Work\CarbonZAP\visitor_count\cctv.mp4"
mask_path = r"C:\Work\CarbonZAP\visitor_count\people_counting\images\mask_ppl.png"
annotation_path = r"C:\Work\CarbonZAP\visitor_count\people_counting\images\annotations.json"
model_path = r"C:\Work\CarbonZAP\visitor_count\yolov8l.pt"
output_path = "output_video.mp4"
save_interval = 2
recognition_interval = 3
known_faces_folder = r"C:\Work\CarbonZAP\visitor_count\face_recognition\dataset\known"
max_jump_distance = 200
scale_limit = 0.3
buffer_size = 5
enable_debug = True

"""Initialize Tracker, Model, and Config"""
logging.info("Initializing tracker and model...")
tracker = initialize_tracker()
cap, mask, lines, polygons, model = load_config(video_path, mask_path, annotation_path, model_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

"""Initialize Counters and Buffers"""
line_counts = {i: 0 for i in range(len(lines))}
polygon_counts = {i: 0 for i in range(len(polygons))}
last_boxes, last_centers, track_colors = {}, {}, {}
bbox_buffer, bbox_history = defaultdict(list), defaultdict(list)
counted_ids = set()
recognized_ids = {}  # track_id -> name

"""Setup Visual Annotators"""
box_annotator = sv.BoxAnnotator(thickness=2)
line_zone = sv.LineZone(start=sv.Point(*lines[0][0]), end=sv.Point(*lines[0][1]))

"""Setup Face Recognition"""
face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

logging.info("Initializing DeepFace model...")
try:
    deepface_model = DeepFace.build_model("VGG-Face")
    logging.info("DeepFace model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load DeepFace model: {e}")
    deepface_model = None

"""Main Video Processing Loop"""
frame_count = 0
while cap.isOpened():
    try:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        detections = sv.Detections.from_ultralytics(model(frame, verbose=False)[0])
        detections = detections[detections.class_id == 0]

        if len(detections) == 0:
            out.write(frame)
            cv2.imshow("People Counting", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        detection_array = np.hstack((detections.xyxy, detections.confidence.reshape(-1, 1), detections.class_id.reshape(-1, 1)))
        tracks = tracker.update(detection_array, frame)
        if len(tracks) == 0:
            out.write(frame)
            cv2.imshow("People Counting", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        annotated_detections = sv.Detections(
            xyxy=tracks[:, :4],
            class_id=tracks[:, -1].astype(int),
            tracker_id=tracks[:, 4].astype(int)
        )

        crossed_in, crossed_out = line_zone.trigger(annotated_detections)
        frame_copy = frame.copy()
        frame = box_annotator.annotate(scene=frame, detections=annotated_detections)

        rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

        # === FACE RECOGNITION setiap 10 frame ===
        if frame_count % 10 == 0:
            for bbox, track_id in zip(annotated_detections.xyxy, annotated_detections.tracker_id):
                if track_id in recognized_ids:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                box_crop = rgb_frame[y1:y2, x1:x2]
                cv2.imwrite("box_crop.jpg", cv2.cvtColor(box_crop, cv2.COLOR_RGB2BGR))
                faces = detect_faces(box_crop, face_detector)
                
                for (fx1, fy1, fx2, fy2) in faces:
                    face_crop = rgb_frame[fy1:fy2, fx1:fx2]
                    temp_file = "temp_face.jpg"
                    cv2.imwrite(temp_file, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                    try:
                        results = DeepFace.find(
                            img_path=temp_file,
                            db_path=known_faces_folder,
                            model=deepface_model,
                            enforce_detection=False,
                            silent=True
                        )
                        results_df = results[0] if isinstance(results, list) else results
                        if not results_df.empty:
                            name = os.path.basename(results_df.iloc[0]['identity']).split('.')[0]
                            recognized_ids[track_id] = name
                        else:
                            recognized_ids[track_id] = "Unknown"
                            save_unknown_face(face_crop)
                    except Exception as e:
                        logging.warning(f"Face recognition error: {e}")
                        recognized_ids[track_id] = "Error"

        # === Tampilkan Nama, Tracking & Counting ===
        for bbox, track_id in zip(annotated_detections.xyxy, annotated_detections.tracker_id):
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            name = recognized_ids.get(track_id, "")
            if name:
                cv2.putText(frame, name, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

        frame = draw_overlays(frame, lines, polygons, line_counts, polygon_counts, line_zone)
        out.write(frame)
        cv2.imshow("People Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        break

"""Cleanup"""
cap.release()
out.release()
cv2.destroyAllWindows()

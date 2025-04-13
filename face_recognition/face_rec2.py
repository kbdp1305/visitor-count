import os
import cv2
import time
import uuid
import numpy as np
from deepface import DeepFace
import mediapipe as mp

# === Config ===
known_faces_folder = r"C:\Work\CarbonZAP\visitor_count\face_recognition\dataset\known"
video_source = r"C:\Work\CarbonZAP\visitor_count\cctv.mp4"
save_interval = 2  # in seconds
last_saved_time = 0
os.makedirs(known_faces_folder, exist_ok=True)

# === Check if the folder is empty ===
def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0

# === Detect faces using MediaPipe ===
def detect_faces(frame, face_detector):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    h, w, _ = frame.shape
    faces = []

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            w_box, h_box = int(bbox.width * w), int(bbox.height * h)
            x, y = max(x, 0), max(y, 0)
            x2, y2 = min(x + w_box, w), min(y + h_box, h)
            # x2 = int((bbox.xmin + bbox.width) * w)
            # y2 = int((bbox.ymin + bbox.height) * h)
            faces.append((x, y, x2, y2))
    return faces

# === Save unknown face to folder ===
def save_unknown_face(face_crop):
    global last_saved_time
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(known_faces_folder, filename)
    cv2.imwrite(save_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
    print(f"[+] Saved unknown face to: {save_path}")
    last_saved_time = time.time()

# === Main process per frame ===
def process_frame(frame, face_detector):
    global last_saved_time
    faces = detect_faces(frame, face_detector)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Check if known faces folder is empty and save first unknown face
    if is_folder_empty(known_faces_folder) and len(faces) > 0:
        face_crop = rgb[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]]
        save_unknown_face(face_crop)

    for (x, y, x2, y2) in faces:
        face_crop = rgb[y:y2, x:x2]
        temp_file = "temp_face.jpg"
        cv2.imwrite(temp_file, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

        try:
            results = DeepFace.find(
                img_path=temp_file,
                db_path=known_faces_folder,
                model_name="VGG-Face",
                enforce_detection=False,
                silent=True
            )
            results_df = results[0] if isinstance(results, list) else results

            if not results_df.empty:
                name = os.path.basename(results_df.iloc[0]['identity']).split('.')[0]
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
                if time.time() - last_saved_time > save_interval:
                    save_unknown_face(face_crop)

        except Exception as e:
            print(f"Recognition error: {e}")
            name = "Error"
            color = (0, 255, 255)

        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

# === Main Loop ===
def run_video_loop(video_source):
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame, face_detector)
        cv2.imshow("Face Recognition", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Start ===
if __name__ == "__main__":
    run_video_loop(video_source)

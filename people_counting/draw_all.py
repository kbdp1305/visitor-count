import cv2
import json
import numpy as np

# Paths
video_path = r"C:\Work\CarbonZAP\visitor_count\cctv.mp4"
output_json = r"C:\Work\CarbonZAP\visitor_count\people-counting\images\annotations.json"

# State
drawing_mode = "line"  # 'line' or 'polygon'
current_points = []
lines = []
polygons = []

# Mouse click handler
def click_event(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Point added: ({x}, {y})")

# Load first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to load video.")
    exit()

cv2.namedWindow("Annotation Tool")
cv2.setMouseCallback("Annotation Tool", click_event)

while True:
    temp_frame = frame.copy()

    # Draw finalized lines
    for line in lines:
        cv2.line(temp_frame, line[0], line[1], (0, 255, 0), 2)

    # Draw finalized polygons
    for poly in polygons:
        cv2.polylines(temp_frame, [np.array(poly, np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)

    # Draw current points
    for pt in current_points:
        cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)

    # Draw preview (line or polygon)
    if drawing_mode == "line" and len(current_points) == 2:
        cv2.line(temp_frame, current_points[0], current_points[1], (0, 200, 200), 1)
    elif drawing_mode == "polygon" and len(current_points) >= 2:
        for i in range(1, len(current_points)):
            cv2.line(temp_frame, current_points[i - 1], current_points[i], (200, 200, 0), 1)
        if len(current_points) >= 3:
            cv2.line(temp_frame, current_points[-1], current_points[0], (200, 200, 0), 1)  # close preview


    # Instruction text
    cv2.putText(temp_frame, f"Mode: {drawing_mode.upper()} | 1: Line  2: Polygon  Enter: Finish Current  S: Save  R: Reset  Q: Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Annotation Tool", temp_frame)
    key = cv2.waitKey(1)

    if key == ord('1'):
        drawing_mode = "line"
        current_points.clear()
        print("Switched to LINE mode.")
    elif key == ord('2'):
        drawing_mode = "polygon"
        current_points.clear()
        print("Switched to POLYGON mode.")
    elif key == 13:  # Enter key
        if drawing_mode == "line" and len(current_points) == 2:
            lines.append(current_points.copy())
            print(f"Line saved: {current_points}")
            current_points.clear()
        elif drawing_mode == "polygon" and len(current_points) >= 3:
            polygons.append(current_points.copy())
            print(f"Polygon saved: {current_points}")
            current_points.clear()
        else:
            print("Not enough points to finalize.")
    elif key == ord('s'):
        annotations = {
            "lines": lines,
            "polygons": polygons
        }
        with open(output_json, 'w') as f:
            json.dump(annotations, f)
        print(f"Saved to {output_json}")
        break
    elif key == ord('r'):
        lines.clear()
        polygons.clear()
        current_points.clear()
        print("All annotations reset.")
    elif key == ord('q'):
        print("Exited without saving.")
        break

cv2.destroyAllWindows()

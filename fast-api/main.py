from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import cv2
import base64

from storage import load_cameras, save_cameras

app = FastAPI(title="Camera API")

# Camera model
class Camera(BaseModel):
    ip: str
    name: str
    rois: List[List[int]]
    roi_names: List[str]


# ========== Camera Management Endpoints ==========

@app.post("/camera", response_model=Camera)
def add_camera(camera: Camera):
    cameras = load_cameras()
    if any(existing["ip"] == camera.ip for existing in cameras):
        raise HTTPException(status_code=400, detail="Camera IP already exists.")
    camera_data = camera.dict()
    cameras.append(camera_data)
    save_cameras(cameras)
    return camera

@app.post("/cameras/bulk")
def add_multiple_cameras(camera_list: List[Camera]):
    cameras = load_cameras()
    added = []
    for camera in camera_list:
        if any(existing["ip"] == camera.ip for existing in cameras):
            continue
        cam_data = camera.dict()
        cameras.append(cam_data)
        added.append(cam_data)
    save_cameras(cameras)
    return {"added": added, "total": len(added)}

@app.get("/cameras", response_model=List[Camera])
def get_all_cameras():
    return load_cameras()

@app.get("/camera/{ip}", response_model=Camera)
def get_camera_by_ip(ip: str):
    cameras = load_cameras()
    for camera in cameras:
        if camera["ip"] == ip:
            return camera
    raise HTTPException(status_code=404, detail="Camera not found.")

# ========== One-Time Test Frame ==========

@app.get("/camera/test/{camera_id}")
def test_camera_stream(camera_id: str):
    try:
        cam_index = int(camera_id) if camera_id.isdigit() else camera_id
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise HTTPException(status_code=404, detail="Failed to open video stream.")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(status_code=500, detail="Could not read from the camera.")
        return {"message": f"Successfully connected to camera {camera_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Real-Time Stream ==========

def generate_frames(camera_ip: str):
    cap = cv2.VideoCapture(camera_ip)
    if not cap.isOpened():
        yield b"--frame\r\nContent-Type: text/plain\r\n\r\nFailed to open camera stream\r\n\r\n"
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
    cap.release()

@app.get("/camera/stream/{camera_ip}")
def stream_camera(camera_ip: str):
    cameras = load_cameras()
    if not any(cam["ip"] == camera_ip for cam in cameras):
        raise HTTPException(status_code=404, detail="Camera IP not found in saved cameras.")
    cam_index = int(camera_ip) if camera_ip.isdigit() else camera_ip
    return StreamingResponse(generate_frames(cam_index), media_type="multipart/x-mixed-replace; boundary=frame")

# ========== Run Server ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

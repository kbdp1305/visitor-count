import json
import os
from typing import List, Dict

CAMERA_FILE = "cameras.json"

def load_cameras() -> List[Dict]:
    if os.path.exists(CAMERA_FILE):
        with open(CAMERA_FILE, "r") as f:
            return json.load(f)
    return []

def save_cameras(cameras: List[Dict]):
    with open(CAMERA_FILE, "w") as f:
        json.dump(cameras, f, indent=4)

# import json

# camera_file_path = r"C:\Work\CarbonZAP\visitor_count\fast-api\cameras.json"

# try:
#     with open(camera_file_path, "r") as f:
#         cameras = json.load(f)
#         print("📸 Loaded Cameras:")
#         for cam in cameras:
#             print(json.dumps(cam, indent=4))
# except FileNotFoundError:
#     print("File not found.")
# except json.JSONDecodeError as e:
#     print(f"JSON decoding error: {e}")

import torch
import roboflow
import cv2
import os
import ultralytics
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import base64
from IPython.display import Image, display
import numpy as np
from inference_sdk.webrtc import VideoFileSource, StreamConfig, VideoMetadata
import glob

load_dotenv()

API_KEY = os.getenv("API_KEY") 
WORKSPACE = os.getenv("WORKSPACE") 
PROJECT_NAME = os.getenv("PROJECT_NAME")
VERSION_NUMBER = int(os.getenv("VERSION")) 

input_path = "predict/input/"
output_dir = "predict/output"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists("Fire-Smoke-Detection-Yolov11-1/"):
    rf = roboflow.Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
    dataset = project.version(VERSION_NUMBER).download("yolov11")

    print("Dataset downloaded successfully to:", dataset.location)
else:
    print("Dataset already exists.")

# Initialize the client for images
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="xGXvprcgEcDhhAphPosq"
)

# Get list of image files
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_path, ext)))

# Process each image
for image_file in image_files:
    result = client.run_workflow(
        workspace_name="ransomworkspace",
        workflow_id="detect-count-and-visualize-2",
        images={
            "image": image_file
        },
        use_cache=True
    )
    encoded_image_string = result[0]['output_image']
    decoded_image = base64.b64decode(encoded_image_string)
    
    # Create output filename
    base_name = os.path.basename(image_file)
    name, ext = os.path.splitext(base_name)
    output_image_path = os.path.join(output_dir, f"{name}_predicted{ext}")
    
    with open(output_image_path, 'wb') as f:
        f.write(decoded_image)
    
    print(f"Processed {image_file}, saved to {output_image_path}")

# Initialize client for videos
client_video = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="xGXvprcgEcDhhAphPosq"
)

# Get list of video files
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
video_files = []
for ext in video_extensions:
    video_files.extend(glob.glob(os.path.join(input_path, ext)))

# Process each video
for video_file in video_files:
    source = VideoFileSource(video_file, realtime_processing=False)
    
    VIDEO_OUTPUT = "output_image"
    DATA_OUTPUT = "count_objects"
    
    config = StreamConfig(
        stream_output=[],
        data_output=["output_image","count_objects","predictions"],
        requested_plan="webrtc-gpu-medium",
        requested_region="us",
    )
    
    session = client_video.webrtc.stream(
        source=source,
        workflow="detect-count-and-visualize-3",
        workspace="ransomworkspace",
        image_input="image",
        config=config
    )
    
    frames = []
    
    @session.on_data()
    def on_data(data: dict, metadata: VideoMetadata):
        timestamp_ms = metadata.pts * metadata.time_base * 1000
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data[VIDEO_OUTPUT]["value"]), np.uint8), cv2.IMREAD_COLOR)
        frames.append((timestamp_ms, metadata.frame_id, img))
        print(f"Processed frame {metadata.frame_id} for {video_file}")
    
    session.run()
    
    # Stitch frames into output video
    if frames:
        frames.sort(key=lambda x: x[1])
        fps = (len(frames) - 1) / ((frames[-1][0] - frames[0][0]) / 1000) if len(frames) > 1 else 30
        h, w = frames[0][2].shape[:2]
        base_name = os.path.basename(video_file)
        name, ext = os.path.splitext(base_name)
        output_video_path = os.path.join(output_dir, f"{name}_output.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _, _, frame in frames:
            out.write(frame)
        out.release()
        print(f"Processed {video_file}, saved to {output_video_path}")
    else:
        print(f"No frames processed for {video_file}")

print("All images and videos processed.")
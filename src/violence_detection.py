import os, cv2, time, torch, gc, numpy as np, psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import torchvision.models.video as models
import torchvision.transforms as transforms
import torch.nn as nn

# Import from your custom modules
from preprocessing import Preprocessor
from report_generation import export_violence_report
from stats import stats_monitor

class ViolenceDetectionModel(nn.Module):
    """Optimized 3D CNN model for violence detection"""
    def __init__(self, num_classes=2):
        super(ViolenceDetectionModel, self).__init__()
        self.base_model = models.r3d_18(weights="DEFAULT")
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def preprocess_clip(clip):
    """Optimized clip preprocessing with memory management"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_clip = []
    for i in range(0, len(clip), 16):
        batch = clip[i:i+8]
        processed_clip.extend([transform(frame) for frame in batch])
        del batch
        gc.collect()

    return torch.stack(processed_clip, dim=0).permute(1, 0, 2, 3).unsqueeze(0)

def extract_video_clips(video_path, clip_length=32, overlap=16, max_clips=100):
    """Memory-optimized clip extraction with chunking"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    clips, clip_start_times = [], []
    buffer, frame_count = [], 0

    while len(clips) < max_clips:
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if len(buffer) == clip_length:
            clips.append(buffer.copy())
            clip_start_times.append(frame_count - clip_length + 1)
            
            # Memory-efficient overlap handling
            buffer = buffer[-overlap:] if overlap > 0 else []
            
            # Periodic cleanup
            if len(clips) % 10 == 0:
                gc.collect()

        frame_count += 1

    cap.release()
    return clips, clip_start_times, fps

def load_violence_detection_model(device):
    """Model loading with memory optimization"""
    print("Loading optimized Violence Detection Model...")
    model = ViolenceDetectionModel().to(device)
    model.eval()
    
    # Freeze all layers except final classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.base_model.fc.parameters():
        param.requires_grad = True
        
    return model

def detect_violence_in_clip(clip, start_time, fps, model, device, threshold=0.65, preprocessor=None):
    """Optimized clip detection with resource monitoring"""
    if psutil.virtual_memory().percent > 85:
        gc.collect()
        return None

    try:
        # Apply view transformation augmentation
        if preprocessor:
            clip = [preprocessor.transform_view(frame, angle=np.random.uniform(-15,15)) for frame in clip]

        clip_tensor = preprocess_clip(clip).to(device)
        
        with torch.no_grad():
            outputs = model(clip_tensor)
            violence_prob = torch.nn.functional.softmax(outputs, dim=1)[0][1].item()

            # Apply temporal smoothing if preprocessor available
            if preprocessor:
                violence_prob = preprocessor.smooth_predictions([violence_prob])[0]

            if violence_prob > threshold:
                detection = {
                    'time': start_time / fps,
                    'probability': violence_prob,
                    'frame_idx': start_time,
                    'thumbnail': clip[0],
                    'video_path': "current_video",
                    'attention_map': preprocessor.generate_attention_map(model, clip) if preprocessor else None
                }
                stats_monitor.record_detection('violence', violence_prob, True, "current_video")
                return detection
    except RuntimeError as e:
        print(f"Error processing clip: {str(e)}")
        torch.cuda.empty_cache()
        
    return None

def display_video_with_detections(video_path, detections):
    """Display video with violence detection markers"""
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("Violence Detection", cv2.WINDOW_NORMAL)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        for det in [d for d in detections if d['frame_idx'] == frame_idx]:
            cv2.putText(frame, f"Violence: {det['probability']:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Violence Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()

def detect_violence_in_video(video_path, model, device, threshold=0.65, show_video=True):
    """Optimized video processing with controlled parallelism"""
    start_time = time.time()
    clips, clip_start_times, fps = extract_video_clips(video_path)
    violence_detections = []
    
    # Limit concurrent workers based on available memory
    max_workers = 2 if torch.cuda.is_available() else 4
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (clip, start_time) in enumerate(zip(clips, clip_start_times)):
            if i >= 100:  # Safety limit
                break
            futures.append(executor.submit(
                detect_violence_in_clip, clip, start_time, fps, model, device, threshold
            ))
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                result['video_path'] = video_path
                violence_detections.append(result)
    
    # Optional video display
    if show_video and violence_detections:
        display_video_with_detections(video_path, violence_detections)
    
    stats_monitor.record_processing_time(time.time() - start_time, os.path.basename(video_path))
    return violence_detections

def run_violence_detection(video_files, use_gpu=True):
    """Main function with resource management"""
    device = torch.device("cuda" if use_gpu else "cpu")
    model = load_violence_detection_model(device)
    preprocessor = Preprocessor()
    all_detections = {}
    
    for video_file in video_files:
        print(f"Processing {video_file}...")
        preprocessor.prep()
        try:
            violence_detections = detect_violence_in_video(
                video_file, model, device, show_video=True
            )
            all_detections[video_file] = violence_detections
            
            print(f"Found {len(violence_detections)} violent clips" if violence_detections 
                  else "No violence detected")
                
            # Explicit cleanup between videos
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue

    flattened_detections = []
    for video_file, detections in all_detections.items():
        if detections:
            flattened_detections.extend(detections)
    
    if flattened_detections:
        export_violence_report(flattened_detections, "combined_report")

    print("Violence detection complete!")
    return all_detections
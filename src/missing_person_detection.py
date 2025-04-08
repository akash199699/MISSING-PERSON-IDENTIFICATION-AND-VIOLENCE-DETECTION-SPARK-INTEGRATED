import sys, os, time, cv2, numpy as np, torch, asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Import from your custom modules
from utils import select_files
from preprocessing import Preprocessor
from report_generation import export_to_pdf
from stats import stats_monitor

def setup_missing_person_detection(use_gpu=True):
    """Initialize models and device for face detection"""
    print("Setting up Missing Person Detection System")
    
    device = torch.device('cuda' if use_gpu else 'cpu')
    print("Using", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    if device.type == 'cuda':
        resnet = resnet.half()
        
    return device, mtcnn, resnet

def load_reference_images(device, mtcnn, resnet):
    """Load and process reference images of the missing person"""
    print("Select reference images of the missing person:")
    ref_filenames = select_files("Select Reference Images", 
                               [("Image files", "*.jpg *.jpeg *.png")])
    
    if not ref_filenames:
        sys.exit("No reference image selected.")

    ref_embeddings = []
    for ref_filename in ref_filenames:
        ref_img = Image.open(ref_filename).convert("RGB")
        faces, probs = mtcnn(ref_img, return_prob=True)
        if faces is None or (hasattr(faces, '__len__') and len(faces) == 0):
            continue
            
        # Use highest probability face if multiple are detected
        ref_face = faces[int(np.argmax(probs))] if faces.ndim == 4 else faces
        with torch.no_grad():
            face_tensor = ref_face.unsqueeze(0).to(device)
            if device.type == 'cuda':
                face_tensor = face_tensor.half()  # Convert input to half precision
            emb = resnet(face_tensor)
        ref_embeddings.append(emb)

    if not ref_embeddings:
        sys.exit("No valid faces detected in the reference images.")

    return ref_embeddings, ref_filenames

def load_video_files():
    """Load video files to search for the missing person"""
    print("Select video files to analyze:")
    video_files = select_files("Select Video Files", 
                             [("Video files", "*.mp4 *.avi *.mov")])
    
    if not video_files:
        sys.exit("No video files selected.")
    return video_files

def fast_dominant_color(pil_img_region):
    """Compute the dominant color as the mean color of the region."""
    np_region = np.array(pil_img_region)
    return tuple(map(int, np_region.mean(axis=(0, 1))[:3]))

async def async_video_loader(video_path, buffer_size=64):
    """Asynchronous video loader that buffers frames to reduce disk I/O latency"""
    cap = cv2.VideoCapture(video_path)
    frames_buffer = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames_buffer) >= buffer_size:
            yield frames_buffer
            frames_buffer = []
    if frames_buffer:
        yield frames_buffer
    cap.release()

def process_batch(batch_info, video_filename, mtcnn, resnet, device, ref_embeddings, detection_threshold, preprocessor=None):
    """Process a batch of frames with statistical monitoring"""
    start_time = time.time()
    detections = []
    face_tensors = []
    face_meta = []

    for frame_idx, fps, orig_rgb in batch_info:
        pil_img = Image.fromarray(orig_rgb)
        boxes, _ = mtcnn.detect(pil_img)
        if boxes is None:
            continue
        faces, probs = mtcnn(pil_img, return_prob=True)
        if faces is None or faces.ndim != 4:
            continue
        for i, face in enumerate(faces):
            face_tensors.append(face.unsqueeze(0))
            face_meta.append((frame_idx, fps, orig_rgb, boxes[i], pil_img))

    if not face_tensors:
        stats_monitor.record_processing_time(time.time() - start_time, os.path.basename(video_filename))
        return detections

    faces_batch = torch.cat(face_tensors, dim=0).to(device)
    if device.type == 'cuda':
        faces_batch = faces_batch.half()

    with torch.no_grad():
        embeddings = resnet(faces_batch)

    for idx, embedding in enumerate(embeddings):
        frame_idx, fps, orig_rgb, box, pil_img = face_meta[idx]
        for ref_embedding in ref_embeddings:
            cos_sim = torch.nn.functional.cosine_similarity(
                ref_embedding, embedding.unsqueeze(0).to(device)
            ).item()
            
            if cos_sim > detection_threshold:
                normalized_face = preprocessor.normalize_pose(np.array(pil_img), box) if preprocessor else None
                detection_time = frame_idx / fps if fps else frame_idx
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate torso region
                torso_bottom = min(y2 + int((y2 - y1) * 1.5), pil_img.height)
                torso_region = pil_img.crop((x1, y2, x2, torso_bottom))
                
                detections.append({
                    'frame_idx': frame_idx,
                    'time': detection_time,
                    'similarity': cos_sim,
                    'video_filename': os.path.basename(video_filename),
                    'frame_img': orig_rgb,
                    'box': (x1, y1, x2, y2),
                    'normalized_face': normalized_face,
                    'dominant_color': fast_dominant_color(torso_region)
                })
                
                stats_monitor.record_detection('face', cos_sim, True, video_filename)

    stats_monitor.record_processing_time(time.time() - start_time, os.path.basename(video_filename))
    return detections

def process_video(video_filename, mtcnn, resnet, device, ref_embeddings, frame_interval=60, batch_size=16, detection_threshold=0.65):
    """Process a single video file using asynchronous loader"""
    start_time = time.time()
    detections_video = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    buffer_generator = async_video_loader(video_filename, buffer_size=64)
    batch_info = []
    frame_idx = 0
    
    # Get video FPS
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # Open a window to display the video
    cv2.namedWindow("Missing Person Detection", cv2.WINDOW_NORMAL)

    try:
        while True:
            try:
                frames_buffer = loop.run_until_complete(buffer_generator.__anext__())
            except StopAsyncIteration:
                break
                
            for frame in frames_buffer:
                if frame_idx % frame_interval == 0:
                    batch_info.append((frame_idx, fps, frame))
                    if len(batch_info) >= batch_size:
                        detections_video.extend(process_batch(
                            batch_info, video_filename, mtcnn, resnet, 
                            device, ref_embeddings, detection_threshold
                        ))
                        batch_info = []

                    # Display the frame with bounding boxes
                    display_frame = frame.copy()
                    for det in detections_video:
                        if det['frame_idx'] == frame_idx:
                            x1, y1, x2, y2 = det['box']
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    cv2.imshow("Missing Person Detection", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                        
                frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        if batch_info:
            detections_video.extend(process_batch(
                batch_info, video_filename, mtcnn, resnet, 
                device, ref_embeddings, detection_threshold
            ))
        
        stats_monitor.record_processing_time(
            time.time() - start_time,
            os.path.basename(video_filename)
        )
        loop.close()
        cv2.destroyAllWindows()

    return detections_video

def run_missing_person_detection():
    """Main function to run the missing person detection pipeline"""
    # Setup
    device, mtcnn, resnet = setup_missing_person_detection()
    preprocessor = Preprocessor()
    preprocessor.prep()

    # Parameters
    frame_interval = 60
    detection_threshold = 0.65
    batch_size = 16

    # Load reference images and videos
    ref_embeddings, ref_filenames = load_reference_images(device, mtcnn, resnet)
    video_files = load_video_files()

    print("Starting video processing...")
    start_time = time.time()
    all_detections = []

    # Process videos concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(video_files), os.cpu_count())) as executor:
        future_to_video = {
            executor.submit(
                process_video, vf, mtcnn, resnet, device, ref_embeddings,
                frame_interval, batch_size, detection_threshold
            ): vf for vf in video_files
        }
        for future in as_completed(future_to_video):
            all_detections.extend(future.result())

    print(f"Processing completed in {time.time() - start_time:.2f}s")

    if all_detections:
        # Sort detections by similarity (highest first)
        all_detections.sort(key=lambda x: x['similarity'], reverse=True)
        export_to_pdf(all_detections, ref_filenames=ref_filenames)
    else:
        print("No matches found.")

    print("Missing person detection complete!")
    return all_detections
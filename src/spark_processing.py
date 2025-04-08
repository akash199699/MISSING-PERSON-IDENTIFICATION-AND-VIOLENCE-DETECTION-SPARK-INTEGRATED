from pyspark.sql import SparkSession
from pyspark import SparkFiles
import os, sys, tempfile, time, logging
import numpy as np
import torch
import cv2
from PIL import Image
from functools import partial
from typing import Tuple, List, Dict, Any
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark=True
torch.set_float32_matmul_precision('high')

def initialize_spark():
    """Initialize Spark session with optimized configuration for video processing"""
    checkpoint_dir = tempfile.mkdtemp(prefix="spark_checkpoints_")
    venv_python = "C:/Users/91903/OneDrive/Desktop/1111198/III/MINI PROJECT/FDBD/DE NOVO/venv/Scripts/python.exe"
    conf = config.SPARK_CONF

    try:
        builder = SparkSession.builder \
            .appName(conf.get("app", "CCTV Analysis System")) \
            .master(conf.get("master", "local[*]")) \
            .config("spark.pyspark.python", venv_python) \
            .config("spark.pyspark.driver.python", venv_python)

        # Apply user-defined or default Spark settings from config
        for key, value in conf.items():
            if key not in ["app", "master"]:
                builder = builder.config(f"spark.{key}", value)

        # Additional hard-coded optimizations
        builder = builder \
            .config("spark.dynamicAllocation.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.kryoserializer.buffer.max", "512m") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.maxResultSize", "4g") \
            .config("spark.network.timeout", "600s") \
            .config("spark.worker.timeout", "600") \
            .config("spark.rpc.message.maxSize", "256") \
            .config("spark.executor.heartbeatInterval", "60s") \
            .config("spark.task.maxFailures", "4")

        spark = builder.getOrCreate()
        spark.sparkContext.setCheckpointDir(checkpoint_dir)
        logger.info(f"Initialized Spark session with checkpoint directory: {checkpoint_dir}")
        return spark

    except Exception as e:
        logger.error(f"Failed to initialize Spark session: {e}")
        return None

class VideoPipeline:
    def __init__(self, spark_context):
        """Initialize the video processing pipeline with Spark context"""
        self.sc = spark_context
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing pipeline with device: {self.device}")
        
    def prepare_models(self) -> Tuple[Any, Any, Any]:
        """Initialize models directly without saving to files"""
        from facenet_pytorch import MTCNN, InceptionResnetV1
        try:
            from violence_detection import ViolenceDetectionModel
            
            # Initialize models on driver
            device = self.device
            use_half = device.type == 'cuda'
            mtcnn = MTCNN(keep_all=True, device=device)
            resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            violence_model = ViolenceDetectionModel().to(device)
            
            if use_half:
                resnet = resnet.half()
                violence_model = violence_model.half()
            else:
                resnet = resnet.float()
                violence_model = violence_model.float()
            
            logger.info("Models initialized successfully")
            return mtcnn, resnet, violence_model
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def broadcast_models(self):
        """Broadcast models to all workers"""
        logger.info("Broadcasting models to workers")
        models = self.prepare_models()
        return self.sc.broadcast(models)
    
    def broadcast_embeddings(self, ref_embeddings):
        """Broadcast reference embeddings to all workers"""
        logger.info(f"Broadcasting {len(ref_embeddings)} reference embeddings to workers")
        return self.sc.broadcast(ref_embeddings)

def extract_video_metadata(video_path):
    """Extract basic metadata from a video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'path': video_path, 'valid': False, 'error': 'Failed to open video file'}
        
        # Get basic video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'path': video_path,
            'valid': True,
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration,
            'size_mb': os.path.getsize(video_path) / (1024 * 1024)
        }
    except Exception as e:
        return {'path': video_path, 'valid': False, 'error': str(e)}

def create_video_batches(video_paths, batch_size=10):
    """Group videos into balanced batches for processing"""
    # Get metadata for all videos
    video_metadata = [extract_video_metadata(path) for path in video_paths]
    valid_videos = [v for v in video_metadata if v['valid']]
    
    if not valid_videos:
        logger.warning("No valid videos found")
        return []
    
    # Sort by size (largest first) for better load balancing
    valid_videos.sort(key=lambda x: x['size_mb'], reverse=True)
    
    # Create balanced batches using a greedy approach
    batches = []
    current_batch = []
    current_size = 0
    target_size = sum(v['size_mb'] for v in valid_videos) / (len(valid_videos) / batch_size)
    
    for video in valid_videos:
        current_batch.append(video['path'])
        current_size += video['size_mb']
        
        if len(current_batch) >= batch_size or current_size >= target_size:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
    
    if current_batch:
        batches.append(current_batch)
    
    logger.info(f"Created {len(batches)} balanced video batches")
    return batches

def load_models_on_worker(models_bc):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn, resnet, violence_model = models_bc.value
    
    # Move models to device and convert precision
    mtcnn = mtcnn.to(device)
    resnet = resnet.to(device)
    violence_model = violence_model.to(device)
    
    # Convert precision if using CUDA
    if device.type == 'cuda':
        resnet = resnet.half()
        violence_model = violence_model.half()
    
    return device, mtcnn, resnet, violence_model

def process_video_batch_for_missing_person(batch, frame_interval, models_bc, ref_embeddings_bc, detection_threshold=0.65):
    """Process a batch of videos for missing person detection using broadcasted models"""
    device, mtcnn, resnet, _ = load_models_on_worker(models_bc)
    ref_embeddings = [emb.to(device).half() if device.type == 'cuda' else emb.to(device) 
                      for emb in ref_embeddings_bc.value]
    
    all_detections = []
    
    for video_path in batch:
        try:
            detections = process_single_video_missing_person(
                video_path, frame_interval, device, mtcnn, resnet, 
                ref_embeddings, detection_threshold
            )
            all_detections.extend(detections)
            logger.info(f"Processed {video_path}: Found {len(detections)} detections")
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
    
    return all_detections

def process_single_video_missing_person(video_path, frame_interval, device, mtcnn, resnet, ref_embeddings, detection_threshold):
    """Process a single video for missing person detection"""
    detections = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return detections
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # Default to 30 FPS if not detected
    
    frame_idx = 0
    use_half_precision = device.type == 'cuda'
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process only every N frames
        if frame_idx % frame_interval == 0:
            # Convert frame to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # Detect faces
            boxes, probs = mtcnn.detect(pil_img, landmarks=False)
            
            if boxes is not None:
                # Get face embeddings
                try:
                    faces = mtcnn(pil_img)
                    
                    if faces is not None and isinstance(faces, torch.Tensor):
                        # Handle single face vs. batch of faces
                        if faces.dim() == 3:  # Single face
                            faces = faces.unsqueeze(0)
                            
                        # Ensure faces tensor is on the same device as the model
                        faces = faces.to(device)
                        
                        for i, face in enumerate(faces):
                            if i >= len(boxes):  # Safety check
                                continue
                                
                            face_tensor = face.to(device)
                            
                            # Ensure consistent precision
                            if use_half_precision:
                                face_tensor = face_tensor.half()
                            else:
                                face_tensor = face_tensor.float()
                            
                            # Get embedding
                            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_half_precision):
                                embedding = resnet(face_tensor.unsqueeze(0))
                            
                            # Compare with reference embeddings
                            for ref_idx, ref_embedding in enumerate(ref_embeddings):
                                # Ensure both tensors are on same device with same type
                                ref_emb = ref_embedding.to(device)
                                
                                if use_half_precision:
                                    ref_emb = ref_emb.half()
                                else:
                                    ref_emb = ref_emb.float()
                                
                                cos_sim = torch.nn.functional.cosine_similarity(
                                    ref_emb, embedding
                                ).item()
                                
                                if cos_sim > detection_threshold:
                                    # Calculate dominant color of clothing
                                    x1, y1, x2, y2 = map(int, boxes[i])
                                    torso_top = y2
                                    torso_bottom = min(y2 + int((y2 - y1) * 1.5), pil_img.height)
                                    try:
                                        torso_region = pil_img.crop((x1, torso_top, x2, torso_bottom))
                                        np_region = np.array(torso_region)
                                        dominant_color = tuple(map(int, np_region.mean(axis=(0, 1))[:3]))
                                    except Exception:
                                        dominant_color = (0, 0, 0)  # Default if calculation fails
                                    
                                    # Add detection
                                    detection_time = frame_idx / fps
                                    detections.append({
                                        'frame_idx': frame_idx,
                                        'time': detection_time,
                                        'similarity': cos_sim,
                                        'video_filename': os.path.basename(video_path),
                                        'video_path': video_path,
                                        'box': (x1, y1, x2, y2),
                                        'dominant_color': dominant_color,
                                        'frame_img': rgb_frame.copy()
                                    })
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx} in {video_path}: {str(e)}")
        
        frame_idx += 1
    
    cap.release()
    return detections

def optimize_video_processing_parameters(batch_size=10, frame_interval=60):
    """Dynamically determine optimal processing parameters based on hardware"""
    cpu_count = os.cpu_count() or 4
    
    # Determine if GPU is available
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU available with {gpu_memory_gb:.2f} GB memory")
        # Adjust batch size based on GPU memory
        batch_size = 15 if gpu_memory_gb >= 8 else (10 if gpu_memory_gb >= 4 else 5)
    else:
        logger.info(f"Using CPU with {cpu_count} cores")
        # Adjust batch size based on CPU cores
        batch_size = max(1, min(cpu_count // 2, 8))
    
    # Set frame interval based on batch size
    frame_interval = max(30, 60 // batch_size * 30)
    
    logger.info(f"Using batch_size={batch_size}, frame_interval={frame_interval}")
    return batch_size, frame_interval

def process_video_batch_for_violence(batch, models_bc, clip_length=32, clip_stride=16, detection_threshold=0.7):
    """Process a batch of videos for violence detection using broadcasted models"""
    device, _, _, violence_model = load_models_on_worker(models_bc)
    all_detections = []
    
    for video_path in batch:
        try:
            detections = process_single_video_violence(
                video_path, device, violence_model, clip_length, 
                clip_stride, detection_threshold
            )
            all_detections.extend(detections)
            logger.info(f"Processed {video_path} for violence: Found {len(detections)} detections")
        except Exception as e:
            logger.error(f"Error processing {video_path} for violence: {str(e)}")
    
    return all_detections

def process_single_video_violence(video_path, device, model, clip_length=32, clip_stride=16, detection_threshold=0.7):
    """Process a single video for violence detection"""
    import torchvision.transforms as transforms
    
    detections = []
    cap = cv2.VideoCapture(video_path)
    use_half_precision = device.type == 'cuda'
    
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return detections
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # Default to 30 FPS if not detected
    
    # Set up preprocessing
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Buffer for frames
    buffer = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(rgb_frame)
        
        # When buffer reaches clip length, process it
        if len(buffer) == clip_length:
            # Process clip
            processed_clip = []
            for frame in buffer:
                pil_frame = Image.fromarray(frame)
                processed_frame = transform(pil_frame)
                processed_clip.append(processed_frame)
            
            # Create tensor
            clip_tensor = torch.stack(processed_clip, dim=0)
            clip_tensor = clip_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            
            if use_half_precision:
                clip_tensor = clip_tensor.half()
            else:
                clip_tensor = clip_tensor.float()
            
            # Run inference
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_half_precision):
                outputs = model(clip_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                violence_prob = probabilities[0][1].item()  # Assuming class 1 is violence
                
                if violence_prob > detection_threshold:
                    start_time = (frame_idx - clip_length + 1) / fps
                    detections.append({
                        'time': start_time,
                        'probability': violence_prob,
                        'frame_idx': frame_idx - clip_length + 1,
                        'video_filename': os.path.basename(video_path),
                        'video_path': video_path,
                        'thumbnail': buffer[0].copy()
                    })
            
            # Slide buffer by stride
            buffer = buffer[clip_stride:]
        
        frame_idx += 1
    
    cap.release()
    return detections

def load_reference_images(ref_files):
    """Load reference images of the missing person"""
    from facenet_pytorch import MTCNN, InceptionResnetV1
    
    logger.info(f"Loading reference images from {ref_files}")
    
    if not ref_files:
        logger.error(f"No reference images found")
        return [], []
    
    # Set up models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    if device.type == 'cuda':
        resnet = resnet.half()
    
    # Process reference images
    ref_embeddings = []
    valid_filenames = []
    
    for ref_filename in ref_files:
        try:
            ref_img = Image.open(ref_filename).convert("RGB")
            faces, probs = mtcnn(ref_img, return_prob=True)
            
            if faces is None or (hasattr(faces, '__len__') and len(faces) == 0):
                logger.warning(f"No face detected in {ref_filename}")
                continue
                
            # Use highest probability face if multiple are detected
            ref_face = faces[int(np.argmax(probs))].to(device) if faces.ndim == 4 else faces
            
            if device.type == 'cuda':
                ref_face = ref_face.half()
            else:
                ref_face = ref_face.float()
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                emb = resnet(ref_face.unsqueeze(0))
                
            # Store embeddings on CPU for serialization
            if device.type == 'cuda':
                emb = emb.half()
            ref_embeddings.append(emb.cpu())
            valid_filenames.append(ref_filename)
            logger.info(f"Processed reference image: {ref_filename}")
            
        except Exception as e:
            logger.error(f"Error processing reference image {ref_filename}: {str(e)}")
    
    if not ref_embeddings:
        logger.error("No valid faces detected in the reference images")
        return [], []
    
    return ref_embeddings, valid_filenames

def run_distributed_pipeline(spark, video_files, ref_files=None, run_violence=False):
    """Run the full distributed analysis pipeline using optimized model broadcasting"""
    start_time = time.time()
    logger.info(f"Starting distributed analysis pipeline")
    
    # Create pipeline instance
    pipeline = VideoPipeline(spark.sparkContext)
    
    # Find all video files
    video_paths = video_files
    
    if not video_paths:
        logger.error(f"No video files found")
        return [], [], []
    
    logger.info(f"Found {len(video_paths)} video files")
    
    # Broadcast models to all workers
    models_bc = pipeline.broadcast_models()
    
    # Load reference images if provided
    ref_embeddings = []
    ref_filenames = []
    ref_embeddings_bc = None
    
    if ref_files:
        ref_embeddings, ref_filenames = load_reference_images(ref_files)
        if ref_embeddings:
            ref_embeddings_bc = pipeline.broadcast_embeddings(ref_embeddings)
    
    # Run missing person detection if reference embeddings are available
    missing_detections = []
    if ref_embeddings and ref_embeddings_bc:
        logger.info("Running distributed missing person detection")
        
        # Create balanced batches of videos
        batch_size, frame_interval = optimize_video_processing_parameters()
        video_batches = create_video_batches(video_paths, batch_size=batch_size)
        
        if video_batches:
            # Process video batches in parallel
            batches_rdd = spark.sparkContext.parallelize(video_batches, len(video_batches))
            results_rdd = batches_rdd.map(
                lambda batch: process_video_batch_for_missing_person(
                    batch, frame_interval, models_bc, ref_embeddings_bc
                )
            )
            
            # Collect and flatten results
            all_missing_detections = []
            for batch_detections in results_rdd.collect():
                all_missing_detections.extend(batch_detections)
            
            # Sort by similarity (highest first)
            all_missing_detections.sort(key=lambda x: x['similarity'], reverse=True)
            missing_detections = all_missing_detections
            
    # Run violence detection if requested
    violence_detections = []
    if run_violence:
        logger.info("Running violence detection")
        target_videos = video_paths
        if missing_detections:
            target_videos = list(set([det['video_path'] for det in missing_detections]))
        
        # Create balanced batches for violence detection
        batch_size, _ = optimize_video_processing_parameters()
        video_batches = create_video_batches(target_videos, batch_size=batch_size)
        
        if video_batches:
            batches_rdd = spark.sparkContext.parallelize(video_batches, len(video_batches))
            results_rdd = batches_rdd.map(
                lambda batch: process_video_batch_for_violence(batch, models_bc)
            )
            
            all_violence_detections = []
            for batch_detections in results_rdd.collect():
                all_violence_detections.extend(batch_detections)
            
            all_violence_detections.sort(key=lambda x: x['probability'], reverse=True)
            violence_detections = all_violence_detections
    
    elapsed_time = time.time() - start_time
    logger.info(f"Distributed pipeline completed in {elapsed_time:.2f}s")
    return missing_detections, violence_detections, ref_filenames

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed CCTV Analysis System")
    parser.add_argument("--video_dir", required=True, help="Directory containing video files")
    parser.add_argument("--ref_images_dir", help="Directory containing reference images for missing person")
    parser.add_argument("--violence_only", action="store_true", help="Run only violence detection")
    parser.add_argument("--output_dir", default="output", help="Directory to save reports")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Spark
    spark = initialize_spark()
    
    try:
        # Find all video files
        video_files = []
        if os.path.isdir(args.video_dir):
            for root, _, files in os.walk(args.video_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_files.append(os.path.join(root, file))
        else:
            video_files = [args.video_dir]  # Single file
        
        # Find reference images
        ref_files = []
        if not args.violence_only and args.ref_images_dir:
            if os.path.isdir(args.ref_images_dir):
                for root, _, files in os.walk(args.ref_images_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            ref_files.append(os.path.join(root, file))
            elif os.path.isfile(args.ref_images_dir):
                ref_files = [args.ref_images_dir]  # Single file
        
        # Run the distributed pipeline
        missing_detections, violence_detections, ref_filenames = run_distributed_pipeline(
            spark,
            video_files,
            ref_files=None if args.violence_only else ref_files,
            run_violence=True
        )
        
        # Generate reports
        if missing_detections:
            from report_generation import export_to_pdf
            export_to_pdf(
                missing_detections, 
                ref_filenames=ref_filenames,
                pdf_filename=os.path.join(args.output_dir, "missing_person_report.pdf")
            )
        
        if violence_detections:
            from report_generation import export_violence_report
            from collections import defaultdict
            
            # Group violence detections by video
            videos_detections = defaultdict(list)
            for det in violence_detections:
                videos_detections[det['video_path']].append(det)
            
            # Generate report for each video
            for video_path, detections in videos_detections.items():
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                export_violence_report(
                    detections,
                    video_path,
                    pdf_filename=os.path.join(args.output_dir, f"violence_{video_name}.pdf")
                )
        
        # Generate combined report
        from report_generation import generate_combined_report
        generate_combined_report(
            missing_detections,
            violence_detections,
            ref_filenames=ref_filenames,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n" + "="*50)
        print("CCTV ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total Missing Person Detections: {len(missing_detections)}")
        print(f"Total Violence Detections: {len(violence_detections)}")
        
        if missing_detections:
            print("\nTop Videos with Missing Person Detections:")
            from collections import Counter
            for video, count in Counter([det['video_filename'] for det in missing_detections]).most_common(3):
                print(f"  - {video}: {count} detections")
        
        if violence_detections:
            print("\nTop Videos with Violence Detections:")
            from collections import Counter
            for video, count in Counter([det['video_filename'] for det in violence_detections]).most_common(3):
                print(f"  - {video}: {count} detections")
        
        print(f"\nReports have been saved to: {os.path.abspath(args.output_dir)}")
        print("="*50)
    
    finally:
        # Stop Spark session
        spark.stop()
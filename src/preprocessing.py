import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from pyspark.sql import SparkSession
import torchvision.transforms as transforms

class Preprocessor:
    """Central preprocessing class containing all shared preprocessing utilities"""
    
    def __init__(self):
        self.kalman_tracker = self._init_kalman_filter()
        self.face_detector = MTCNN(keep_all=True)
        self.face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()
        
    @staticmethod
    def _init_kalman_filter():
        """Initialize Kalman filter for object tracking"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        return kalman

    def track_objects(self, detections):
        """Track objects using Kalman filter"""
        tracked_boxes = []
        for bbox in detections:
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2)/2, (y1 + y2)/2
            
            self.kalman_tracker.predict()
            updated = self.kalman_tracker.correct(np.array([center_x, center_y], dtype=np.float32))
            
            updated_x, updated_y = updated[:2].flatten()
            w, h = x2-x1, y2-y1
            tracked_boxes.append((
                int(updated_x - w/2), int(updated_y - h/2),
                int(updated_x + w/2), int(updated_y + h/2)
            ))
        return tracked_boxes

    @staticmethod
    def normalize_pose(image, landmarks):
        """Normalize face pose using affine transform"""
        try:
            src = np.array(landmarks[:3], dtype=np.float32)
            dst = np.array([[50,50], [150,50], [100,150]], dtype=np.float32)
            M = cv2.getAffineTransform(src, dst)
            return cv2.warpAffine(image, M, (200,200))
        except Exception as e:
            print(f"Pose normalization failed: {e}")
            return image

    @staticmethod
    def transform_view(image, angle=0, scale=1.0):
        """Apply view transformation"""
        try:
            h, w = image.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            return cv2.warpAffine(image, M, (w,h))
        except Exception as e:
            print(f"View transform failed: {e}")
            return image

    @staticmethod
    def smooth_predictions(probabilities, window=5):
        """Apply temporal smoothing to prediction probabilities"""
        return [np.mean(probabilities[max(0,i-window+1):i+1]) 
                for i in range(len(probabilities))]

    @staticmethod
    def prep():
        """Prepare data for processing"""
        print("Preparing Data for Processing...")

    def generate_attention_map(self, model, frames):
        """Generate Grad-CAM attention maps"""
        try:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64,64)),
                transforms.ToTensor()
            ])
            input_tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0)
            
            model.eval()
            features = model.base_model.layer4
            output = model(input_tensor)
            
            output[:,1].backward()
            grads = model.base_model.layer4.weight.grad
            pooled_grads = torch.mean(grads, dim=[0,2,3,4])
            
            activations = features.detach()
            for i in range(activations.shape[1]):
                activations[:,i,:,:,:] *= pooled_grads[i]
                
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = np.maximum(heatmap.cpu().numpy(), 0)
            return heatmap / np.max(heatmap)
        except Exception as e:
            print(f"Attention map generation failed: {e}")
            return None

    @staticmethod
    def init_spark():
        """Initialize Spark session for distributed processing"""
        try:
            return SparkSession.builder \
                .appName("CCTVProcessor") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "4g") \
                .getOrCreate()
        except Exception as e:
            print(f"Spark init failed: {e}")
            return None

    @staticmethod
    def partition_videos(video_paths, n_partitions=None):
        """Optimize video partitioning"""
        n_partitions = n_partitions or os.cpu_count() or 4
        sizes = [(f, os.path.getsize(f)) for f in video_paths if os.path.exists(f)]
        
        partitions = [[] for _ in range(n_partitions)]
        current_sizes = [0] * n_partitions
        
        for f, s in sorted(sizes, key=lambda x: x[1], reverse=True):
            idx = np.argmin(current_sizes)
            partitions[idx].append(f)
            current_sizes[idx] += s
            
        return partitions
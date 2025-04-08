import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from config import config

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

class StatsMonitor:
    def __init__(self):
        self.reset()
        self.last_log_time = time.time()
        os.makedirs(config.STATS_LOG_DIR, exist_ok=True)
        
    def reset(self):
        self.detections = []
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.start_time = time.time()
        self.frame_counts = defaultdict(int)
        self.processing_times = []
        
    def record_detection(self, detection_type, confidence, is_correct, video_source=None):
        timestamp = time.time()
        entry = {
            'timestamp': timestamp,
            'type': detection_type,
            'confidence': confidence,
            'correct': is_correct,
            'video': video_source
        }
        self.detections.append(entry)
        
        if is_correct:
            if confidence >= (0.65 if detection_type == 'face' else 0.7):
                self.true_positives += 1
            else:
                self.true_negatives += 1
        else:
            if confidence >= (0.65 if detection_type == 'face' else 0.7):
                self.false_positives += 1
            else:
                self.false_negatives += 1
                
        if video_source:
            self.frame_counts[video_source] += 1
            
        # Periodic logging
        if time.time() - self.last_log_time > config.STATS_LOG_INTERVAL:
            self.log_stats()
            self.last_log_time = time.time()
    
    def record_processing_time(self, elapsed_time, video_source=None):
        """Record processing time for a specific video"""
        self.processing_times.append((elapsed_time, video_source))

    def _calculate_video_times(self):
        """Calculate total processing time per video"""
        video_times = defaultdict(float)
        for time, video in self.processing_times:
            if video:  # Only count if video source is specified
                video_times[video] += time
        return dict(video_times)
    
    def get_performance_metrics(self):
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return {}
            
        accuracy = (self.true_positives + self.true_negatives) / total
        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'fps': sum(self.frame_counts.values()) / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0,
            'video_times': self._calculate_video_times(),
            'frame_counts': dict(self.frame_counts)
        }
    
    def generate_histograms(self):
        face_confidences = [d['confidence'] for d in self.detections if d['type'] == 'face']
        violence_confidences = [d['confidence'] for d in self.detections if d['type'] == 'violence']
        
        try:
            plt.figure(figsize=(12, 5))
        
            if face_confidences:
                plt.subplot(1, 2, 1)
                plt.hist(face_confidences, bins=config.HISTOGRAM_BINS, alpha=0.7, color='blue')
                plt.title('Face Recognition Confidence')
                plt.xlabel('Confidence Score')
                plt.ylabel('Count')
        
            if violence_confidences:
                plt.subplot(1, 2, 2)
                plt.hist(violence_confidences, bins=config.HISTOGRAM_BINS, alpha=0.7, color='red')
                plt.title('Violence Detection Confidence')
                plt.xlabel('Confidence Score')
                plt.ylabel('Count')
        
            hist_path = os.path.join(config.STATS_LOG_DIR, f'histograms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()
            return hist_path
        except Exception as e:
            print(f"Error generating histogram: {e}")
            return None
    
    def log_stats(self):
        if not config.PERFORMANCE_LOG:
            return
            
        stats = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.get_performance_metrics(),
            'processing_times': self.processing_times,
            'frame_counts': dict(self.frame_counts)
        }
        
        log_file = os.path.join(config.STATS_LOG_DIR, f'stats_{datetime.now().strftime("%Y%m%d")}.json')
        
        try:
            existing_data = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing_data = json.load(f)
            
            existing_data.append(stats)
            
            with open(log_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            self.generate_histograms()
        except Exception as e:
            print(f"Error logging stats: {e}")

# Global stats monitor instance
stats_monitor = StatsMonitor()

def calculate_performance_stats():
    """Calculate comprehensive performance statistics"""
    stats = stats_monitor.get_performance_metrics()
    
    return {
        "total_processing_time": time.time() - stats_monitor.start_time,
        "frames_processed": sum(stats_monitor.frame_counts.values()),
        "frames_per_second": stats.get('fps', 0),
        "system_accuracy": stats.get('accuracy', 0),
        "system_precision": stats.get('precision', 0),
        "system_recall": stats.get('recall', 0),
        "processing_times": stats.get('video_times', {}),
        "true_positives": stats.get('true_positives', 0),
        "false_positives": stats.get('false_positives', 0),
        "true_negatives": stats.get('true_negatives', 0),
        "false_negatives": stats.get('false_negatives', 0)
    }
import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()
    FP16 = DEVICE == "cuda"
    
    # Violence Detection
    VIOLENCE_MODEL = "r3d"  # "slowfast", "r3d", or "i3d"
    VIOLENCE_THRESH = 0.65
    CLIP_LENGTH = 32
    CLIP_STRIDE = 16
    
    # Missing Person
    FACE_THRESH = 0.65
    FRAME_INTERVAL = 60
    BATCH_SIZE = 16

    # Distributed Processing
    SPARK_ENABLED = True
    
    # Paths
    MODEL_CACHE = os.getenv("MODEL_CACHE", "./models")
    OUTPUT_DIR = "./Output"

    # Stats logging
    STATS_LOG_DIR = os.getenv("STATS_LOG_DIR", "./stats_logs")
    STATS_LOG_INTERVAL = 60  # seconds
    HISTOGRAM_BINS = 20
    PERFORMANCE_LOG = True

    # Default Spark config
    SPARK_CONF = {
        "app": "CCTVAnalysisSystem",
        "executor.memory": "4g",
        "driver.memory": "4g",
        "executor.cores": "8",
        "spark.driver.extraJavaOptions": "-Dio.netty.tryReflectionSetAccessible=true",
        "spark.executor.extraJavaOptions": "-Dio.netty.tryReflectionSetAccessible=true",
        "spark.memory.fraction": "0.8",
        "spark.memory.storageFraction": "0.3"
    }

    def __post_init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODEL_CACHE, exist_ok=True)

    def update_spark_conf(self, master, memory, cores, instances):
        self.SPARK_CONF.update({
            "master": master,
            "executor.memory": memory,
            "executor.cores": str(cores),
            "executor.instances": str(instances)
        })
    
config = Config()
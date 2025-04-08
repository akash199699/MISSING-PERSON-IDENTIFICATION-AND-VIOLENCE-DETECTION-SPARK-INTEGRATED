import subprocess
import time

def start_spark_service(service_type, master_url=None):
    """Start a Spark service (master or worker)"""
    base_cmd = r'C:\spark\bin\spark-class.cmd org.apache.spark.deploy.'
    cmd = base_cmd + (
        'master.Master' if service_type == 'master' 
        else f'worker.Worker {master_url}'
    )
    
    subprocess.Popen(cmd, shell=True)
    print(f"Spark {service_type.capitalize()} started.")
    time.sleep(5)  # Give it a few seconds to initialize

def start_spark_cluster(master_url="spark://172.18.111.237:7077"):
    """Start both Spark master and worker services"""
    start_spark_service('master')
    start_spark_service('worker', master_url)

# Example usage
if __name__ == "__main__":
    start_spark_cluster()
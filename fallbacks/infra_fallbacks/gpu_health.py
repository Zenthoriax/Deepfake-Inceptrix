import logging
import psutil

logger = logging.getLogger(__name__)

class GPUHealthMonitor:
    def __init__(self, fallback_to_cpu=True):
        self.gpu_available = True
        self.fallback_to_cpu = fallback_to_cpu
        self.consecutive_ooms = 0
        
    def check_health(self):
        # In reality, this would query NVML
        # For our mock, we just return the boolean state
        return self.gpu_available
        
    def report_oom(self):
        self.consecutive_ooms += 1
        logger.error(f"GPU OOM reported. Count: {self.consecutive_ooms}")
        
        if self.consecutive_ooms >= 3:
            self.gpu_available = False
            logger.critical("GPU marked as UNAVAILABLE due to repeated OOMs")
            
    def reset_health(self):
        self.gpu_available = True
        self.consecutive_ooms = 0
        logger.info("GPU health reset to AVAILABLE")
        
    def get_worker_device(self):
        if self.check_health():
            return "cuda"
            
        if self.fallback_to_cpu:
            logger.warning("Fell back to CPU worker")
            return "cpu"
            
        raise Exception("GPU unavailable and CPU fallback is disabled")

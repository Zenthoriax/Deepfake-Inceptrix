from enum import IntEnum
import logging

logger = logging.getLogger(__name__)

class GracefulMode(IntEnum):
    LEVEL_0 = 0  # Full Intelligence Mode (Heavy models, complete pipelines)
    LEVEL_1 = 1  # Reduced Model Mode (Bypass heavy feature extraction)
    LEVEL_2 = 2  # Core Models Only (Faster inference tiers)
    LEVEL_3 = 3  # Emergency Detection Only (Static responses or lightweight heuristics only)

class SystemStressMonitor:
    def __init__(self):
        self.current_mode = GracefulMode.LEVEL_0
        self.cpu_load = 0.0
        self.gpu_load = 0.0
        self.queue_depth = 0
        
    def evaluate_stress(self, cpu_load, gpu_load, queue_depth):
        self.cpu_load = cpu_load
        self.gpu_load = gpu_load
        self.queue_depth = queue_depth
        
        # Calculate intended mode based on thresholds
        new_mode = GracefulMode.LEVEL_0
        
        if self.gpu_load > 0.95 or self.queue_depth > 1000:
            new_mode = GracefulMode.LEVEL_3
        elif self.gpu_load > 0.85 or self.queue_depth > 500:
            new_mode = GracefulMode.LEVEL_2
        elif self.gpu_load > 0.70 or self.queue_depth > 200:
            new_mode = GracefulMode.LEVEL_1
            
        if new_mode != self.current_mode:
            self._transition_mode(new_mode)
            
        return self.current_mode
        
    def _transition_mode(self, new_mode):
        logger.warning(f"System transitioning from {self.current_mode.name} to {new_mode.name} due to infra stress.")
        self.current_mode = new_mode

    def should_skip_pipeline(self, pipeline_criticality):
        """Returns True if a pipeline should be skipped based on current GracefulMode."""
        # Assume pipeline_criticality comes from pipelin_fallbacks/criticality_map.py (CRITICAL/IMPORTANT/OPTIONAL)
        if self.current_mode >= GracefulMode.LEVEL_1 and pipeline_criticality == "OPTIONAL":
            return True
        if self.current_mode >= GracefulMode.LEVEL_2 and pipeline_criticality == "IMPORTANT":
            return True
        return False
        
    def get_model_tier_cap(self):
        """Returns the maximum model tier allowed (0=primary, 1=secondary, 2=emergency)"""
        if self.current_mode >= GracefulMode.LEVEL_3:
            return 2 # Force emergency
        if self.current_mode >= GracefulMode.LEVEL_2:
            return 1 # Force secondary or emergency
        return 0 # Allow primary

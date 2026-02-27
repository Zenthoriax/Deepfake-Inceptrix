from enum import Enum

class Criticality(Enum):
    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT"
    OPTIONAL = "OPTIONAL"

PIPELINE_CRITICALITY_MAP = {
    # 1. Request Validation
    "Request Validation": Criticality.CRITICAL,
    "Authentication": Criticality.CRITICAL,
    "Input Normalization": Criticality.CRITICAL,
    
    # 2. Frame Sampling
    "Frame Sampling": Criticality.CRITICAL,
    "Face Detection": Criticality.CRITICAL,
    "Face Alignment": Criticality.IMPORTANT,
    "Face Tracking": Criticality.IMPORTANT,
    
    # 3. Features
    "Texture Artifact": Criticality.IMPORTANT,
    "Frequency Domain": Criticality.IMPORTANT,
    "Temporal Consistency": Criticality.IMPORTANT,
    "rPPG Signal": Criticality.OPTIONAL,
    "Head Pose": Criticality.OPTIONAL,
    "Eye Blink": Criticality.OPTIONAL,
    "Lip Sync": Criticality.IMPORTANT,
    "Metadata Forensics": Criticality.OPTIONAL,
    
    # 4. Model Inference
    "Model Inference": Criticality.CRITICAL,
    
    # 5. Orchestration
    "Orchestration": Criticality.CRITICAL,
    
    # 6. Fusion & Calibration
    "Fusion & Calibration": Criticality.CRITICAL,
    
    # 7. Forensic Intelligence
    "Forensic Intelligence": Criticality.IMPORTANT,
    
    # 8. Report Generation
    "Report Generation": Criticality.IMPORTANT,
    
    # 9. Logging
    "Logging": Criticality.OPTIONAL,
    
    # 10. Drift Detection
    "Drift Detection": Criticality.OPTIONAL
}

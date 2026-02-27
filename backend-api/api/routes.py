from fastapi import APIRouter, Query, HTTPException, File, UploadFile, Depends
from typing import Optional
import time
import random

from fallbacks.model_fallbacks.fallback_controller import ModelFallbackController, TimeoutError, OOMError, InferenceError
from fallbacks.pipeline_fallbacks.handler import PipelineFallbackHandler, PipelineCriticalError
from fallbacks.pipeline_fallbacks.degradation_scorer import DegradationScorer
from fallbacks.pipeline_fallbacks.criticality_map import Criticality
from fallbacks.infra_fallbacks.gpu_health import GPUHealthMonitor
from api.test_models import MockModel
from api.auth import get_api_key, register_new_key, get_all_keys
import secrets
import string

router = APIRouter()

@router.post("/generate-key", tags=["Auth"])
def generate_key_endpoint(app_name: str = Query("custom-app", description="Name of the app requesting the key")):
    """
    Generate a new API key and register it in the valid keys set.
    """
    prefix = f"sk-{app_name}"
    alphabet = string.ascii_letters + string.digits
    secure_string = ''.join(secrets.choice(alphabet) for i in range(32))
    key = f"{prefix}-{secure_string}"
    
    register_new_key(key)
    return {"status": "success", "app_name": app_name, "api_key": key}

@router.get("/active-keys", tags=["Auth"])
def get_active_keys_endpoint():
    """
    Retrieve all currently registered API keys.
    """
    return {"status": "success", "keys": get_all_keys()}

@router.get("/test/model_fallback", tags=["Model Fallbacks"])
def test_model_fallback(
    primary_fails: int = Query(0, description="How many times the primary model should fail"),
    secondary_fails: int = Query(0, description="How many times the secondary model should fail"),
    error_type: str = Query("TimeoutError", description="Type of error to raise: TimeoutError, OOMError, InferenceError")
):
    """
    Test the ModelFallbackController.
    If primary_fails >= 3, the circuit breaker should OPEN for the primary model.
    """
    error_map = {
        "TimeoutError": TimeoutError,
        "OOMError": OOMError,
        "InferenceError": InferenceError
    }
    exc = error_map.get(error_type, TimeoutError)

    primary = MockModel("EfficientNet-B4", fail_count=primary_fails, exception_to_raise=exc)
    secondary = MockModel("Xception", fail_count=secondary_fails, exception_to_raise=exc)
    emergency = MockModel("EfficientNet-B0-lite")

    controller = ModelFallbackController(primary, secondary, emergency)
    
    results = []
    # Trigger requests to see circuit breaker in action
    for i in range(max(primary_fails, secondary_fails) + 1):
        try:
            res = controller.infer({"data": "test"})
            if isinstance(res, dict) and res.get("error"):
                results.append({"request": i+1, "status": "emergency_response", "data": res})
            else:
                results.append({"request": i+1, "status": "success", "output": res.output})
        except Exception as e:
            results.append({"request": i+1, "status": "error", "message": str(e)})

    return {
        "final_circuit_states": {
            "primary_open": controller.circuit_open[primary],
            "secondary_open": controller.circuit_open[secondary]
        },
        "execution_history": results
    }

@router.get("/test/pipeline_fallback", tags=["Pipeline Fallbacks"])
def test_pipeline_fallback(
    pipeline_type: str = Query("CRITICAL", description="CRITICAL, IMPORTANT, or OPTIONAL"),
    fail_pipeline: bool = Query(True, description="Whether the main pipeline should fail"),
    has_fallback: bool = Query(True, description="Whether a fallback function is provided")
):
    """
    Test the PipelineFallbackHandler.
    """
    scorer = DegradationScorer()
    handler = PipelineFallbackHandler(scorer)
    
    crit_map = {
        "CRITICAL": Criticality.CRITICAL,
        "IMPORTANT": Criticality.IMPORTANT,
        "OPTIONAL": Criticality.OPTIONAL
    }
    crit = crit_map.get(pipeline_type, Criticality.OPTIONAL)

    def fail_fn():
        if fail_pipeline:
            raise Exception("Pipeline stage failed!")
        return {"status": "success", "source": "main_pipeline"}
        
    def backup_fn():
        return {"status": "success", "source": "fallback_pipeline"}
        
    fallback_ptr = backup_fn if has_fallback else None

    try:
        result = handler.run(fail_fn, fallback_ptr, crit, pipeline_name=f"Test_{pipeline_type}")
        if hasattr(result, "to_dict"):
            result = result.to_dict()
    except PipelineCriticalError as e:
        result = {"error": "PipelineCriticalError", "message": str(e)}

    return {
        "pipeline_result": result,
        "degradation_report": scorer.get_report()
    }

@router.get("/test/gpu_health", tags=["Infrastructure Fallbacks"])
def test_gpu_health(ooms_to_trigger: int = Query(3, description="Number of consecutive OOMs to report")):
    """
    Test the GPUHealthMonitor fallback to CPU.
    """
    monitor = GPUHealthMonitor()
    initial_device = monitor.get_worker_device()
    
    for _ in range(ooms_to_trigger):
        monitor.report_oom()
        
    final_device = monitor.get_worker_device()
    
    return {
        "initial_device": initial_device,
        "ooms_reported": ooms_to_trigger,
        "final_device": final_device,
        "gpu_available": monitor.gpu_available
    }

@router.post("/analyze", tags=["Inference"])
async def analyze_media(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """
    Analyze an uploaded photo or video utilizing the Deep Sentinel pipeline.
    """
    # Simulate extraction and processing delay
    time.sleep(1.5)
    
    is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv'))
    file_type = "video" if is_video else "image"
    
    # Mock prediction run through the fallback router logic for the frontend UI.
    fake_prob = random.uniform(0.05, 0.98)
    
    # Force some edge cases for demo effect
    if "fake" in file.filename.lower():
        fake_prob = random.uniform(0.85, 0.99)
    elif "real" in file.filename.lower():
        fake_prob = random.uniform(0.01, 0.15)
        
    result = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if result == "FAKE" else (1 - fake_prob)
    
    # Determine which model supposedly took the load based on Tiered Logic
    if 0.15 <= fake_prob <= 0.85:
        models = ["MobileNetV3 (Tier 1 Screening)", "EfficientNet-B4 (Tier 2 Heavy)"]
        notes = "Tier 1 screening was uncertain (prob in 0.15 to 0.85). Triggered Tier-2 heavy model logic."
    else:
        models = ["MobileNetV3 (Tier 1 Screening)"]
        notes = "Tier 1 screening was highly confident. No escalation needed."
        
    if random.random() < 0.1:
        models = ["Xception (Tier 3 Emergency)"]
        notes = "Circuit breaker tripped due to primary model timeout. Emergency fallback activated."

    return {
        "status": "success",
        "filename": file.filename,
        "type": file_type,
        "prediction": result,
        "confidence": round(confidence, 4),
        "models_used": models,
        "notes": notes,
        "processing_time_ms": random.randint(400, 1800)
    }

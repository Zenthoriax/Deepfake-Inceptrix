import logging

logger = logging.getLogger(__name__)

class PredictionResult:
    def __init__(self, output, confidence, extra_flags=None):
        self.output = output
        self.confidence = confidence
        self.flags = extra_flags or []

def fuse(results, strategy='weighted'):
    # A placeholder fusion implementation
    if not results:
        return PredictionResult(None, 0.0)
    
    if len(results) == 1:
        return results[0]
        
    valid_results = [r for r in results if r.confidence >= 0.40]
    
    if not valid_results:
        # All results were low confidence, return the best one anyway
        best_result = max(results, key=lambda r: r.confidence)
        best_result.flags.append("low_confidence_fusion")
        return best_result

    # Simple weighted average
    total_conf = sum(r.confidence for r in valid_results)
    if total_conf == 0:
        return valid_results[0]
        
    # Assuming output is a dict with numerical values
    # For now, just return highest confidence if we can't properly mathematical fuse
    best_result = max(valid_results, key=lambda r: r.confidence)
    
    # Add flag
    best_result.flags.append("fused_output")
    
    return best_result

class ConfidenceRouter:
    def __init__(self, confidence_threshold=0.65, absolute_min_threshold=0.40):
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.ABSOLUTE_MIN_THRESHOLD = absolute_min_threshold

    def route_and_fuse(self, input_data, primary_model, secondary_model):
        if not primary_model:
            raise ValueError("Primary model is required")

        result = primary_model.predict(input_data)
        
        # If confidence is good, return immediately
        if result.confidence >= self.CONFIDENCE_THRESHOLD:
            return result

        logger.info(f"Primary model returned low confidence ({result.confidence}). Invoking secondary.")
        
        # Secondary model fallback
        results_to_fuse = []
        if result.confidence >= self.ABSOLUTE_MIN_THRESHOLD:
            results_to_fuse.append(result)
        else:
            result.flags.append("excluded_from_fusion_due_to_low_confidence")
            
        if secondary_model:
            secondary_result = secondary_model.predict(input_data)
            if secondary_result.confidence >= self.ABSOLUTE_MIN_THRESHOLD:
                results_to_fuse.append(secondary_result)
            
        if not results_to_fuse:
            # Both failed threshold, return primary with flagged low confidence
            result.flags.append("all_models_low_confidence")
            return result
            
        final_result = fuse(results_to_fuse, strategy='weighted')
        
        if final_result.confidence < self.CONFIDENCE_THRESHOLD:
            final_result.flags.append("final_low_confidence")
            
        return final_result

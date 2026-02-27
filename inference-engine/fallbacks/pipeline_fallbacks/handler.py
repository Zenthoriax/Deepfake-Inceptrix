import logging
from .criticality_map import Criticality
from .null_signal import NullSignal

logger = logging.getLogger(__name__)

class PipelineCriticalError(Exception):
    pass

class PipelineFallbackHandler:
    def __init__(self, degradation_scorer):
        self.scorer = degradation_scorer

    def run(self, pipeline_fn, fallback_fn, criticality, pipeline_name=None):
        name = pipeline_name or pipeline_fn.__name__
        try:
            return pipeline_fn()
        except Exception as e:
            self._log_pipeline_error(name, e)
            
            if criticality == Criticality.CRITICAL:
                if fallback_fn:
                    self.scorer.record_degraded(name, "critical_fallback_used")
                    return fallback_fn()
                raise PipelineCriticalError(f"Critical pipeline {name} failed and has no fallback: {e}")
                
            elif criticality == Criticality.IMPORTANT:
                if fallback_fn:
                    self.scorer.record_degraded(name, "important_fallback_used")
                    return fallback_fn()
                else:
                    self.scorer.record_missing(name)
                    return NullSignal(reason=str(e))
                    
            else:  # OPTIONAL
                self.scorer.record_missing(name)
                return NullSignal(reason=str(e))  # skip gracefully

    def _log_pipeline_error(self, name, error):
        logger.error(f"Pipeline {name} failed with error: {error}")

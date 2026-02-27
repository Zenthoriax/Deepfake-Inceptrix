import time
import logging

class TimeoutError(Exception): pass
class OOMError(Exception): pass
class InferenceError(Exception): pass

class ModelFallbackController:
    def __init__(self, primary, secondary, emergency, timeout=2.0, threshold=3):
        self.chain = [primary, secondary, emergency]
        self.circuit_open = {m: False for m in self.chain}
        self.failure_counts = {m: 0 for m in self.chain}
        self.THRESHOLD = threshold
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def infer(self, input_data):
        for model in self.chain:
            if self.circuit_open[model]:
                continue
            try:
                # We assume model.predict expects a timeout parameter
                result = model.predict(input_data, timeout=self.timeout)
                self.failure_counts[model] = 0  # reset on success
                return result
            except (TimeoutError, OOMError, InferenceError) as e:
                self._handle_failure(model, e)
            except Exception as e:
                self.logger.error(f"Unexpected error in model {model}: {e}")
                self._handle_failure(model, e)

        return self._emergency_static_response()

    def _handle_failure(self, model, error):
        self.failure_counts[model] += 1
        self.logger.warning(f"Model {model} failed with error: {error}. Failure count: {self.failure_counts[model]}")
        if self.failure_counts[model] >= self.THRESHOLD:
            self.circuit_open[model] = True
            self.logger.error(f"Circuit opened for model {model} after {self.failure_counts[model]} failures.")

    def _emergency_static_response(self):
        return {
            "probability_fake": None,
            "error": "inference_unavailable",
            "retry_after": 30
        }

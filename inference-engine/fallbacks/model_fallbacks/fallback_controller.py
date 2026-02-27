import time
import logging
from collections import deque

class TimeoutError(Exception): pass
class OOMError(Exception): pass
class InferenceError(Exception): pass

class CircuitState:
    CLOSED = "CLOSED"       # Healthy - all requests pass
    OPEN = "OPEN"           # Unhealthy - fast fail / bypass
    HALF_OPEN = "HALF_OPEN" # Probing - let 1 request pass to check health

class ModelFallbackController:
    def __init__(self, primary, secondary, emergency, 
                 timeout=2.0, 
                 failure_rate_threshold=0.5, # 50% failure rate triggers OPEN
                 window_size=10,             # Last 10 requests tracked
                 cooldown_seconds=30,        # Time to wait before HALF_OPEN
                 max_latency_ms=800):        # Latency > threshold triggers degrade
        self.chain = [primary, secondary, emergency]
        self.timeout = timeout
        self.failure_rate_threshold = failure_rate_threshold
        self.window_size = window_size
        self.cooldown_seconds = cooldown_seconds
        self.max_latency_ms = max_latency_ms
        self.logger = logging.getLogger(__name__)

        # State management per model
        self.states = {m: CircuitState.CLOSED for m in self.chain}
        
        # Rolling window history (using 1 for success, 0 for failure)
        self.history = {m: deque(maxlen=self.window_size) for m in self.chain}
        
        # Track when circuit was opened for cooldown
        self.last_failure_time = {m: 0 for m in self.chain}

    def infer(self, input_data):
        for i, model in enumerate(self.chain):
            self._evaluate_state(model)
            
            if self.states[model] == CircuitState.OPEN:
                continue
                
            start_time = time.time()
            try:
                # We assume model.predict expects a timeout parameter
                result = model.predict(input_data, timeout=self.timeout)
                latency_ms = (time.time() - start_time) * 1000
                
                self._record_success(model)
                
                # Latency-Aware Routing
                # If latency is too high, we don't return failure, but we log a warning
                # and if there is a next model in chain, we might offload future requests.
                # For this request, we got a result, so return it but add latency flag.
                if latency_ms > self.max_latency_ms:
                    self.logger.warning(f"Model {model.name} SLA breach: {latency_ms:.1f}ms (threshold: {self.max_latency_ms}ms). Degrading future tier weight if possible.")
                    if not hasattr(result, 'flags'):
                        result.flags = []
                    result.flags.append(f"high_latency_{latency_ms:.0f}ms")
                    
                    # If this is a probing request in HALF_OPEN, a slow response might 
                    # send it back to OPEN depending on strictness. Here we keep it CLOSED
                    # but maybe penalize.
                    
                return result
                
            except (TimeoutError, OOMError, InferenceError) as e:
                self._record_failure(model, e)
            except Exception as e:
                self.logger.error(f"Unexpected error in model {model.name}: {e}")
                self._record_failure(model, e)

        return self._emergency_static_response()

    def _evaluate_state(self, model):
        state = self.states[model]
        if state == CircuitState.OPEN:
            # Check if cooldown has expired => switch to HALF_OPEN
            if time.time() - self.last_failure_time[model] > self.cooldown_seconds:
                self.states[model] = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit for {model.name} moved to HALF_OPEN (probing).")

    def _record_success(self, model):
        self.history[model].append(1)
        if self.states[model] == CircuitState.HALF_OPEN:
            # Probe succeeded! Reset circuit
            self.states[model] = CircuitState.CLOSED
            self.history[model].clear() # Reset window
            self.logger.info(f"Circuit for {model.name} recovered and is now CLOSED.")

    def _record_failure(self, model, error):
        self.history[model].append(0)
        self.last_failure_time[model] = time.time()
        
        self.logger.warning(f"Model {model.name} failed with error: {error}.")
        
        if self.states[model] == CircuitState.HALF_OPEN:
            # Probe failed! Back to OPEN
            self.states[model] = CircuitState.OPEN
            self.logger.error(f"Probe failed. Circuit for {model.name} remaining OPEN.")
            return
            
        # Calculate failure rate if window is full
        if len(self.history[model]) == self.window_size:
            failures = self.window_size - sum(self.history[model])
            failure_rate = failures / self.window_size
            
            if failure_rate >= self.failure_rate_threshold:
                self.states[model] = CircuitState.OPEN
                self.logger.error(f"Failure rate {failure_rate*100:.1f}% exceeded threshold. Circuit for {model.name} moved to OPEN.")

    def _emergency_static_response(self):
        return {
            "probability_fake": None,
            "error": "inference_unavailable",
            "retry_after": self.cooldown_seconds
        }

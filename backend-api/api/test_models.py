from pydantic import BaseModel
from typing import Optional, List

# Simulating model prediction result
class MockPredictionResult:
    def __init__(self, output, confidence, flags=None):
        self.output = output
        self.confidence = confidence
        self.flags = flags or []

class MockModel:
    def __init__(self, name, fail_count=0, exception_to_raise=None, confidence=0.8):
        self.name = name
        self.fail_count = fail_count
        self.current_fails = 0
        self.exception_to_raise = exception_to_raise
        self.confidence = confidence

    def predict(self, input_data, timeout=2.0):
        if self.current_fails < self.fail_count:
            self.current_fails += 1
            if self.exception_to_raise:
                raise self.exception_to_raise(f"{self.name} failed")
            raise Exception(f"{self.name} failed")
        return MockPredictionResult(f"Output from {self.name}", self.confidence)

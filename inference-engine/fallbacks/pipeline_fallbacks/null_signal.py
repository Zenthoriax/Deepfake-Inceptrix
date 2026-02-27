class NullSignal:
    def __init__(self, reason="Pipeline skipped"):
        self.is_null = True
        self.reason = reason
        self.data = None
        self.confidence = 0.0
        
    def __bool__(self):
        return False
        
    def to_dict(self):
        return {
            "signal": None,
            "status": "missing",
            "reason": self.reason
        }

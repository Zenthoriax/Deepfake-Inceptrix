class DegradationScorer:
    def __init__(self):
        self.missing_signals = []
        self.degraded_signals = []
        
        # We can tune these penalties
        self.MISSING_PENALTY = 10
        self.DEGRADED_PENALTY = 5

    def record_missing(self, signal_name):
        self.missing_signals.append(signal_name)

    def record_degraded(self, signal_name, reason):
        self.degraded_signals.append(f"{signal_name} ({reason})")

    def get_score(self):
        score = (len(self.missing_signals) * self.MISSING_PENALTY) + \
                (len(self.degraded_signals) * self.DEGRADED_PENALTY)
        # Cap score at 100
        return min(score, 100)
        
    def get_confidence_adjustment(self):
        score = self.get_score()
        # Scale: max -0.50 adjustment for score of 100
        return -(score / 100.0) * 0.50

    def get_report(self):
        return {
            "degradation_score": self.get_score(),
            "missing_signals": self.missing_signals,
            "degraded_signals": self.degraded_signals,
            "confidence_adjustment": round(self.get_confidence_adjustment(), 2)
        }

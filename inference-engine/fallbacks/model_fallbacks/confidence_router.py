import logging
import math

logger = logging.getLogger(__name__)

class PredictionResult:
    def __init__(self, output, probability_fake=None, logit_real=None, logit_fake=None, extra_flags=None):
        self.output = output
        self.probability_fake = probability_fake
        self.logit_real = logit_real
        self.logit_fake = logit_fake
        self.flags = extra_flags or []
        
        # Calculate uncertainty based on logit margin if available, else probability
        if logit_real is not None and logit_fake is not None:
            margin = abs(logit_real - logit_fake)
            # Smaller margin -> higher uncertainty.
            self.uncertainty = math.exp(-margin)
        elif probability_fake is not None:
            # 0.5 is max uncertainty (1.0). 0 or 1 is min uncertainty (0.0).
            self.uncertainty = 1.0 - 2.0 * abs(probability_fake - 0.5)
        else:
            self.uncertainty = 0.0

    @property
    def needs_escalation(self):
        """
        Determines if the result is uncertain enough to require escalation to a heavier model.
        Following the rule:
        If probability > 0.85 → Confident FAKE (Return)
        If probability < 0.15 → Confident REAL (Return)
        If 0.15 <= probability <= 0.85 → Uncertain (Escalate)
        """
        if self.probability_fake is not None:
            return 0.15 <= self.probability_fake <= 0.85
            
        # If we only have logits, calculate probability via softmax or check margin
        if self.logit_real is not None and self.logit_fake is not None:
            margin = abs(self.logit_real - self.logit_fake)
            # A margin < 1.5 roughly correlates to the 0.15 - 0.85 probability window depending on temperature
            return margin < 1.7
            
        # Fallback to general uncertainty score if neither are populated
        return self.uncertainty > 0.3


class DynamicConfidenceRouter:
    """
    Tier-based Model Router.
    Tier 1 runs first (e.g., MobileNetV3). If uncertain, it escalates to Tier 2 (e.g., EfficientNet-B4).
    """
    def __init__(self, lower_bound=0.15, upper_bound=0.85):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def route_and_fuse(self, input_data, primary_model, secondary_model):
        if not primary_model:
            raise ValueError("Primary model is required")

        # Tier 1 execution (Fast screening model)
        result = primary_model.predict(input_data)
        
        # Check if we need to escalate based on strict confidence logit margins / probabilities
        if not result.needs_escalation:
            prob_str = f"{result.probability_fake:.3f}" if result.probability_fake is not None else "N/A"
            logger.info(f"Tier 1 model is confident (prob: {prob_str}). Returning without escalation.")
            return result

        prob_str = f"{result.probability_fake:.3f}" if result.probability_fake is not None else "N/A"
        logger.info(f"Tier 1 model is UNCERTAIN (prob: {prob_str}). Escalating to Tier 2 heavy model.")
        result.flags.append("escalated_to_tier2")
        
        # Tier 2 execution (Heavy precision model)
        if secondary_model:
            secondary_result = secondary_model.predict(input_data)
            secondary_result.flags.append("from_tier2_escalation")
            
            # The heavier model is the final decider. We don't blur the lines by fusing a confident 
            # heavy model with an uncertain lightweight model. We just trust the heavy model.
            return secondary_result
            
        # If no secondary model is loaded for some reason, return the uncertain primary
        return result

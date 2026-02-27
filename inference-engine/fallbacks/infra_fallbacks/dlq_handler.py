import logging
import json

logger = logging.getLogger(__name__)

class DeadLetterQueueHandler:
    def __init__(self, dlq_file="logs/dlq.jsonl"):
        self.dlq_file = dlq_file
        
    def process_failed_message(self, message, error_reason):
        try:
            dlq_entry = {
                "message": message,
                "error": str(error_reason),
                "status": "manual_review_required"
            }
            # Append to a dead letter log file (simulating a DLQ)
            with open(self.dlq_file, 'a') as f:
                f.write(json.dumps(dlq_entry) + "\n")
            logger.info(f"Message moved to DLQ: {error_reason}")
        except Exception as e:
            logger.error(f"Failed to write to DLQ: {e}")

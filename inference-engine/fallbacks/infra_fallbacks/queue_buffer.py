import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

class QueueBuffer:
    def __init__(self, max_seconds=120):
        self.buffer = deque()
        self.max_seconds = max_seconds
        
    def enqueue(self, job):
        # job should be a dict with at least 'timestamp' and 'payload'
        job['timestamp'] = time.time()
        self.buffer.append(job)
        logger.info(f"Job buffered. Buffer size: {len(self.buffer)}")
        self._evict_stale()
        
    def dequeue_all(self):
        self._evict_stale()
        jobs = list(self.buffer)
        self.buffer.clear()
        return jobs
        
    def _evict_stale(self):
        current_time = time.time()
        evicted_count = 0
        while self.buffer and (current_time - self.buffer[0]['timestamp']) > self.max_seconds:
            self.buffer.popleft()
            evicted_count += 1
            
        if evicted_count > 0:
            logger.warning(f"Evicted {evicted_count} stale jobs from buffer")

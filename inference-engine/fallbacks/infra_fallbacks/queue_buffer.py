import time
import json
import logging

logger = logging.getLogger(__name__)

class RedisQueueBuffer:
    def __init__(self, redis_client=None, queue_name="deep_sentinel:fallback_queue", max_seconds=120):
        # In a real setup, redis_client would be an initialized redis.Redis instance
        self.redis = redis_client
        self.queue_name = queue_name
        self.max_seconds = max_seconds
        
    def enqueue(self, job):
        if not self.redis:
            logger.error("No Redis client configured for persistent queue. Dropping message.")
            return False
            
        job['timestamp'] = time.time()
        job_data = json.dumps(job)
        
        try:
            # LPUSH to left of list
            self.redis.lpush(self.queue_name, job_data)
            logger.info("Job durably buffered in Redis queue.")
            self._evict_stale()
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue to Redis: {e}")
            return False
        
    def dequeue_all(self):
        if not self.redis:
            return []
            
        self._evict_stale()
        jobs = []
        try:
            # Pop all items using pipeline or while list has items
            while True:
                item = self.redis.rpop(self.queue_name)
                if not item:
                    break
                jobs.append(json.loads(item))
            return jobs
        except Exception as e:
            logger.error(f"Failed to dequeue from Redis: {e}")
            return jobs
        
    def _evict_stale(self):
        if not self.redis: return
        
        current_time = time.time()
        evicted_count = 0
        try:
            # In a real large-scale system, we'd use ZSETs for timestamp eviction
            # For this simple list-based queue, we peek from right (oldest)
            while True:
                # LINDEX gets item without removing
                oldest_item_raw = self.redis.lindex(self.queue_name, -1)
                if not oldest_item_raw:
                    break
                    
                oldest_item = json.loads(oldest_item_raw)
                if (current_time - oldest_item.get('timestamp', 0)) > self.max_seconds:
                    self.redis.rpop(self.queue_name)
                    evicted_count += 1
                else:
                    break # Oldest item is still fresh, others must be too
                    
            if evicted_count > 0:
                logger.warning(f"Evicted {evicted_count} stale jobs from Redis buffer")
        except Exception as e:
            logger.error(f"Error during Redis stale eviction: {e}")

# Maintain interface compatibility for existing code imports
QueueBuffer = RedisQueueBuffer

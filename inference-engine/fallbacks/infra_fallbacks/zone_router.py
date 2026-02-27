import logging
import time

logger = logging.getLogger(__name__)

class ZoneRouter:
    def __init__(self, primary_zone='us-east-1a', zones=None):
        self.primary_zone = primary_zone
        self.zones = zones or ['us-east-1a', 'us-east-1b', 'us-west-2a']
        self.zone_health = {z: True for z in self.zones}
        # Simulate active zone
        self.active_zone = primary_zone
        
    def check_health(self, zone):
        # In a real system, this would ping a health endpoint
        return self.zone_health.get(zone, False)
        
    def mark_zone_down(self, zone):
        self.zone_health[zone] = False
        logger.error(f"Zone {zone} marked as DOWN")
        self._failover()
        
    def mark_zone_up(self, zone):
        self.zone_health[zone] = True
        logger.info(f"Zone {zone} marked as UP")
        # Try to failback to primary if it's up
        if zone == self.primary_zone and self.active_zone != self.primary_zone:
            self._failback()
            
    def _failover(self):
        for zone in self.zones:
            if self.zone_health.get(zone, False):
                self.active_zone = zone
                logger.warning(f"Failed over to zone: {self.active_zone}")
                return
        logger.critical("ALL ZONES DOWN")
        self.active_zone = None
        
    def _failback(self):
        self.active_zone = self.primary_zone
        logger.info(f"Failed back to primary zone: {self.active_zone}")
        
    def route_request(self, request):
        if not self.active_zone:
            raise Exception("No healthy zones available to route request")
        
        # Route to active
        request.routed_zone = self.active_zone
        return request

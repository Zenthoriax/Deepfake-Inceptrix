from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
import os

# Define the standard header name external apps must use
API_KEY_NAME = "x-deep-sentinel-api-key"

# The header extraction dependency
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# A mock dict of valid keys mapped to their call usage counts
# You can set the primary key via environment variable, or fallback to a default test key.
VALID_API_KEYS = {
    os.environ.get("DEEP_SENTINEL_ROOT_KEY", "sk-deep-sentinel-test-key-992x"): 0,
    "sk-mobile-app-client-v1": 0,
    "sk-web-dashboard-client-v1": 0
}

def register_new_key(key: str):
    VALID_API_KEYS[key] = 0

def get_all_keys():
    return [{"key": k, "usage": v} for k, v in VALID_API_KEYS.items()]

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency function to validate API keys on protected routes.
    """
    if api_key in VALID_API_KEYS:
        VALID_API_KEYS[api_key] += 1
        return api_key
        
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate API credentials. Please provide a valid 'x-deep-sentinel-api-key' header.",
    )

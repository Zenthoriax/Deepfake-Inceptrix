import secrets
import string

def generate_api_key(prefix="sk-sentinel"):
    """
    Generate a secure, random API key for external applications.
    Format: prefix-random_string
    """
    alphabet = string.ascii_letters + string.digits
    secure_string = ''.join(secrets.choice(alphabet) for i in range(32))
    key = f"{prefix}-{secure_string}"
    return key

if __name__ == "__main__":
    print("========================================")
    print(" DEEP SENTINEL API KEY GENERATOR")
    print("========================================\n")
    
    app_name = input("Enter the name of the external app needing access (e.g. 'mobile-client'): ").strip()
    prefix = f"sk-{app_name}" if app_name else "sk-sentinel"
    
    new_key = generate_api_key(prefix)
    
    print("\n✅ API Key Generated Successfully!")
    print(f"Key: {new_key}")
    print("\nTo use this key in production, add it to your VALID_API_KEYS set in api/auth.py")
    print("External apps must pass this key via the 'X-DeepSentinel-API-Key' HTTP Header.")
    print("----------------------------------------\n")

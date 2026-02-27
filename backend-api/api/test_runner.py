import requests
import json
import time

base_url = "http://127.0.0.1:8000"

def wait_for_server():
    for _ in range(10):
        try:
            requests.get(f"{base_url}/")
            return True
        except:
            time.sleep(1)
    return False

def run_tests():
    if not wait_for_server():
        print("Server failed to start")
        return

    print("=================== MODEL FALLBACK TEST ===================")
    # Test 1: Primary fails 3 times, should open circuit
    res = requests.get(f"{base_url}/test/model_fallback?primary_fails=3&secondary_fails=0")
    print(json.dumps(res.json(), indent=2))

    print("\n=================== PIPELINE FALLBACK TEST ===================")
    # Test 2: CRITICAL pipeline fails, fallback provided
    res1 = requests.get(f"{base_url}/test/pipeline_fallback?pipeline_type=CRITICAL&fail_pipeline=true&has_fallback=true")
    print("CRITICAL with fallback:")
    print(json.dumps(res1.json(), indent=2))
    
    # Test 3: IMPORTANT pipeline fails, no fallback (should give Degradation Score)
    res2 = requests.get(f"{base_url}/test/pipeline_fallback?pipeline_type=IMPORTANT&fail_pipeline=true&has_fallback=false")
    print("\nIMPORTANT without fallback:")
    print(json.dumps(res2.json(), indent=2))

    print("\n=================== INFRA Fallback TEST ===================")
    # Test 4: GPU OOMs 3 times
    res3 = requests.get(f"{base_url}/test/gpu_health?ooms_to_trigger=3")
    print(json.dumps(res3.json(), indent=2))

if __name__ == "__main__":
    run_tests()

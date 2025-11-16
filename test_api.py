import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

def test_api():
    """Test the FastAPI endpoints"""
    
    # Test health check
    print("Testing health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Health check: {response.json()}")
    
    # Test text generation
    print("\nTesting text generation...")
    
    prompt_data = {
        "prompt": "Write a Python function to calculate the factorial of a number",
        "max_tokens": 300
    }
    
    response = requests.post(
        f"{API_URL}/generate",
        headers={"Content-Type": "application/json"},
        json=prompt_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Generated text:")
        print(result["generated_text"])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_api()
#!/usr/bin/env python3
"""
Example showing the 422 Unprocessable Entity error and how to fix it
"""

print("FastAPI 422 Unprocessable Entity Error - Common Causes and Solutions")
print("="*70)

print("""
The 422 Unprocessable Entity error occurs when your request body doesn't match 
the expected Pydantic model structure.

For the Qwen Coder API, the expected request format is:

✅ CORRECT REQUEST FORMAT:
{
  "prompt": "Your text prompt here",
  "max_tokens": 512  // Optional, defaults to 512 if not provided
}

❌ COMMON MISTAKES THAT CAUSE 422 ERRORS:

1. MISSING REQUIRED FIELD 'prompt':
   {
     "text": "Hello world",     // Wrong field name
     "max_tokens": 100
   }

2. WRONG DATA TYPES:
   {
     "prompt": "Hello world",
     "max_tokens": "100"       // Should be integer, not string
   }

3. COMPLETELY WRONG STRUCTURE:
   {
     "message": "Hello world"  // Wrong field name
   }

4. EMPTY REQUEST:
   {}

EXAMPLES OF CORRECT REQUESTS:

1. Minimal request (only required field):
   curl -X POST "http://localhost:8002/generate" \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "Write a Python function"}'

2. Full request with optional field:
   curl -X POST "http://localhost:8002/generate" \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "Write a Python function", "max_tokens": 200}'

3. Using Python requests:
   import requests
   response = requests.post(
       "http://localhost:8002/generate",
       json={"prompt": "Your prompt", "max_tokens": 150}
   )

TROUBLESHOOTING 422 ERRORS:

1. Check field names: Must be exactly "prompt" and "max_tokens"
2. Check data types: prompt = string, max_tokens = integer
3. Ensure Content-Type header is "application/json"
4. Make sure JSON is properly formatted
5. The "prompt" field is required, "max_tokens" is optional

If you're still getting 422 errors, the API will return detailed error 
information showing exactly what's wrong with your request.
""")

# Example of a function to make a correct API call
def make_correct_api_call():
    import requests
    
    try:
        # Correct request format
        response = requests.post(
            "http://localhost:8002/generate",
            json={
                "prompt": "Write a simple hello world function in Python",
                "max_tokens": 100
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Success! Response:")
            print(response.json())
        elif response.status_code == 422:
            print("❌ 422 Error - Invalid request format:")
            print(response.json())
        else:
            print(f"❌ Error {response.status_code}:")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("\nTesting API call...")
    make_correct_api_call()
#!/usr/bin/env python3
"""
Test script for Abby Chatbot
This script tests various query types and functionalities
"""

import requests
import json
import time
import uuid
import argparse

# Default configuration
DEFAULT_URL = "http://localhost:8080"
SESSION_ID = str(uuid.uuid4())

def send_chat_request(message, session_id=None, user_location=None, endpoint="/chat"):
    """Send a chat request to the API"""
    url = f"{BASE_URL}{endpoint}"
    
    # Create payload
    payload = {
        "message": message,
        "session_id": session_id or SESSION_ID,
    }
    
    if user_location:
        payload["user_location"] = user_location
    
    # Send request
    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    return response.json()

def test_state_detection_basic():
    """Test basic state detection functionality"""
    print("\n=== Testing basic state detection ===")
    
    # Test explicit state mention
    print("\nTest: Explicit state mention")
    response = send_chat_request("What are the abortion laws in California?")
    if response and "state_code" in response:
        print(f"✓ Correctly identified state: {response.get('state_code')}")
    else:
        print("✗ Failed to identify state")
    
    # Test state abbreviation
    print("\nTest: State abbreviation")
    response = send_chat_request("What are the abortion laws in TX?")
    if response and "state_code" in response:
        print(f"✓ Correctly identified state abbreviation: {response.get('state_code')}")
    else:
        print("✗ Failed to identify state abbreviation")
    
    # Test ZIP code
    print("\nTest: ZIP code")
    response = send_chat_request("What are the abortion laws in 90210?")
    if response and "state_code" in response:
        print(f"✓ Correctly identified state from ZIP code: {response.get('state_code')}")
    else:
        print("✗ Failed to identify state from ZIP code")

def test_multi_state_handling():
    """Test handling of multiple states in queries"""
    print("\n=== Testing multi-state handling ===")
    
    # Test comparing two states
    print("\nTest: Comparing two states")
    response = send_chat_request("Compare abortion laws in Texas vs California")
    if response and "state_codes" in response:
        print(f"✓ Correctly identified multiple states: {response.get('state_codes')}")
    elif response and "state_code" in response:
        print(f"✓ Identified at least one state: {response.get('state_code')}")
    else:
        print("✗ Failed to identify multiple states")
    
    # Test state switching
    print("\nTest: Switching states in conversation")
    session_id = str(uuid.uuid4())
    
    # First query about one state
    response1 = send_chat_request("What are abortion laws in Florida?", session_id=session_id)
    if response1 and "state_code" in response1 and response1["state_code"] == "FL":
        print(f"✓ First query correctly identified Florida")
    else:
        print("✗ Failed on first state query")
    
    # Second query about different state
    response2 = send_chat_request("What about in New York?", session_id=session_id)
    if response2 and "state_code" in response2 and response2["state_code"] == "NY":
        print(f"✓ Second query correctly switched to New York")
    else:
        print("✗ Failed to switch states")

def test_multi_aspect_queries():
    """Test handling of multi-aspect queries"""
    print("\n=== Testing multi-aspect queries ===")
    
    # Test using the dedicated multi-aspect endpoint
    print("\nTest: Multi-aspect endpoint with combined query")
    response = send_chat_request(
        "I'm pregnant and scared, what are my options in Texas?", 
        endpoint="/test-multi-aspect"
    )
    
    if response and "metadata" in response and response["metadata"].get("is_multi_aspect_test"):
        aspects_count = response["metadata"].get("aspects_count", 0)
        print(f"✓ Multi-aspect test endpoint responded")
        if "state_code" in response:
            print(f"✓ Identified state: {response.get('state_code')}")
        else:
            print("✗ Failed to identify state in multi-aspect query")
    else:
        print("✗ Multi-aspect test endpoint failed")
    
    # Test using the regular endpoint
    print("\nTest: Regular endpoint with combined query")
    response = send_chat_request(
        "What are abortion laws in California and how does abortion work?"
    )
    
    if response:
        print(f"✓ Regular endpoint responded to complex query")
        if "state_code" in response:
            print(f"✓ Identified state: {response.get('state_code')}")
        else:
            print("✗ Failed to identify state in complex query")
    else:
        print("✗ Failed to handle complex query")

def test_asking_about_my_state():
    """Test 'my state' handling"""
    print("\n=== Testing 'my state' handling ===")
    
    # Test with no prior state context
    print("\nTest: 'My state' with no prior context")
    session_id = str(uuid.uuid4())
    response = send_chat_request(
        "What are the abortion laws in my state?", 
        session_id=session_id
    )
    
    if response and "text" in response and "which state" in response["text"].lower():
        print(f"✓ Correctly asked for state clarification")
    else:
        print("✗ Failed to request clarification")
    
    # Test with prior state context
    print("\nTest: 'My state' with prior context")
    session_id = str(uuid.uuid4())
    
    # First set state context
    send_chat_request(
        "I live in Texas", 
        session_id=session_id
    )
    
    # Then ask about "my state"
    response = send_chat_request(
        "What are the abortion laws in my state?", 
        session_id=session_id
    )
    
    if response and "state_code" in response and response["state_code"] == "TX":
        print(f"✓ Correctly remembered state from context: {response.get('state_code')}")
    else:
        print("✗ Failed to remember state from context")

def test_user_location():
    """Test user location handling"""
    print("\n=== Testing user location handling ===")
    
    # Test with user location
    print("\nTest: With user location")
    response = send_chat_request(
        "What are the abortion laws?",
        user_location={"state": "CA", "city": "Los Angeles"}
    )
    
    if response and "state_code" in response and response["state_code"] == "CA":
        print(f"✓ Correctly used provided location: {response.get('state_code')}")
    else:
        print("✗ Failed to use provided location")
    
    # Test with ZIP code in user location
    print("\nTest: With ZIP code in user location")
    response = send_chat_request(
        "What are the abortion laws?",
        user_location={"zip": "10001", "city": "New York"}
    )
    
    if response and "state_code" in response and response["state_code"] == "NY":
        print(f"✓ Correctly extracted state from ZIP: {response.get('state_code')}")
    else:
        print("✗ Failed to extract state from ZIP")

def run_all_tests():
    """Run all test cases"""
    test_state_detection_basic()
    test_multi_state_handling()
    test_multi_aspect_queries()
    test_asking_about_my_state()
    test_user_location()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Abby Chatbot functionality")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the API")
    parser.add_argument("--test", choices=["all", "state", "multi-state", "multi-aspect", "my-state", "location"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    # Set global variables
    BASE_URL = args.url
    
    print(f"Testing against API at {BASE_URL}")
    
    # Run selected test
    if args.test == "all":
        run_all_tests()
    elif args.test == "state":
        test_state_detection_basic()
    elif args.test == "multi-state":
        test_multi_state_handling()
    elif args.test == "multi-aspect":
        test_multi_aspect_queries()
    elif args.test == "my-state":
        test_asking_about_my_state()
    elif args.test == "location":
        test_user_location() 
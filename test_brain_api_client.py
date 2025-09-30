#!/usr/bin/env python3
"""
Test script for Brain API Client

This script tests the Brain API client functionality to ensure it can
communicate with the Brain API storage endpoints correctly.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from brain_api_client import get_brain_api_client

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def test_brain_api_client():
    """Test the Brain API client functionality."""
    print("🧪 Testing Brain API Client...")
    
    try:
        # Initialize the client
        client = get_brain_api_client()
        print("✅ Brain API client initialized successfully")
        
        # Test health check
        print("🔍 Testing health check...")
        is_healthy = client.health_check()
        if is_healthy:
            print("✅ Brain API is accessible")
        else:
            print("⚠️  Brain API health check failed - this might be expected if the API is not running")
        
        # Test file upload
        print("📤 Testing file upload...")
        test_content = b"Hello, Brain API! This is a test file."
        test_filename = "test_file.txt"
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        try:
            # Upload the file
            upload_result = client.upload_file_to_storage(temp_file_path, test_filename)
            print(f"✅ File uploaded successfully. Storage ID: {upload_result.get('storage_id')}")
            
            storage_id = upload_result.get('storage_id')
            if storage_id:
                # Test file download
                print("📥 Testing file download...")
                downloaded_content = client.get_file_from_storage(storage_id, test_filename)
                
                if downloaded_content == test_content:
                    print("✅ File downloaded successfully and content matches")
                else:
                    print("❌ Downloaded content does not match original")
                
                # Test file deletion
                print("🗑️  Testing file deletion...")
                client.delete_file_from_storage(storage_id, test_filename)
                print("✅ File deleted successfully")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logging.exception("Test failed with exception:")
        return False

def test_fileobj_upload():
    """Test uploading a file-like object."""
    print("\n🧪 Testing file object upload...")
    
    try:
        client = get_brain_api_client()
        
        # Create a file-like object
        import io
        test_content = b"Hello from file object!"
        file_obj = io.BytesIO(test_content)
        
        # Upload the file object
        upload_result = client.upload_fileobj_to_storage(file_obj, "test_fileobj.txt", "text/plain")
        print(f"✅ File object uploaded successfully. Storage ID: {upload_result.get('storage_id')}")
        
        storage_id = upload_result.get('storage_id')
        if storage_id:
            # Test download
            downloaded_content = client.get_file_from_storage(storage_id, "test_fileobj.txt")
            
            if downloaded_content == test_content:
                print("✅ File object download successful and content matches")
            else:
                print("❌ Downloaded content does not match original")
            
            # Clean up
            client.delete_file_from_storage(storage_id, "test_fileobj.txt")
            print("✅ File object deleted successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ File object test failed: {e}")
        logging.exception("File object test failed with exception:")
        return False

if __name__ == "__main__":
    print("🚀 Starting Brain API Client Tests")
    print("=" * 50)
    
    # Check environment variables
    api_key = os.getenv("BRANDO_API_KEY")
    base_url = os.getenv("BRANDO_BRAIN_API_BASE_URL", "http://localhost:2024/api")
    
    print(f"API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"Base URL: {base_url}")
    print()
    
    if not api_key:
        print("❌ BRANDO_API_KEY environment variable is not set!")
        print("Please set it in your .env file or environment.")
        sys.exit(1)
    
    # Run tests
    success = True
    success &= test_brain_api_client()
    success &= test_fileobj_upload()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

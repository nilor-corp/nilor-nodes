"""
Brain API Client for ComfyUI Nodes

This client provides methods to interact with the Brain API storage endpoints,
replacing the need for pre-signed URLs in the ComfyUI workflow.
"""

import requests
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BrainApiClient:
    """
    Client for interacting with Brain API storage endpoints.
    
    This client handles authentication and provides methods for uploading,
    downloading, and deleting files through the Brain API storage endpoints.
    """
    
    def __init__(self):
        """Initialize the Brain API client with configuration from environment variables."""
        self.base_url = os.getenv("BRANDO_BRAIN_API_BASE_URL", "http://localhost:2024/api")
        self.api_key = os.getenv("BRANDO_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "BRANDO_API_KEY environment variable is required for Brain API authentication"
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "ComfyUI-NilorNodes/1.0"
        }
        
        logging.info(f"Brain API Client initialized with base URL: {self.base_url}")
    
    def upload_file_to_storage(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Upload a file to Brain API storage and return storage metadata.
        
        Args:
            file_path: Local path to the file to upload
            filename: Name to use for the uploaded file
            
        Returns:
            Dict containing storage_id and filename
            
        Raises:
            requests.RequestException: If upload fails
            FileNotFoundError: If file_path doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        url = f"{self.base_url}/storage/upload"
        
        try:
            with open(file_path, 'rb') as file:
                files = {'file': (filename, file, 'application/octet-stream')}
                
                logging.info(f"Uploading file '{filename}' to Brain API storage...")
                response = requests.post(
                    url, 
                    files=files, 
                    headers=self.headers, 
                    timeout=300
                )
                response.raise_for_status()
                
                result = response.json()
                logging.info(f"Upload successful. Storage ID: {result.get('storage_id')}")
                return result
                
        except requests.RequestException as e:
            logging.error(f"Failed to upload file '{filename}': {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error uploading file '{filename}': {e}")
            raise
    
    def upload_fileobj_to_storage(self, file_obj, filename: str, content_type: str = 'application/octet-stream') -> Dict[str, Any]:
        """
        Upload a file-like object to Brain API storage and return storage metadata.
        
        Args:
            file_obj: File-like object to upload
            filename: Name to use for the uploaded file
            content_type: MIME type of the file
            
        Returns:
            Dict containing storage_id and filename
            
        Raises:
            requests.RequestException: If upload fails
        """
        url = f"{self.base_url}/storage/upload"
        
        try:
            files = {'file': (filename, file_obj, content_type)}
            
            logging.info(f"Uploading file object '{filename}' to Brain API storage...")
            response = requests.post(
                url, 
                files=files, 
                headers=self.headers, 
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            logging.info(f"Upload successful. Storage ID: {result.get('storage_id')}")
            return result
            
        except requests.RequestException as e:
            logging.error(f"Failed to upload file object '{filename}': {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error uploading file object '{filename}': {e}")
            raise
    
    def download_file_from_storage(self, storage_id: str, filename: str, dest_path: str) -> str:
        """
        Download a file from Brain API storage to a local path.
        
        Args:
            storage_id: Storage ID of the file to download
            filename: Name of the file to download
            dest_path: Local path where the file should be saved
            
        Returns:
            Path to the downloaded file
            
        Raises:
            requests.RequestException: If download fails
        """
        url = f"{self.base_url}/storage/{storage_id}"
        params = {'filename': filename}
        
        try:
            logging.info(f"Downloading file '{filename}' (storage_id: {storage_id}) from Brain API storage...")
            response = requests.get(
                url, 
                params=params, 
                headers=self.headers, 
                timeout=300,
                stream=True
            )
            response.raise_for_status()
            
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info(f"Download successful. File saved to: {dest_path}")
            return dest_path
            
        except requests.RequestException as e:
            logging.error(f"Failed to download file '{filename}' (storage_id: {storage_id}): {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error downloading file '{filename}': {e}")
            raise
    
    def get_file_from_storage(self, storage_id: str, filename: str) -> bytes:
        """
        Get file content from Brain API storage as bytes.
        
        Args:
            storage_id: Storage ID of the file to download
            filename: Name of the file to download
            
        Returns:
            File content as bytes
            
        Raises:
            requests.RequestException: If download fails
        """
        url = f"{self.base_url}/storage/{storage_id}"
        params = {'filename': filename}
        
        try:
            logging.info(f"Getting file '{filename}' (storage_id: {storage_id}) from Brain API storage...")
            response = requests.get(
                url, 
                params=params, 
                headers=self.headers, 
                timeout=300
            )
            response.raise_for_status()
            
            logging.info(f"File retrieval successful. Size: {len(response.content)} bytes")
            return response.content
            
        except requests.RequestException as e:
            logging.error(f"Failed to get file '{filename}' (storage_id: {storage_id}): {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error getting file '{filename}': {e}")
            raise
    
    def delete_file_from_storage(self, storage_id: str, filename: str) -> None:
        """
        Delete a file from Brain API storage.
        
        Args:
            storage_id: Storage ID of the file to delete
            filename: Name of the file to delete
            
        Raises:
            requests.RequestException: If deletion fails
        """
        url = f"{self.base_url}/storage/{storage_id}"
        params = {'filename': filename}
        
        try:
            logging.info(f"Deleting file '{filename}' (storage_id: {storage_id}) from Brain API storage...")
            response = requests.delete(
                url, 
                params=params, 
                headers=self.headers, 
                timeout=60
            )
            response.raise_for_status()
            
            logging.info(f"File deletion successful")
            
        except requests.RequestException as e:
            logging.error(f"Failed to delete file '{filename}' (storage_id: {storage_id}): {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error deleting file '{filename}': {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if the Brain API is accessible and authentication is working.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Try to access a simple endpoint to verify connectivity
            url = f"{self.base_url}/health"  # Assuming there's a health endpoint
            response = requests.get(url, headers=self.headers, timeout=10)
            return response.status_code == 200
        except:
            # If health endpoint doesn't exist, try the storage upload endpoint
            # with a HEAD request to check authentication
            try:
                url = f"{self.base_url}/storage/upload"
                response = requests.head(url, headers=self.headers, timeout=10)
                return response.status_code in [200, 405]  # 405 Method Not Allowed is OK for HEAD
            except:
                return False


# Global client instance
_brain_api_client = None

def get_brain_api_client() -> BrainApiClient:
    """
    Get or create the global Brain API client instance.
    
    Returns:
        BrainApiClient instance
    """
    global _brain_api_client
    if _brain_api_client is None:
        _brain_api_client = BrainApiClient()
    return _brain_api_client

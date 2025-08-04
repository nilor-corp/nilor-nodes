# Imports:
import boto3
import json
import time
import logging
import requests
import os

# --- Configuration ---
# Use environment variables with sensible defaults for Docker and local testing.
SQS_ENDPOINT_URL = os.getenv("SQS_ENDPOINT_URL", "http://localhost:9324")
SQS_QUEUE_NAME = os.getenv("SQS_QUEUE_NAME", "jobs_to_process")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188") + "/prompt"

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def submit_to_local_comfyui(payload: dict) -> bool:
    """
    Sends the job payload to the local ComfyUI /prompt endpoint.
    """
    try:
        logging.info(f"Submitting job to local ComfyUI at {COMFYUI_URL}")
        # The payload from the Brain API is the complete, ready-to-run object.
        response = requests.post(COMFYUI_URL, json=payload, timeout=20)
        response.raise_for_status()
        logging.info(f"Successfully submitted job to local ComfyUI. Response: {response.json()}")
        return True
    except requests.RequestException as e:
        logging.error(f"Failed to POST job to local ComfyUI: {e}")
        return False

def consume_jobs():
    """
    Connects to ElasticMQ and enters a loop to poll for, process, and delete messages.
    """
    logging.info(f"Worker Consumer starting up. Connecting to SQS at {SQS_ENDPOINT_URL}...")
    
    # 1. Create an SQS client pointing to the local ElasticMQ instance.
    try:
        sqs_client = boto3.client(
            'sqs',
            endpoint_url=SQS_ENDPOINT_URL,
            region_name=AWS_REGION,
            aws_access_key_id='x', # Dummy credentials for local dev
            aws_secret_access_key='x'
        )
        
        # Get the full queue URL from its name.
        response = sqs_client.get_queue_url(QueueName=SQS_QUEUE_NAME)
        queue_url = response['QueueUrl']
        logging.info(f"Successfully connected to queue: {queue_url}")
    except Exception as e:
        logging.error(f"Could not connect to SQS queue. Exiting. Error: {e}")
        return

    # 2. Start the main consumer loop.
    logging.info("Starting to poll for messages...")
    while True:
        try:
            # 3. Long-poll for messages. This is more efficient than rapid, short polling.
            #    WaitTimeSeconds tells SQS to hold the connection open for up to 20 seconds
            #    if no messages are available.
            response = sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20 
            )

            # Check if the 'Messages' key exists and if it's not empty.
            if 'Messages' in response and len(response['Messages']) > 0:
                message = response['Messages'][0]
                receipt_handle = message['ReceiptHandle']
                
                logging.info(f"Received job. Body: {message['Body']}")
                workflow_payload = json.loads(message['Body'])
                
                # THE FIX: Pass the entire payload, not just the 'prompt' field, to the ComfyUI server.
                submit_successful = submit_to_local_comfyui(workflow_payload)
                
                if submit_successful:
                    sqs_client.delete_message(
                        QueueUrl=queue_url,
                        ReceiptHandle=receipt_handle
                    )
                    logging.info("Message processed and deleted from queue.")
                else:
                    logging.warning("Local submission failed. Message will be retried after visibility timeout.")

            else:
                logging.info("No messages received. Polling again...")

        except Exception as e:
            logging.error(f"An error occurred in the consumer loop: {e}")
            # Wait for a moment before retrying to avoid spamming logs on a persistent error.
            time.sleep(10)

# This allows the script to be run directly for testing,
# but prevents it from running automatically when imported as a module.
if __name__ == "__main__":
    consume_jobs()

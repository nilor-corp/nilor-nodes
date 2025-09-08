"""
Worker Consumer Service for ComfyUI

This script runs as a continuous background service on each ComfyUI worker.
Its purpose is to poll the `jobs_to_process` SQS queue for new jobs,
submit them to the local ComfyUI server, and manage the message lifecycle.
"""
import os
import json
import logging
import time
import requests
import boto3
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Load from the .env file in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f"Loaded environment variables from {dotenv_path}")
else:
    logging.info("No .env file found, relying on shell environment variables.")


# --- Configuration ---
SQS_ENDPOINT_URL = os.getenv("SQS_ENDPOINT_URL", "http://localhost:9324")
SQS_JOBS_TO_PROCESS_QUEUE_NAME = os.getenv("SQS_JOBS_TO_PROCESS_QUEUE_NAME", "jobs_to_process")
SQS_JOB_STATUS_UPDATES_QUEUE_NAME = os.getenv("SQS_JOB_STATUS_UPDATES_QUEUE_NAME", "job_status_updates")
COMFYUI_API_URL = os.getenv("COMFYUI_API_URL", "http://127.0.0.1:8188") + "/prompt"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "local")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "local")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
POLL_WAIT_TIME_SECONDS = 20 # SQS Long Polling
MAX_MESSAGES = 1

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkerConsumer:
    def __init__(self):
        self.sqs_client = boto3.client(
            'sqs',
            endpoint_url=SQS_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION
        )
        self.jobs_queue_url = self._get_queue_url(SQS_JOBS_TO_PROCESS_QUEUE_NAME)
        self.status_updates_queue_url = self._get_queue_url(SQS_JOB_STATUS_UPDATES_QUEUE_NAME)

    def _get_queue_url(self, queue_name):
        """Retrieves the SQS queue URL."""
        try:
            response = self.sqs_client.get_queue_url(QueueName=queue_name)
            logging.info(f"Successfully retrieved queue URL for '{queue_name}'")
            return response['QueueUrl']
        except self.sqs_client.exceptions.QueueDoesNotExist:
            logging.error(f"Queue '{queue_name}' does not exist. Please ensure it's created.")
            raise

    def consume_loop(self):
        """The main loop to continuously poll for and process messages."""
        logging.info(f"Starting worker consumer. Polling queue: {self.jobs_queue_url}")
        while True:
            try:
                logging.debug(f"Waiting for messages (long poll for {POLL_WAIT_TIME_SECONDS}s)...")
                response = self.sqs_client.receive_message(
                    QueueUrl=self.jobs_queue_url,
                    MaxNumberOfMessages=MAX_MESSAGES,
                    WaitTimeSeconds=POLL_WAIT_TIME_SECONDS
                )

                messages = response.get('Messages', [])
                if not messages:
                    continue # Go back to polling

                for message in messages:
                    self.process_message(message)

            except Exception as e:
                logging.error(f"An unexpected error occurred in the consumer loop: {e}", exc_info=True)
                time.sleep(10) # Wait before retrying to avoid spamming logs

    def process_message(self, message):
        """Processes a single message from the queue."""
        receipt_handle = message['ReceiptHandle']
        try:
            logging.info(f"Received message: {message['MessageId']}")
            workflow_payload = json.loads(message['Body'])
            
            # The actual job data is nested under 'Message' if it comes from SNS,
            # but directly in the body if sent directly to SQS. We'll handle both.
            if 'Message' in workflow_payload:
                workflow_data = json.loads(workflow_payload['Message'])
            else:
                workflow_data = workflow_payload
            
            logging.info(f"Submitting job to local ComfyUI server at {COMFYUI_API_URL}")
            response = requests.post(COMFYUI_API_URL, json=workflow_data, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            logging.info(f"Successfully submitted job to ComfyUI. Prompt ID: {response.json().get('prompt_id')}")
            
            # Send a status update to the status queue
            self._send_status_update(workflow_data, "running")

            # If submission is successful, delete the message from the queue
            self.sqs_client.delete_message(
                QueueUrl=self.jobs_queue_url,
                ReceiptHandle=receipt_handle
            )
            logging.info(f"Deleted message {message['MessageId']} from the queue.")

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit job to ComfyUI: {e}. Message will be retried after visibility timeout.")
            # Do NOT delete the message, let it become visible again for retry
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse message body: {e}. Discarding malformed message.")
            # Delete the malformed message to prevent poison pill
            self.sqs_client.delete_message(
                QueueUrl=self.jobs_queue_url,
                ReceiptHandle=receipt_handle
            )
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing message: {e}. It will be retried.", exc_info=True)
            # Do NOT delete the message, let it retry

    def _send_status_update(self, workflow_data, status):
        """Sends a status update message to the SQS queue."""
        job_id = workflow_data.get("client_id")
        if not job_id:
            logging.warning("No 'client_id' found in workflow data; skipping status update.")
            return

        try:
            message_body = json.dumps({"job_id": job_id, "status": status})
            self.sqs_client.send_message(
                QueueUrl=self.status_updates_queue_url,
                MessageBody=message_body
            )
            logging.info(f"Sent status update for job {job_id}: {status}")
        except Exception as e:
            logging.error(f"Failed to send status update for job {job_id}: {e}", exc_info=True)


def consume_jobs():
    """Entry point function to be called in a background thread."""
    consumer = WorkerConsumer()
    consumer.consume_loop()

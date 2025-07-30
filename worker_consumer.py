# Imports:
import boto3
import json
import time
import logging

# --- Configuration ---
# These should eventually come from environment variables passed to the worker's container.
SQS_ENDPOINT_URL = "http://localhost:9324"
SQS_QUEUE_NAME = "jobs_to_process"
AWS_REGION = "us-east-1"

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                
                # --- PROOF OF CONCEPT PROCESSING ---
                # In this initial version, we just print the message body.
                # In the future, this is where it will be submitted to the local ComfyUI.
                logging.info(f"Received message. Body: {message['Body']}")
                
                # 4. Delete the message from the queue to prevent it from being re-processed.
                #    This is a critical step.
                sqs_client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle
                )
                logging.info("Message processed and deleted from queue.")
            else:
                # This is not an error, just an empty response from long polling.
                logging.info("No messages received. Polling again...")

        except Exception as e:
            logging.error(f"An error occurred in the consumer loop: {e}")
            # Wait for a moment before retrying to avoid spamming logs on a persistent error.
            time.sleep(10)

# This allows the script to be run directly for testing,
# but prevents it from running automatically when imported as a module.
if __name__ == "__main__":
    consume_jobs()

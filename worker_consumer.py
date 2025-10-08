"""
Worker Consumer Service for ComfyUI

This script runs as a continuous background service on each ComfyUI worker.
Its purpose is to poll the `jobs_to_process` SQS queue for new jobs,
submit them to the local ComfyUI server, and manage the message lifecycle.
It also listens to the ComfyUI websocket to send a "running" status update
at the precise moment that job execution begins.
"""

import os
import json
import logging
import asyncio
import aiohttp
import websockets
from aiobotocore.session import get_session
from dotenv import load_dotenv
from botocore.exceptions import EndpointConnectionError
from .logger import logger

# --- Load Environment Variables ---
# Load from the .env file in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(
        f"✅\u2009 Nilor-Nodes: Loaded environment variables from {dotenv_path}"
    )
else:
    logger.info(
        "⚠️\u2009 Nilor-Nodes: No .env file found, relying on shell environment variables."
    )

# --- Configuration ---
SQS_ENDPOINT_URL = os.getenv("SQS_ENDPOINT_URL", "http://localhost:9324")
SQS_JOBS_TO_PROCESS_QUEUE_NAME = os.getenv(
    "SQS_JOBS_TO_PROCESS_QUEUE_NAME", "jobs_to_process"
)
SQS_JOB_STATUS_UPDATES_QUEUE_NAME = os.getenv(
    "SQS_JOB_STATUS_UPDATES_QUEUE_NAME", "job_status_updates"
)
COMFYUI_API_URL = os.getenv("COMFYUI_API_URL", "http://127.0.0.1:8188") + "/prompt"
COMFYUI_WS_URL = os.getenv("COMFYUI_WS_URL", "ws://127.0.0.1:8188") + "/ws"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "local")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "local")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
POLL_WAIT_TIME_SECONDS = 20  # SQS Long Polling
MAX_MESSAGES = 1


class WorkerConsumer:
    def __init__(self):
        self.session = get_session()
        self.prompt_id_to_content_id_map = {}
        self.sent_running_status_prompts = set()
        self.content_context_by_content_id = {}
        self.jobs_queue_url = None
        self.status_updates_queue_url = None
        self.http_session = None

    async def _initialize_sqs(self):
        """Initializes SQS queue URLs. Returns True on success, False on failure."""
        async with self.session.create_client(
            "sqs",
            region_name=AWS_DEFAULT_REGION,
            endpoint_url=SQS_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        ) as client:
            try:
                self.jobs_queue_url = await self._get_queue_url(
                    client, SQS_JOBS_TO_PROCESS_QUEUE_NAME
                )
                self.status_updates_queue_url = await self._get_queue_url(
                    client, SQS_JOB_STATUS_UPDATES_QUEUE_NAME
                )
                return True
            except EndpointConnectionError as e:
                # Quiet the noisy traceback by logging a concise warning instead
                logger.warning(
                    f"⚠️\u2009 Nilor-Nodes (worker_consumer): SQS endpoint is unreachable at {SQS_ENDPOINT_URL}: {e}. "
                )
                return False
            except Exception as e:
                logger.error(
                    f"⚠️\u2009 Nilor-Nodes (worker_consumer): Failed to initialize SQS queues: {e}"
                )
                return False

    async def _get_queue_url(self, client, queue_name):
        """Retrieves the SQS queue URL."""
        try:
            response = await client.get_queue_url(QueueName=queue_name)
            return response["QueueUrl"]
        except client.exceptions.QueueDoesNotExist:
            logger.error(
                f"⚠️\u2009 Nilor-Nodes (worker_consumer): SQS queue '{queue_name}' does not exist."
            )
            raise

    async def listen_for_comfy_events(self):
        while True:
            try:
                async with websockets.connect(COMFYUI_WS_URL) as websocket:
                    logger.info(
                        f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Connected to ComfyUI websocket at {COMFYUI_WS_URL}"
                    )
                    while True:
                        message = await websocket.recv()
                        if isinstance(message, str):
                            try:
                                event = json.loads(message)
                                event_type = event.get("type")
                                data = event.get("data", {})
                                prompt_id = data.get("prompt_id")

                                if not prompt_id and "sid" in data:
                                    prompt_id = data["sid"]

                                if not prompt_id:
                                    continue

                                # Use the first progress event as a signal that the job is running.
                                if (
                                    event_type in ["progress", "progress_state"]
                                    and prompt_id in self.prompt_id_to_content_id_map
                                    and prompt_id
                                    not in self.sent_running_status_prompts
                                ):
                                    content_id = self.prompt_id_to_content_id_map[
                                        prompt_id
                                    ]
                                    ctx = self.content_context_by_content_id.get(
                                        content_id, {}
                                    )
                                    policy = ctx.get("status_policy") or {}
                                    running_status = policy.get(
                                        "running_status", "running"
                                    )
                                    logger.info(
                                        f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Execution started for prompt_id {prompt_id} (content_id: {content_id}) via '{event_type}' event. Sending '{running_status}' status."
                                    )
                                    await self._send_status_update(
                                        content_id,
                                        running_status,
                                        ctx.get("venue"),
                                        ctx.get("canvas"),
                                        ctx.get("scene"),
                                    )
                                    self.sent_running_status_prompts.add(prompt_id)

                                # Handle execution errors
                                elif event_type == "execution_error":
                                    logger.error(
                                        f"🛑\u2009 Nilor-Nodes (worker_consumer): Received execution error for prompt_id {prompt_id}: {data}"
                                    )
                                    if prompt_id in self.prompt_id_to_content_id_map:
                                        content_id = (
                                            self.prompt_id_to_content_id_map.pop(
                                                prompt_id
                                            )
                                        )
                                        ctx = self.content_context_by_content_id.get(
                                            content_id, {}
                                        )
                                        policy = ctx.get("status_policy") or {}
                                        fail_status = policy.get(
                                            "fail_status", "failed"
                                        )
                                        try:
                                            await self._send_status_update(
                                                content_id,
                                                fail_status,
                                                ctx.get("venue"),
                                                ctx.get("canvas"),
                                                ctx.get("scene"),
                                            )
                                        except Exception:
                                            pass
                                        self.content_context_by_content_id.pop(
                                            content_id, None
                                        )
                                    self.sent_running_status_prompts.discard(prompt_id)

                                # Log successful execution
                                elif event_type == "executed":
                                    logger.info(
                                        f"✅ Nilor-Nodes (worker_consumer): Prompt {prompt_id} executed successfully according to websocket event. Final node is responsible for sending completion message."
                                    )
                                    if prompt_id in self.prompt_id_to_content_id_map:
                                        content_id = (
                                            self.prompt_id_to_content_id_map.pop(
                                                prompt_id
                                            )
                                        )
                                        self.content_context_by_content_id.pop(
                                            content_id, None
                                        )
                                    self.sent_running_status_prompts.discard(prompt_id)

                                elif event_type not in ["progress", "progress_state"]:
                                    logger.info(
                                        f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Received ComfyUI websocket event of type '{event_type}': {data}"
                                    )

                            except json.JSONDecodeError:
                                logger.debug(
                                    "⚠️\u2009 Nilor-Nodes (worker_consumer): Received non-JSON text message from websocket, ignoring."
                                )
                        else:
                            logger.debug(
                                "⚠️\u2009 Nilor-Nodes (worker_consumer): Received binary message from websocket, ignoring."
                            )
            except (
                websockets.exceptions.ConnectionClosedError,
                ConnectionRefusedError,
            ) as e:
                logger.warning(
                    f"🛑\u2009 Nilor-Nodes (worker_consumer): ComfyUI websocket connection failed: {e}. Retrying in 5 seconds..."
                )
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(
                    f"🛑\u2009 Nilor-Nodes (worker_consumer): An unexpected error occurred in the websocket listener: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(10)

    async def consume_loop(self):
        """The main loop to continuously poll for and process messages.
        Keeps retrying SQS initialization and polling if the endpoint is down.
        """

        # Start the websocket listener in the background immediately
        listener_task = asyncio.create_task(self.listen_for_comfy_events())

        try:

            while True:
                # Ensure SQS is initialized; if not, keep attempting to initialize
                if self.jobs_queue_url is None or self.status_updates_queue_url is None:
                    initialized = await self._initialize_sqs()
                    if not initialized:
                        logging.warning(
                            "⚠️\u2009 Nilor-Nodes (worker_consumer): SQS initialization failed. Retrying in 10 seconds..."
                        )
                        await asyncio.sleep(10)
                        continue
                    logger.info(
                        f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Starting worker consumer. Polling queue: {self.jobs_queue_url}"
                    )

                logger.debug(
                    "ℹ️\u2009 Nilor-Nodes (worker_consumer): Polling for messages..."
                )
                try:
                    async with self.session.create_client(
                        "sqs",
                        region_name=AWS_DEFAULT_REGION,
                        endpoint_url=SQS_ENDPOINT_URL,
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    ) as client:
                        response = await client.receive_message(
                            QueueUrl=self.jobs_queue_url,
                            MaxNumberOfMessages=MAX_MESSAGES,
                            WaitTimeSeconds=POLL_WAIT_TIME_SECONDS,
                        )

                    messages = response.get("Messages", [])
                    if not messages:
                        logger.debug(
                            "ℹ️\u2009 Nilor-Nodes (worker_consumer): No messages received."
                        )
                        continue

                    for message in messages:
                        try:
                            await self.process_message(message)

                            # On successful processing, delete the message
                            async with self.session.create_client(
                                "sqs",
                                region_name=AWS_DEFAULT_REGION,
                                endpoint_url=SQS_ENDPOINT_URL,
                                aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                            ) as client:
                                await client.delete_message(
                                    QueueUrl=self.jobs_queue_url,
                                    ReceiptHandle=message["ReceiptHandle"],
                                )
                            logger.info(
                                f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Deleted message {message['MessageId']} from queue."
                            )
                        except json.JSONDecodeError:
                            # This is a poison pill message, log it but don't retry.
                            # It will be moved to the DLQ after enough failed receives.
                            logger.error(
                                f"🛑\u2009 Nilor-Nodes (worker_consumer): Message {message['MessageId']} is a poison pill (JSON decode failed) and will be ignored."
                            )
                        except Exception as e:
                            logger.error(
                                f"🛑\u2009 Nilor-Nodes (worker_consumer): Processing failed for message {message['MessageId']}: {e}. It will be returned to the queue for retry."
                            )

                except EndpointConnectionError as e:
                    # Lost connection to SQS; reset and re-initialize on next loop
                    logger.warning(
                        f"⚠️\u2009 Nilor-Nodes (worker_consumer): Lost connection to SQS at {SQS_ENDPOINT_URL}: {e}. Will retry initialization in 10 seconds."
                    )
                    self.jobs_queue_url = None
                    self.status_updates_queue_url = None
                    await asyncio.sleep(10)
                except Exception as e:
                    logger.error(
                        f"🛑\u2009 Nilor-Nodes (worker_consumer): An error occurred in the consume loop: {e}"
                    )
                    await asyncio.sleep(10)  # Wait before retrying
        finally:
            listener_task.cancel()
            await asyncio.gather(listener_task, return_exceptions=True)
            logger.info(
                "⚠️\u2009 Nilor-Nodes (worker_consumer): Websocket listener stopped."
            )

    async def process_message(self, message):
        """Processes a single SQS message."""
        logger.info(
            f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Processing message: {message['MessageId']}"
        )

        try:
            body = json.loads(message["Body"])
            # SQS messages are often double-encoded, with the actual payload inside a 'Message' key.
            if "Message" in body:
                job_payload = json.loads(body["Message"])
            else:
                job_payload = body

            content_id = job_payload.get("content_id")

            # Validate that the payload has the required keys before submitting.
            if not content_id or "prompt" not in job_payload:
                logger.error(
                    f"🛑\u2009 Nilor-Nodes (worker_consumer): Invalid message format: missing 'content_id' or 'prompt'. Payload: {job_payload}"
                )
                return

            # Submit to ComfyUI
            await self._submit_job_to_comfyui(content_id, job_payload)

            # Cache context for subsequent status updates
            try:
                self.content_context_by_content_id[content_id] = {
                    "venue": job_payload.get("venue"),
                    "canvas": job_payload.get("canvas"),
                    "scene": job_payload.get("scene"),
                    "status_policy": job_payload.get("status_policy") or {},
                }
            except Exception:
                self.content_context_by_content_id[content_id] = {}

        except Exception as e:
            logger.error(
                f"🛑\u2009 Nilor-Nodes (worker_consumer): An unexpected error occurred while processing message: {e}. It will be retried."
            )
            # Re-raise to prevent deletion from queue if we want SQS to handle retry
            raise

    async def _submit_job_to_comfyui(self, content_id, workflow_data):
        """Submits a single job to the ComfyUI API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    COMFYUI_API_URL, json=workflow_data, timeout=30
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    prompt_id = response_json.get("prompt_id")
                    logger.info(
                        f"✅ Nilor-Nodes (worker_consumer): Successfully submitted job to ComfyUI. Prompt ID: {prompt_id}"
                    )
                    self.prompt_id_to_content_id_map[prompt_id] = content_id

            # No need to delete here, the consume_loop handles message deletion
        except aiohttp.ClientError as e:
            logger.error(
                f"🛑\u2009 Nilor-Nodes (worker_consumer): Failed to submit job to ComfyUI: {e}. Message will be retried."
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(
                f"🛑\u2009 Nilor-Nodes (worker_consumer): Failed to parse ComfyUI response: {e}. Discarding malformed response."
            )
        except Exception as e:
            logger.error(
                f"🛑\u2009 Nilor-Nodes (worker_consumer): An unexpected error occurred while submitting job to ComfyUI: {e}",
                exc_info=True,
            )

    async def _send_status_update(
        self, content_id, status, venue=None, canvas=None, scene=None
    ):
        try:
            body = {"content_id": content_id, "status": status}
            if venue is not None:
                body["venue"] = venue
            if canvas is not None:
                body["canvas"] = canvas
            if scene is not None:
                body["scene"] = scene
            message_body = json.dumps(body)
            async with self.session.create_client(
                "sqs",
                region_name=AWS_DEFAULT_REGION,
                endpoint_url=SQS_ENDPOINT_URL,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            ) as client:
                await client.send_message(
                    QueueUrl=self.status_updates_queue_url, MessageBody=message_body
                )
            logger.info(
                f"✅ Nilor-Nodes (worker_consumer): Sent status update for content {content_id}: {status}"
            )
        except Exception as e:
            logger.error(
                f"🛑\u2009 Nilor-Nodes (worker_consumer): Failed to send status update for content {content_id}: {e}",
                exc_info=True,
            )


async def consume_jobs():
    """Entry point function to be called in a background thread."""
    consumer = WorkerConsumer()
    await consumer.consume_loop()

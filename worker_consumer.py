"""
Worker Consumer Service for ComfyUI

This script runs as a continuous background service on each ComfyUI worker.
Its purpose is to poll the `jobs_to_process-comfyui` SQS queue for new jobs,
submit them to the local ComfyUI server, and manage the message lifecycle.
It also listens to the ComfyUI websocket to send a "running" status update
at the precise moment that job execution begins.
"""

import os
import json
import logging
import asyncio
import aiohttp
import socket
import urllib.parse
import websockets
from aiobotocore.session import get_session
from dotenv import load_dotenv
from botocore.exceptions import EndpointConnectionError, ClientError
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
    "SQS_JOBS_TO_PROCESS_QUEUE_NAME", "jobs_to_process-comfyui"
)
SQS_JOB_STATUS_UPDATES_QUEUE_NAME = os.getenv(
    "SQS_JOB_STATUS_UPDATES_QUEUE_NAME", "job_status_updates"
)
SQS_POLL_WAIT_TIME = int(os.getenv("SQS_POLL_WAIT_TIME", "10"))  # SQS Long Polling
SQS_MAX_MESSAGES = 1
COMFYUI_API_URL = os.getenv("COMFYUI_API_URL", "http://127.0.0.1:8188") + "/prompt"
COMFYUI_WS_URL = os.getenv("COMFYUI_WS_URL", "ws://127.0.0.1:8188") + "/ws"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "local")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "local")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


class WorkerConsumer:
    def __init__(self):
        self.session = get_session()
        self.prompt_id_to_content_id_map = {}
        self.sent_running_status_prompts = set()
        self.content_context_by_content_id = {}
        self.jobs_queue_url = None
        self.status_updates_queue_url = None
        self.http_session = None
        self.is_busy = False
        self.websocket_sid = None

        # Stable client_id for routing events to this worker
        env_id = os.getenv("NILOR_WORKER_CLIENT_ID")
        if env_id and env_id.strip():
            self.worker_client_id = env_id.strip()
        else:
            host = socket.gethostname()
            self.worker_client_id = f"nilor-worker-{host}"
        self.current_prompt_id = None

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
                ws_url = f"{COMFYUI_WS_URL}?clientId={urllib.parse.quote(self.worker_client_id)}"
                # Allow large preview frames from ComfyUI without dropping the connection (1009: message too big)
                async with websockets.connect(
                    ws_url,
                    max_size=None,  # disable message size limit
                    read_limit=64 * 1024 * 1024,  # buffer up to 64 MiB per frame
                    max_queue=4,  # small queue to avoid memory spikes
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    logger.debug(
                        f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Connected to ComfyUI websocket at {ws_url}"
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

                                # Allow 'executing' events through even if missing prompt_id
                                if not prompt_id and event_type != "executing":
                                    continue

                                # Capture our websocket client id from initial status message
                                if event_type == "status":
                                    sid = data.get("sid")
                                    if sid:
                                        self.websocket_sid = sid
                                        logger.debug(
                                            f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Captured websocket SID: {sid}"
                                        )

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
                                        f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Execution started for prompt_id {prompt_id} (content_id: {content_id}) via '{event_type}' event."
                                    )
                                    # Mark worker busy as soon as execution starts
                                    self.is_busy = True

                                    await self._send_status_update(
                                        content_id,
                                        running_status,
                                        ctx.get("venue"),
                                        ctx.get("canvas"),
                                        ctx.get("scene"),
                                        ctx.get("job_type"),
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
                                                ctx.get("job_type"),
                                            )
                                        except Exception:
                                            pass
                                        self.content_context_by_content_id.pop(
                                            content_id, None
                                        )
                                    effective_prompt_id = (
                                        prompt_id or self.current_prompt_id
                                    )
                                    if effective_prompt_id:
                                        self._finalize_prompt(
                                            effective_prompt_id,
                                            reason="execution_error",
                                        )

                                # Node-level executed event (many per prompt) — ignore for busy/reset
                                elif event_type == "executed":
                                    logger.debug(
                                        f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Received node executed event for prompt_id {prompt_id}: {data}"
                                    )

                                # Prompt-level completion signal: executing with node None
                                elif event_type == "executing":
                                    node_id = data.get("node")
                                    logger.debug(
                                        f"ℹ️\u2009 Nilor-Nodes (worker_consumer): Received 'executing' event. prompt_id={prompt_id}, node_id={node_id}"
                                    )

                                    norm_node_id = self._normalize_node_id(node_id)
                                    if norm_node_id is None:
                                        effective_prompt_id = (
                                            prompt_id or self.current_prompt_id
                                        )
                                        if effective_prompt_id:
                                            self._finalize_prompt(
                                                effective_prompt_id,
                                                reason="executing node=None",
                                            )

                                elif event_type == "execution_success":
                                    effective_prompt_id = (
                                        prompt_id or self.current_prompt_id
                                    )
                                    if effective_prompt_id:
                                        self._finalize_prompt(
                                            effective_prompt_id,
                                            reason="execution_success",
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

                # Capacity gate: avoid pulling a new job while local ComfyUI is busy
                if self.is_busy:
                    logger.debug(
                        "ℹ️\u2009 Nilor-Nodes (worker_consumer): Skipping poll; worker is busy executing a job."
                    )
                    await asyncio.sleep(5)
                    continue

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
                            MaxNumberOfMessages=SQS_MAX_MESSAGES,
                            WaitTimeSeconds=SQS_POLL_WAIT_TIME,
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
                            logger.debug(
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
                except ClientError as e:
                    # Some SQS providers/endpoints may sporadically return 503 for ReceiveMessage when the queue is idle
                    error_code = None
                    try:
                        error_code = e.response.get("Error", {}).get("Code")
                    except Exception:
                        pass
                    operation_name = getattr(e, "operation_name", "")
                    if operation_name == "ReceiveMessage" and str(error_code) in (
                        "503",
                        "ServiceUnavailable",
                    ):
                        logger.debug(
                            "ℹ️\u2009 Nilor-Nodes (worker_consumer): Queue is empty or endpoint timed out (ReceiveMessage 503). Polling again shortly..."
                        )
                        # await asyncio.sleep(2)
                        continue

                    # Unhandled ClientError; fall back to generic handling
                    logger.error(
                        f"🛑\u2009 Nilor-Nodes (worker_consumer): SQS client error during ReceiveMessage: {e}"
                    )
                    await asyncio.sleep(10)
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

            # Prefer execution_spec for other engines; ComfyUI strictly requires 'prompt'
            if "execution_spec" in job_payload and "prompt" not in job_payload:
                logger.warning(
                    f"⚠️\u2009 Nilor-Nodes (worker_consumer): Received 'execution_spec' without 'prompt'. ComfyUI path requires 'prompt'; skipping message {message['MessageId']}."
                )
                return
            if "execution_spec" in job_payload and "prompt" in job_payload:
                logger.debug(
                    "ℹ️\u2009 Nilor-Nodes (worker_consumer): 'execution_spec' present alongside 'prompt'; ignoring 'execution_spec' for ComfyUI."
                )

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
                    "job_type": job_payload.get("job_type"),
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
            # Attach/override websocket client_id so server targets events to this worker
            payload = (
                dict(workflow_data)
                if isinstance(workflow_data, dict)
                else workflow_data
            )
            if isinstance(payload, dict):
                # Force top-level client_id to this worker's stable ID
                payload["client_id"] = self.worker_client_id
                # Ensure extra_data exists and force its client_id too
                extra = payload.get("extra_data") or {}
                if isinstance(extra, dict):
                    extra["client_id"] = self.worker_client_id
                    payload["extra_data"] = extra

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    COMFYUI_API_URL, json=payload, timeout=30
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    prompt_id = response_json.get("prompt_id")
                    logger.debug(
                        f"✅ Nilor-Nodes (worker_consumer): Successfully submitted job to ComfyUI. Prompt ID: {prompt_id}"
                    )
                    self.prompt_id_to_content_id_map[prompt_id] = content_id
                    # Mark worker busy after successful submission to avoid over-queuing on this machine
                    self.is_busy = True
                    self.current_prompt_id = prompt_id

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
        self, content_id, status, venue=None, canvas=None, scene=None, job_type=None
    ):
        try:
            body = {"content_id": content_id, "status": status}
            if venue is not None:
                body["venue"] = venue
            if canvas is not None:
                body["canvas"] = canvas
            if scene is not None:
                body["scene"] = scene
            if job_type is not None:
                body["job_type"] = job_type
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

    def _finalize_prompt(self, prompt_id, reason: str | None = None):
        # No-op if we're not currently busy; avoids redundant work on duplicate signals
        if not self.is_busy:
            return
        try:
            if reason:
                logger.info(
                    f"✅ Nilor-Nodes (worker_consumer): Finalizing prompt via '{reason}'. Using prompt_id={prompt_id}"
                )
            content_id = self.prompt_id_to_content_id_map.pop(prompt_id, None)
            if content_id is not None:
                self.content_context_by_content_id.pop(content_id, None)
            self.sent_running_status_prompts.discard(prompt_id)
        finally:
            # Only clear busy/state if this finalize corresponds to the current in-flight prompt
            if self.current_prompt_id == prompt_id:
                self.current_prompt_id = None
                self.is_busy = False

    @staticmethod
    def _normalize_node_id(node_id):
        if node_id is None:
            return None
        if isinstance(node_id, str) and node_id.strip() in ("", "None"):
            return None
        return node_id


async def consume_jobs():
    """Entry point function to be called in a background thread."""
    consumer = WorkerConsumer()
    await consumer.consume_loop()

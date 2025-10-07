import torch
import numpy as np
from PIL import Image
import requests
import io
import logging
import imageio.v2 as imageio
import mimetypes
import boto3
import os
import json
from dotenv import load_dotenv
from .brain_api_client import get_brain_api_client

# --- Load Environment Variables ---
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
dotenv_path = os.path.join(current_dir, ".env")
# Load the .env file
load_dotenv(dotenv_path=dotenv_path)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Node Categories ---
category = "Nilor Nodes 👺"
subcategories = {
    "streaming": "/Streaming",
}


# --- MediaStreamInput: Universal Media Downloader ---
class MediaStreamInput:
    """
    A custom node to download an image/video from a pre-signed URL and provide it as a tensor.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_name": (
                    "STRING",
                    {"default": "default_input", "multiline": False},
                ),
                "format": (["image", "image_batch", "video"],),
                "storage_id": (
                    "STRING",
                    {"default": "<auto-filled by system>", "multiline": False},
                ),
                "filename": (
                    "STRING",
                    {"default": "<auto-filled by system>", "multiline": False},
                ),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "download"
    CATEGORY = category + subcategories["streaming"]

    def download(
        self,
        storage_id: str,
        filename: str,
        format: str,
        input_name: str = "default_input",
    ):
        logging.info(
            f"ℹ️\u2009 Nilor-Nodes: MediaStreamInput: Downloading file '{filename}' (storage_id: {storage_id}) for input '{input_name}' with format '{format}'"
        )
        try:
            # Get Brain API client and MinIO endpoint
            brain_client = get_brain_api_client()
            minio_endpoint = os.getenv("MINIO_ENDPOINT")
            if not minio_endpoint:
                raise ValueError(
                    "MINIO_ENDPOINT environment variable is required but not set"
                )

            # Two-phase download for batches: manifest first, then assets
            if format == "image_batch":
                # Get presigned download URL for manifest
                manifest_url_response = brain_client.get_presigned_download_url(
                    storage_id, filename, minio_endpoint
                )
                manifest_url = manifest_url_response["download_url"]

                # Download manifest file directly from MinIO
                manifest_response = requests.get(manifest_url, timeout=300)
                manifest_response.raise_for_status()
                manifest = json.loads(manifest_response.content.decode("utf-8"))

                logging.info(
                    f"ℹ️\u2009 Nilor-Nodes: Processing manifest for '{manifest.get('input_name')}' with {len(manifest.get('files', []))} assets."
                )

                # Sort files by sequence number to ensure correct order
                sorted_files = sorted(
                    manifest.get("files", []), key=lambda x: x.get("sequence", 0)
                )

                # Download all assets using presigned URLs
                asset_responses = []
                for file_info in sorted_files:
                    try:
                        file_storage_id = file_info.get("storage_id")
                        file_filename = file_info.get("filename")
                        if not file_storage_id or not file_filename:
                            raise ValueError(
                                f"Missing storage_id or filename in manifest file info: {file_info}"
                            )

                        # Get presigned download URL for this asset
                        asset_url_response = brain_client.get_presigned_download_url(
                            file_storage_id, file_filename, minio_endpoint
                        )
                        asset_url = asset_url_response["download_url"]

                        # Download asset directly from MinIO
                        asset_response = requests.get(asset_url, timeout=300)
                        asset_response.raise_for_status()
                        asset_responses.append(asset_response.content)
                    except Exception as e:
                        logging.error(
                            f"🛑\u2009 Nilor-Nodes: Failed to download asset {file_info.get('filename')}: {e}"
                        )
                        raise  # Re-raise to fail the entire process

                return self._process_image_batch(asset_responses)

            # --- Single-file download ---
            # Get presigned download URL
            download_url_response = brain_client.get_presigned_download_url(
                storage_id, filename, minio_endpoint
            )
            download_url = download_url_response["download_url"]

            # Download file directly from MinIO
            media_response = requests.get(download_url, timeout=300)
            media_response.raise_for_status()
            media_bytes = media_response.content

            if format == "video":
                return self._process_video(media_bytes)
            elif format == "image":
                return self._process_image(media_bytes)
            else:
                # Should not happen if UI choices are respected
                raise ValueError(
                    f"[🛑] Nilor-Nodes (MediaStreamInput): Unsupported format '{format}' for single media download."
                )

        except Exception as e:
            logging.error(
                f"🛑\u2009 Nilor-Nodes (MediaStreamInput): Failed to download or process media: {e}"
            )
            return (None,)

    def _process_image_batch(self, image_bytes_list):
        logging.info(
            f"ℹ️\u2009 Nilor-Nodes (MediaStreamInput): Processing image batch with {len(image_bytes_list)} images..."
        )
        output_images = []

        for image_bytes in image_bytes_list:
            image_pil = Image.open(io.BytesIO(image_bytes))

            rgb_image_pil = image_pil.convert("RGB")
            image_tensor = torch.from_numpy(
                np.array(rgb_image_pil).astype(np.float32) / 255.0
            ).unsqueeze(0)

            output_images.append(image_tensor)

        # Concatenate along the batch dimension (dim=0)
        images_tensor = torch.cat(output_images, dim=0)

        logging.info(
            f"✅ Nilor-Nodes (MediaStreamInput): Image batch processing successful. Batch shape: {images_tensor.shape}"
        )
        return (images_tensor,)

    def _process_image(self, image_bytes):
        logging.info("ℹ️\u2009 Nilor-Nodes (MediaStreamInput): Processing as image...")
        image_pil = Image.open(io.BytesIO(image_bytes))

        # Ensure image is in RGB
        rgb_image_pil = image_pil.convert("RGB")
        image_tensor = torch.from_numpy(
            np.array(rgb_image_pil).astype(np.float32) / 255.0
        ).unsqueeze(0)

        logging.info("✅ Nilor-Nodes (MediaStreamInput): Image processing successful.")
        return (image_tensor,)

    def _process_video(self, video_bytes):
        logging.info("ℹ️\u2009 Nilor-Nodes (MediaStreamInput): Processing as video...")
        frames = []
        with imageio.get_reader(io.BytesIO(video_bytes), format="mp4") as reader:
            for frame in reader:
                # Convert frame to RGB PIL Image and then to tensor
                pil_image = Image.fromarray(frame).convert("RGB")
                numpy_image = np.array(pil_image).astype(np.float32) / 255.0
                tensor_frame = torch.from_numpy(numpy_image)
                frames.append(tensor_frame)

        if not frames:
            raise ValueError(
                "[🛑] Nilor-Nodes (MediaStreamInput): No frames could be read from the video."
            )

        # Stack frames into a single tensor (batch of images)
        video_tensor = torch.stack(frames)

        logging.info(
            f"✅ Nilor-Nodes (MediaStreamInput): Video processing successful. Image Shape: {video_tensor.shape}"
        )
        return (video_tensor,)


# --- MediaStreamOutput: Universal Media Uploader & SQS Notifier ---
class MediaStreamOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {"default": "default_output", "multiline": False},
                ),
                "images": ("IMAGE",),
                "format": (["png", "mp4"],),
                "framerate": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1}),
                "content_id": (
                    "STRING",
                    {"default": "<auto-filled by system>", "multiline": False},
                ),
                "venue": (
                    "STRING",
                    {"default": "<auto-filled by system>", "multiline": False},
                ),
                "canvas": (
                    "STRING",
                    {"default": "<auto-filled by system>", "multiline": False},
                ),
                "scene": (
                    "STRING",
                    {"default": "<auto-filled by system>", "multiline": False},
                ),
                "job_completions_queue_url": (
                    "STRING",
                    {"multiline": True, "default": "<auto-filled by system>"},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "upload_and_notify"
    OUTPUT_NODE = True
    CATEGORY = category + subcategories["streaming"]

    def upload_and_notify(
        self,
        images,
        format,
        content_id,
        venue,
        canvas,
        scene,
        job_completions_queue_url,
        framerate,
        output_name: str = "default_output",
        prompt=None,
        extra_pnginfo=None,
    ):
        if not content_id:
            raise ValueError(
                "[🛑] Nilor-Nodes (MediaStreamOutput): content_id is a required input for MediaStreamOutput."
            )

        # No longer need to parse output_object_keys since we use storage_ids directly

        # Upload the media using Brain API client
        brain_client = get_brain_api_client()
        storage_result = None

        if format == "png":
            storage_result = self._upload_image(images[0], brain_client, output_name)
        elif format == "mp4":
            storage_result = self._upload_video(
                images, brain_client, framerate, output_name
            )

        # Use the storage_id from the upload result for the SQS message
        if not storage_result:
            logging.error(
                f"🛑\u2009 Nilor-Nodes (MediaStreamOutput): FATAL -- Upload failed or no storage_id returned."
            )
            # Send an empty dictionary to signal failure.
            final_outputs_for_sqs = {}
        else:
            # Use storage_id directly (it's now a string, not a dict)
            storage_id = storage_result
            final_outputs_for_sqs = {output_name: storage_id}

        # After upload, send the filtered dictionary of outputs to the SQS queue.
        completion_message = {
            "content_id": content_id,
            "status": "completed",
            "venue": venue,
            "canvas": canvas,
            "scene": scene,
            "outputs": final_outputs_for_sqs,
        }

        try:
            # Re-initialize the client inside the execution to ensure it picks up env vars correctly.
            sqs_client = boto3.client(
                "sqs",
                endpoint_url=os.getenv("SQS_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "local"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "local"),
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
            logging.info(
                f"ℹ️\u2009 Nilor-Nodes (MediaStreamOutput): Sending completion message for content {content_id} to queue: {job_completions_queue_url}"
            )
            sqs_client.send_message(
                QueueUrl=job_completions_queue_url,
                MessageBody=json.dumps(completion_message),
            )
            logging.info(
                "✅ Nilor-Nodes (MediaStreamOutput): Completion message sent successfully."
            )
        except Exception as e:
            logging.error(
                f"🛑\u2009 Nilor-Nodes (MediaStreamOutput): Failed to send completion message to SQS: {e}"
            )
            raise  # Re-raise to fail the ComfyUI job

        return {"ui": {"images": []}}

    def _upload_image(self, image_tensor, brain_client, output_name):
        logging.info(
            "ℹ️\u2009 Nilor-Nodes (MediaStreamOutput): Uploading as PNG image..."
        )
        i = 255.0 * image_tensor.cpu().numpy()
        img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG", compress_level=4)
        buffer.seek(0)

        filename = f"{output_name}.png"
        minio_endpoint = os.getenv("MINIO_ENDPOINT")
        if not minio_endpoint:
            raise ValueError(
                "MINIO_ENDPOINT environment variable is required but not set"
            )

        # Get presigned upload URL
        upload_url_response = brain_client.get_presigned_upload_url(
            filename, "image/png", minio_endpoint
        )
        upload_url = upload_url_response["upload_url"]
        storage_id = upload_url_response["storage_id"]

        # Upload directly to MinIO
        buffer.seek(0)
        upload_response = requests.put(
            upload_url,
            data=buffer.getvalue(),
            headers={"Content-Type": "image/png"},
            timeout=300,
        )
        upload_response.raise_for_status()

        logging.info(
            f"✅ Nilor-Nodes (MediaStreamOutput): PNG image uploaded successfully. Storage ID: {storage_id}"
        )
        return storage_id

    def _upload_video(self, image_batch_tensor, brain_client, framerate, output_name):
        logging.info(
            f"ℹ️\u2009 Nilor-Nodes (MediaStreamOutput): Uploading as MP4 video. Frame count: {len(image_batch_tensor)}"
        )
        frames = []
        for image_tensor in image_batch_tensor:
            i = 255.0 * image_tensor.cpu().numpy()
            frame = np.clip(i, 0, 255).astype(np.uint8)
            frames.append(frame)

        buffer = io.BytesIO()
        imageio.mimwrite(buffer, frames, format="mp4", fps=framerate, quality=8)
        buffer.seek(0)

        filename = f"{output_name}.mp4"
        minio_endpoint = os.getenv("MINIO_ENDPOINT")
        if not minio_endpoint:
            raise ValueError(
                "MINIO_ENDPOINT environment variable is required but not set"
            )

        # Get presigned upload URL
        upload_url_response = brain_client.get_presigned_upload_url(
            filename, "video/mp4", minio_endpoint
        )
        upload_url = upload_url_response["upload_url"]
        storage_id = upload_url_response["storage_id"]

        # Upload directly to MinIO
        buffer.seek(0)
        upload_response = requests.put(
            upload_url,
            data=buffer.getvalue(),
            headers={"Content-Type": "video/mp4"},
            timeout=300,
        )
        upload_response.raise_for_status()

        logging.info(
            f"✅ Nilor-Nodes (MediaStreamOutput): MP4 video uploaded successfully. Storage ID: {storage_id}"
        )
        return storage_id


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "MediaStreamInput": MediaStreamInput,
    "MediaStreamOutput": MediaStreamOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaStreamInput": "👺 Media Stream Input (Storage)",
    "MediaStreamOutput": "👺 Media Stream Output (Storage)",
}

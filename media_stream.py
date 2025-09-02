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

# --- Load Environment Variables ---
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
dotenv_path = os.path.join(current_dir, '.env')
# Load the .env file
load_dotenv(dotenv_path=dotenv_path)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                "input_name": ("STRING", {"default": "default_input", "multiline": False}),
                "format": (["image", "image_batch", "video"],),
                "presigned_download_url": ("STRING", {
                    "multiline": True,
                    "default": "<auto-filled by system>"
                }),
            },
            "hidden": {
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "download"
    CATEGORY = category + subcategories["streaming"]

    def download(self, presigned_download_url: str, format: str, input_name: str = "default_input"):
        logging.info(f"MediaStreamInput: Downloading from {presigned_download_url} for input '{input_name}' with format '{format}'")
        try:
            # Two-phase download for batches: manifest first, then assets
            if format == 'image_batch':
                manifest_response = requests.get(presigned_download_url, timeout=60)
                manifest_response.raise_for_status()
                manifest = manifest_response.json()
                
                logging.info(f"Processing manifest for '{manifest.get('input_name')}' with {len(manifest.get('files', []))} assets.")
                
                # Sort files by sequence number to ensure correct order
                sorted_files = sorted(manifest.get('files', []), key=lambda x: x.get('sequence', 0))
                
                # Download all assets in parallel
                asset_responses = []
                for file_info in sorted_files:
                    try:
                        resp = requests.get(file_info['presigned_url'], timeout=180)
                        resp.raise_for_status()
                        asset_responses.append(resp.content)
                    except requests.RequestException as e:
                        logging.error(f"Failed to download asset {file_info.get('filename')}: {e}")
                        raise  # Re-raise to fail the entire process
                
                return self._process_image_batch(asset_responses)

            # --- Single-file download ---
            response = requests.get(presigned_download_url, timeout=180)
            response.raise_for_status()
            media_bytes = response.content
            
            if format == 'video':
                return self._process_video(media_bytes)
            elif format == 'image':
                return self._process_image(media_bytes)
            else:
                # Should not happen if UI choices are respected
                raise ValueError(f"Unsupported format '{format}' for single media download.")

        except requests.RequestException as e:
            logging.error(f"MediaStreamInput: Failed to download file: {e}")
            return (None,)
        except Exception as e:
            logging.error(f"MediaStreamInput: Failed to process media: {e}")
            return (None,)

    def _process_image_batch(self, image_bytes_list):
        logging.info(f"Processing image batch with {len(image_bytes_list)} images...")
        output_images = []

        for image_bytes in image_bytes_list:
            image_pil = Image.open(io.BytesIO(image_bytes))
            
            rgb_image_pil = image_pil.convert("RGB")
            image_tensor = torch.from_numpy(np.array(rgb_image_pil).astype(np.float32) / 255.0).unsqueeze(0)
            
            output_images.append(image_tensor)
        
        # Concatenate along the batch dimension (dim=0)
        images_tensor = torch.cat(output_images, dim=0)
        
        logging.info(f"Image batch processing successful. Batch shape: {images_tensor.shape}")
        return (images_tensor,)

    def _process_image(self, image_bytes):
        logging.info("Processing as image...")
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # Ensure image is in RGB
        rgb_image_pil = image_pil.convert("RGB")
        image_tensor = torch.from_numpy(np.array(rgb_image_pil).astype(np.float32) / 255.0).unsqueeze(0)
        
        logging.info("Image processing successful.")
        return (image_tensor,)

    def _process_video(self, video_bytes):
        logging.info("Processing as video...")
        frames = []
        with imageio.get_reader(io.BytesIO(video_bytes), format='mp4') as reader:
            for frame in reader:
                # Convert frame to RGB PIL Image and then to tensor
                pil_image = Image.fromarray(frame).convert("RGB")
                numpy_image = np.array(pil_image).astype(np.float32) / 255.0
                tensor_frame = torch.from_numpy(numpy_image)
                frames.append(tensor_frame)
        
        if not frames:
            raise ValueError("No frames could be read from the video.")

        # Stack frames into a single tensor (batch of images)
        video_tensor = torch.stack(frames)
        
        logging.info(f"Video processing successful. Image Shape: {video_tensor.shape}")
        return (video_tensor,)


# --- MediaStreamOutput: Universal Media Uploader & SQS Notifier ---
class MediaStreamOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_name": ("STRING", {"default": "default_output", "multiline": False}),
                "images": ("IMAGE",),
                "format": (["png", "mp4"],),
                "framerate": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1}),
                "job_id": ("STRING", {"default": "<auto-filled by system>", "multiline": False}),
                "presigned_upload_url": ("STRING", {
                    "multiline": True,
                    "default": "<auto-filled by system>"
                }),
                "job_completions_queue_url": ("STRING", {
                    "multiline": True,
                    "default": "<auto-filled by system>"
                }),
                "output_object_keys": ("STRING", {
                    "multiline": False,
                    "default": "<auto-filled by system>"
                }),
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



    def upload_and_notify(self, images, format, job_id, presigned_upload_url, job_completions_queue_url, output_object_keys, framerate, output_name: str="default_output", prompt=None, extra_pnginfo=None):
        if not job_id:
            raise ValueError("job_id is a required input for MediaStreamOutput.")

        # The `output_object_keys` is received as a string representation of a dictionary.
        # We must parse it back into a dictionary.
        final_outputs_dict = {}
        try:
            # The string may use single quotes, so we replace them for valid JSON.
            final_outputs_dict = json.loads(output_object_keys.replace("'", '"'))
        except Exception as e:
            logging.error(f"FATAL: Could not parse output_object_keys from string: {output_object_keys}. Error: {e}")
            final_outputs_dict = {} # Send empty dict on failure.
        
        # The presigned_upload_url provided to this node is specific to its output_name.
        # We don't need to re-select it. We just need to perform the upload.
        if format == "png":
            self._upload_image(images[0], presigned_upload_url)
        elif format == "mp4":
            self._upload_video(images, presigned_upload_url, framerate)
        
        # This node is responsible for a single output. We find its corresponding object key.
        output_key_for_this_node = final_outputs_dict.get(output_name)
        if not output_key_for_this_node:
            logging.error(f"FATAL: Could not find object key for output name '{output_name}' in output_object_keys.")
            # Send an empty dictionary to signal failure.
            final_outputs_for_sqs = {}
        else:
            final_outputs_for_sqs = {output_name: output_key_for_this_node}

        # After upload, send the filtered dictionary of outputs to the SQS queue.
        completion_message = {
            "job_id": job_id,
            "status": "completed",
            "outputs": final_outputs_for_sqs
        }
        
        try:
            # Re-initialize the client inside the execution to ensure it picks up env vars correctly.
            sqs_client = boto3.client(
                'sqs',
                endpoint_url=os.getenv("SQS_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "local"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "local"),
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            )
            logging.info(f"Sending completion message for job {job_id} to queue: {job_completions_queue_url}")
            sqs_client.send_message(
                QueueUrl=job_completions_queue_url,
                MessageBody=json.dumps(completion_message)
            )
            logging.info("Completion message sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send completion message to SQS: {e}")
            raise # Re-raise to fail the ComfyUI job
            
        return {"ui": {"images": []}}

    def _upload_image(self, image_tensor, url):
        logging.info("Uploading as PNG image...")
        i = 255. * image_tensor.cpu().numpy()
        img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG', compress_level=4)
        buffer.seek(0)
        
        self._perform_upload(buffer, url, 'image/png')

    def _upload_video(self, image_batch_tensor, url, framerate):
        logging.info(f"Uploading as MP4 video. Frame count: {len(image_batch_tensor)}")
        frames = []
        for image_tensor in image_batch_tensor:
            i = 255. * image_tensor.cpu().numpy()
            frame = np.clip(i, 0, 255).astype(np.uint8)
            frames.append(frame)

        buffer = io.BytesIO()
        imageio.mimwrite(buffer, frames, format='mp4', fps=framerate, quality=8)
        buffer.seek(0)

        self._perform_upload(buffer, url, 'video/mp4')

    def _perform_upload(self, buffer, url, content_type):
        try:
            logging.info(f"Uploading to {url} with Content-Type: {content_type}")
            headers = {'Content-Type': content_type}
            response = requests.put(url, data=buffer.read(), headers=headers, timeout=300)
            response.raise_for_status()
            logging.info("Upload successful.")
        except requests.RequestException as e:
            logging.error(f"MediaStreamOutput: Failed to upload media: {e}")
            raise
        except Exception as e:
            logging.error(f"MediaStreamOutput: Failed to process and upload media: {e}")
            raise


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "MediaStreamInput": MediaStreamInput,
    "MediaStreamOutput": MediaStreamOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaStreamInput": "👺 Media Stream Input (URL)",
    "MediaStreamOutput": "👺 Media Stream Output (URL)",
}

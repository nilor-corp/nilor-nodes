import torch
import numpy as np
from PIL import Image
import requests
import io
import logging
import imageio.v2 as imageio
import mimetypes

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
                "presigned_download_url": ("STRING", {
                    "multiline": True, 
                    "default": "http://example.com/image.png"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "download"
    CATEGORY = category + subcategories["streaming"]

    def download(self, presigned_download_url: str):
        logging.info(f"MediaStreamInput: Downloading from {presigned_download_url}")
        try:
            response = requests.get(presigned_download_url, timeout=180)
            response.raise_for_status()
            media_bytes = response.content
            
            # Use the Content-Type header to determine the file type
            content_type = response.headers.get("Content-Type", "")
            logging.info(f"Detected Content-Type: {content_type}")

            if 'video' in content_type:
                return self._process_video(media_bytes)
            else: # Default to image processing
                return self._process_image(media_bytes)

        except requests.RequestException as e:
            logging.error(f"MediaStreamInput: Failed to download file: {e}")
            return (None, None)
        except Exception as e:
            logging.error(f"MediaStreamInput: Failed to process media: {e}")
            return (None, None)

    def _process_image(self, image_bytes):
        logging.info("Processing as image...")
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        output_images = []
        output_masks = []
        
        # Ensure image is in RGB
        rgb_image_pil = image_pil.convert("RGB")
        image_tensor = torch.from_numpy(np.array(rgb_image_pil).astype(np.float32) / 255.0).unsqueeze(0)
        
        if 'A' in image_pil.getbands():
            mask = torch.from_numpy(np.array(image_pil.getchannel('A')).astype(np.float32) / 255.0).unsqueeze(0)
            output_masks.append(mask)
        
        output_images.append(image_tensor)
        
        if not output_masks:
            mask = torch.zeros((1, image_pil.height, image_pil.width), dtype=torch.float32, device="cpu")
            output_masks.append(mask)
        
        images_tensor = torch.cat(output_images, dim=0)
        masks_tensor = torch.cat(output_masks, dim=0)
        
        logging.info("Image processing successful.")
        return (images_tensor, masks_tensor)

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
        
        # THE FIX: The mask must have the same batch dimension as the image tensor.
        batch_size, height, width, _ = video_tensor.shape
        mask_tensor = torch.zeros((batch_size, height, width), dtype=torch.float32, device="cpu")

        logging.info(f"Video processing successful. Image Shape: {video_tensor.shape}, Mask Shape: {mask_tensor.shape}")
        return (video_tensor, mask_tensor)


# --- MediaStreamOutput: Universal Media Uploader ---
class MediaStreamOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "format": (["png", "mp4"],),
                "presigned_upload_url": ("STRING", {
                    "multiline": True,
                    "default": "http://example.com/upload_here"
                }),
                "completion_webhook_url": ("STRING", {
                    "multiline": True,
                    "default": "http://example.com/webhook"
                }),
                "output_object_keys": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "upload"
    OUTPUT_NODE = True
    CATEGORY = category + subcategories["streaming"]

    def upload(self, images, format, presigned_upload_url, completion_webhook_url, output_object_keys, prompt=None, extra_pnginfo=None):
        if format == "png":
            # For "png", we upload the first image of the batch
            self._upload_image(images[0], presigned_upload_url)
        elif format == "mp4":
            self._upload_video(images, presigned_upload_url)
        
        # After all uploads are complete, prepare the webhook payload.
        webhook_payload = {"output_files": [output_object_keys]}
        
        # Send the final completion webhook POST request.
        try:
            logging.info(f"Sending completion webhook to: {completion_webhook_url}")
            requests.post(completion_webhook_url, json=webhook_payload, timeout=30)
        except requests.RequestException as e:
            # If the webhook fails, the job will eventually be marked 'lost' by the reconciler.
            logging.error(f"Failed to send completion webhook: {e}")
            
        return {"ui": {"images": []}}

    def _upload_image(self, image_tensor, url):
        logging.info("Uploading as PNG image...")
        i = 255. * image_tensor.cpu().numpy()
        img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG', compress_level=4)
        buffer.seek(0)
        
        self._perform_upload(buffer, url, 'image/png')

    def _upload_video(self, image_batch_tensor, url):
        logging.info(f"Uploading as MP4 video. Frame count: {len(image_batch_tensor)}")
        frames = []
        for image_tensor in image_batch_tensor:
            i = 255. * image_tensor.cpu().numpy()
            frame = np.clip(i, 0, 255).astype(np.uint8)
            frames.append(frame)

        buffer = io.BytesIO()
        # Use mimwrite for format-agnostic writing, though we specify mp4 here
        imageio.mimwrite(buffer, frames, format='mp4', fps=30, quality=8)
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
            raise # Re-raise the exception to make ComfyUI aware of the failure
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

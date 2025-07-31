import torch
import numpy as np
from PIL import Image
import requests
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

category = "Nilor Nodes 👺"
subcategories = {
    "streaming": "/Streaming",
}

class MediaStreamInput:
    """
    A custom node to download an image from a pre-signed URL and provide it as a tensor.
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
            
            # Open image from response content
            img_bytes = response.content
            image_pil = Image.open(io.BytesIO(img_bytes))

            # Convert PIL image to tensor
            output_images = []
            output_masks = []
            
            image_tensor = torch.from_numpy(np.array(image_pil).astype(np.float32) / 255.0).unsqueeze(0)
            
            if 'A' in image_pil.getbands():
                mask = torch.from_numpy(np.array(image_pil.getchannel('A')).astype(np.float32) / 255.0).unsqueeze(0)
                output_masks.append(mask)
                image_tensor = image_tensor[:, :, :, :3] # Drop alpha channel from image

            output_images.append(image_tensor)

            if not output_masks:
                 # Create a blank mask if one doesn't exist
                mask = torch.zeros((1, image_pil.height, image_pil.width), dtype=torch.float32, device="cpu")
                output_masks.append(mask)
            
            images_tensor = torch.cat(output_images, dim=0)
            masks_tensor = torch.cat(output_masks, dim=0)

            logging.info("MediaStreamInput: Download and processing successful.")
            return (images_tensor, masks_tensor)

        except requests.RequestException as e:
            logging.error(f"MediaStreamInput: Failed to download file: {e}")
            return (None, None)
        except Exception as e:
            logging.error(f"MediaStreamInput: Failed to process image: {e}")
            return (None, None)

class MediaStreamOutput:
    """
    A custom node to upload an image tensor to a pre-signed URL.
    """
    def __init__(self):
        self.output_dir = "output" # Not used directly, but good practice
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "presigned_upload_url": ("STRING", {
                    "multiline": True,
                    "default": "http://example.com/upload_here"
                }),
                "completion_webhook_url": ("STRING", {
                    "multiline": True,
                    "default": "http://example.com/webhook"
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "upload"
    OUTPUT_NODE = True
    CATEGORY = category + subcategories["streaming"]

    def upload(self, images, presigned_upload_url: str, completion_webhook_url: str, prompt=None, extra_pnginfo=None):
        try:
            # For simplicity, we'll upload the first image of the batch.
            # In a real scenario, this might loop and generate multiple upload URLs.
            image_tensor = images[0]
            
            # Convert tensor to PIL Image
            i = 255. * image_tensor.cpu().numpy()
            img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Save PIL image to a byte buffer as PNG
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG', compress_level=4)
            buffer.seek(0)
            
            logging.info(f"MediaStreamOutput: Uploading to {presigned_upload_url}")
            
            # Upload using a PUT request
            headers = {'Content-Type': 'image/png'}
            response = requests.put(presigned_upload_url, data=buffer, headers=headers, timeout=180)
            response.raise_for_status()
            
            logging.info("MediaStreamOutput: Upload successful.")

            # As per Minor Goal 3.2, the completion webhook is NOT called yet.
            # This logic will be added in a later step.
            # if completion_webhook_url:
            #     logging.info(f"MediaStreamOutput: Calling completion webhook: {completion_webhook_url}")
            #     # webhook_response = requests.post(completion_webhook_url, json={"status": "completed"})
            #     # webhook_response.raise_for_status()

            return {"ui": {"images": []}} # Required return for output nodes

        except requests.RequestException as e:
            logging.error(f"MediaStreamOutput: Failed to upload file: {e}")
            return {"ui": {"error": [str(e)]}}
        except Exception as e:
            logging.error(f"MediaStreamOutput: Failed to process and upload image: {e}")
            return {"ui": {"error": [str(e)]}}

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "MediaStreamInput": MediaStreamInput,
    "MediaStreamOutput": MediaStreamOutput,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaStreamInput": "Media Stream Input (URL)",
    "MediaStreamOutput": "Media Stream Output (URL)",
}

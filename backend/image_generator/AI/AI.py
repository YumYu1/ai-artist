import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from PIL import Image
import io

# 모델 로컬 경로
model_path = "/app/local_models/stable-diffusion-3.5-medium"

def load_model():
    try:
        # Load the model without any GPU-specific configurations
        model = SD3Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.float32  # Use CPU-compatible data type
        )

        # Load the pipeline without GPU-specific optimizations
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            transformer=model,
            torch_dtype=torch.float32,
            device="cpu"  # Force the model to use CPU
        )
        
        # CPU 환경에서 효율적인 메모리 관리를 위해 CPU Offload 활성화 (선택 사항)
        pipeline.enable_model_cpu_offload()  # Optional for better memory management in CPU mode
        
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to load the model: {str(e)}") from e

def generate_image(pipeline, prompt, num_inference_steps=20, guidance_scale=7.5, height=512, width=512):
    try:
        # Generate the image with the provided prompt and parameters
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
        ).images[0]
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {str(e)}")

    # Save the image to an in-memory file
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return img_io

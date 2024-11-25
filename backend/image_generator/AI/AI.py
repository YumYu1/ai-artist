import torch
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from PIL import Image
import io

# 모델 로컬 경로
model_path = "./local_models/stable-diffusion-3.5-medium"

def load_model():
    try:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            transformer=model_nf4,
            torch_dtype=torch.bfloat16,
        )
        pipeline.enable_model_cpu_offload()
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to load the model: {str(e)}") from e

def generate_image(pipeline, prompt, num_inference_steps=20, guidance_scale=7.5, height=512, width=512):
    try:
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
        ).images[0]
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {str(e)}")

    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return img_io

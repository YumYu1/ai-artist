import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, send_file, jsonify, render_template
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
import io
from PIL import Image

# 모델 로컬 경로
model_path = "G:/local_models/stable-diffusion-3.5-medium"

app = Flask(__name__)

# 모델 로드
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
    #gpu의 메모리가 부족할 때 cpu로 모델로 옮겨 메모리 사용량을 줄일 수 있는 옵션
    #2070super에서는 사용하지 않으면 메모리 부족으로 실패함
    #pipeline.enable_model_cpu_offload()
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {str(e)}") from e

@app.route('/')
def index():
    return render_template('api.html')

@app.route('/image/inference', methods=['POST'])
def generate_image():
    # 요청이 JSON 데이터인지 확인
    data = request.json
    prompt = data.get('prompt', '')
    num_inference_steps = data.get('num_inference_steps', 20)
    guidance_scale = data.get('guidance_scale', 7.5)
    height = data.get('height', 512)
    width = data.get('width', 512)

    if not prompt:
        return jsonify(error="No prompt provided"), 400

    try:
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
        ).images[0]
    except Exception as e:
        return jsonify(error=f"Model inference failed: {str(e)}"), 500

    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
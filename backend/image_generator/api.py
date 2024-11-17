#pip install flask diffusers torch transformers huggingface-hub pillow accelerate 설치가 필요한 라이브러리
import os
os.environ["KMPDUPLICATELIBOK"] = "TRUE"

from flask import Flask, request, sendfile, jsonify, sendfromdirectory, render_template
import torch
import io
from PIL import Image
from huggingface_hub import login
login("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")

app = Flask(__name)

model_id = "stabilityai/stable-diffusion-3.5-medium"

모델 로드
try:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        torch_dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()
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
    num_inference_steps = data.get('num_inference_steps', 40)
    guidance_scale = data.get('guidance_scale', 7.5)

    if not prompt:
        return jsonify(error="No prompt provided"), 400

    try:
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height = 960,
            width = 528
        ).images[0]
    except Exception as e:
        return jsonify(error=f"Model inference failed: {str(e)}"), 500

    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name == '__main':
    app.run(debug=True)
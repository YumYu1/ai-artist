from flask import Flask, request, send_file, jsonify, render_template
from AI.AI import load_model, generate_image

app = Flask(__name__)

# 모델 로드
pipeline = load_model()

@app.route('/')
def index():
    return render_template('api.html')

@app.route('/image/inference', methods=['POST'])
def generate_image_route():
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
        img_io = generate_image(pipeline, prompt, num_inference_steps, guidance_scale, height, width)
    except Exception as e:
        return jsonify(error=f"Model inference failed: {str(e)}"), 500

    return send_file(img_io, mimetype='image/png')

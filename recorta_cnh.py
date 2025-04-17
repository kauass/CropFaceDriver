from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import fitz 
import mediapipe as mp

app = Flask(__name__)

def decode_image(base64_image):
    base64_cleaned = base64_image.split(',')[-1]
    img_data = base64.b64decode(base64_cleaned)

    if img_data[:4] == b'%PDF':
        pdf = fitz.open(stream=img_data, filetype="pdf")
        if len(pdf) == 0:
            return None
        page = pdf.load_page(0)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("jpg")
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@app.route('/detect-face', methods=['POST'])
def detect_face():
    try:
        data = request.get_json()
        base64_image = data.get('image')

        if not base64_image:
            return jsonify({'error': 'Imagem não fornecida.'}), 400

        image = decode_image(base64_image)

        if image is None or image.size == 0:
            return jsonify({'error': 'Imagem inválida ou não pôde ser lida.'}), 400

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)

            if not results.detections:
                return jsonify({'error': 'Nenhum rosto detectado.'}), 400

            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            ih, iw, _ = image.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            margin = 30
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            w = min(w + margin * 2, iw - x)
            h = min(h + margin * 2, ih - y)

            face_img = image[y:y+h, x:x+w]

            _, buffer = cv2.imencode('.jpg', face_img)
            out_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({'image': out_base64})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000)

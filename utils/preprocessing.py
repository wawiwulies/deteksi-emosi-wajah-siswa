import cv2

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
import numpy as np
from PIL import Image


EMOTION_LABELS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sad",
    5: "surprised",
    6: "neutral",
}

EMOTION_LABELS_ID = {
    "anger": "Marah",
    "disgust": "Tidak nyaman",
    "fear": "Takut",
    "happiness": "Senang",
    "sad": "Sedih",
    "surprised": "Terkejut",
    "neutral": "Netral",
}

EMOTION_SUGGESTIONS = {
    "anger": "Coba tarik napas perlahan, beri jeda sebentar, lalu lanjutkan aktivitas saat sudah lebih tenang.",
    "disgust": "Coba kenali hal yang membuat tidak nyaman, lalu beri jarak sejenak agar pikiran lebih jernih.",
    "fear": "Coba tenangkan diri, tarik napas, dan ceritakan kepada guru atau teman jika membutuhkan bantuan.",
    "happiness": "Pertahankan suasana positif ini dan gunakan energinya untuk belajar atau membantu teman.",
    "sad": "Tidak apa-apa merasa sedih. Coba istirahat sebentar, menulis perasaan, atau bercerita kepada orang terdekat.",
    "surprised": "Coba tenangkan diri dulu, pahami situasinya, lalu lanjutkan aktivitas dengan perlahan.",
    "neutral": "Kondisi kamu terlihat stabil. Ini waktu yang baik untuk mulai fokus belajar.",
}


def image_file_to_rgb(image_file):
    image = Image.open(image_file).convert("RGB")
    return np.array(image)


def detect_largest_face(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    # Tambahkan sedikit area sekitar wajah agar crop tidak terlalu ketat.
    pad = int(0.18 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(rgb_image.shape[1], x + w + pad)
    y2 = min(rgb_image.shape[0], y + h + pad)

    face_rgb = rgb_image[y1:y2, x1:x2]
    return face_rgb, (int(x), int(y), int(w), int(h))


def draw_face_box(rgb_image, face_box):
    image = rgb_image.copy()
    if face_box is None:
        return image

    x, y, w, h = face_box
    cv2.rectangle(image, (x, y), (x + w, y + h), (93, 95, 239), 3)
    return image


def prepare_face_for_model(face_rgb):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, 48, 48, 1), resized


def predict_emotion(model, preprocessed_face):
    probabilities = model.predict(preprocessed_face, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    emotion_key = EMOTION_LABELS[predicted_index]
    confidence = float(probabilities[predicted_index])
    return emotion_key, confidence, probabilities

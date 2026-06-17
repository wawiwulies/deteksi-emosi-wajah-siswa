from pathlib import Path

import streamlit as st
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    ReLU,
)
from tensorflow.keras.models import Model, load_model


MODEL_DIR = Path("models")
MODEL_EXTENSIONS = ("*.keras", "*.h5")
NOTEBOOK_EXTENSIONS = ("*.ipynb",)
PREFERRED_MODEL_NAMES = (
    "facial_emotion_cnn_retrained.keras",
    "facial_emotion_cnn_best.keras",
    "facial_emotion_cnn_model.keras",
    "Facial_expression_model.keras",
    "Facial_expression_weights.keras",
)


def _conv_bn_relu(x, filters, kernel_size=(3, 3), strides=(1, 1), padding="same"):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    return ReLU()(x)


def residual_block(x, filters):
    f1, f2, f3 = filters
    shortcut = x

    x = _conv_bn_relu(x, f1, kernel_size=(1, 1))
    x = _conv_bn_relu(x, f2, kernel_size=(3, 3))
    x = Conv2D(f3, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(f3, (1, 1), padding="same")(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    return ReLU()(x)


def identity_block(x, filters):
    f1, f2, f3 = filters
    shortcut = x

    x = _conv_bn_relu(x, f1, kernel_size=(1, 1))
    x = _conv_bn_relu(x, f2, kernel_size=(3, 3))
    x = Conv2D(f3, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    return ReLU()(x)


def build_fallback_cnn(input_shape=(48, 48, 1), num_classes=7):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = residual_block(x, [64, 128, 32])
    x = identity_block(x, [64, 128, 32])
    x = residual_block(x, [128, 256, 64])
    x = identity_block(x, [128, 256, 64])
    x = residual_block(x, [256, 512, 128])
    x = identity_block(x, [256, 512, 128])

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="fallback_emotion_cnn")


def find_model_file():
    for name in PREFERRED_MODEL_NAMES:
        model_path = MODEL_DIR / name
        if model_path.exists():
            return model_path

    for pattern in MODEL_EXTENSIONS:
        files = sorted(MODEL_DIR.glob(pattern))
        if files:
            return files[0]
    return None


@st.cache_resource(show_spinner=False)
def load_emotion_model(model_path):
    model_path = Path(model_path)

    try:
        return load_model(model_path), f"Model berhasil dimuat dari `{model_path}`."
    except Exception as model_error:
        fallback_model = build_fallback_cnn()

        try:
            fallback_model.load_weights(model_path)
            return (
                fallback_model,
                "File model terbaca sebagai weights. Arsitektur fallback berhasil dibuat dan weights berhasil dimuat.",
            )
        except Exception as weights_error:
            message = (
                "Model belum bisa dimuat. Pastikan file `.keras` atau `.h5` adalah full model "
                "Keras, atau weights yang cocok dengan arsitektur fallback.\n\n"
                f"Error load_model: {model_error}\n\n"
                f"Error load_weights: {weights_error}"
            )
            raise RuntimeError(message) from weights_error


def list_model_candidates():
    MODEL_DIR.mkdir(exist_ok=True)
    candidates = []
    for pattern in MODEL_EXTENSIONS:
        candidates.extend(sorted(MODEL_DIR.glob(pattern)))
    return sorted(
        candidates,
        key=lambda path: (
            PREFERRED_MODEL_NAMES.index(path.name)
            if path.name in PREFERRED_MODEL_NAMES
            else len(PREFERRED_MODEL_NAMES),
            path.name.lower(),
        ),
    )


def list_notebook_candidates():
    MODEL_DIR.mkdir(exist_ok=True)
    candidates = []
    for pattern in NOTEBOOK_EXTENSIONS:
        candidates.extend(sorted(MODEL_DIR.glob(pattern)))
    return candidates

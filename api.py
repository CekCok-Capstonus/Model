from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, regularizers

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_BILSTM   = os.path.join(BASE_DIR, "src", "models",    "best_model_bilstm.keras")
MODEL_GRU      = os.path.join(BASE_DIR, "src", "models",    "best_model_gru.keras")
MODEL_CNN      = os.path.join(BASE_DIR, "src", "models",    "best_model_cnn_bilstm.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "src", "tokenizer", "tokenizer.pkl")
TFIDF_PATH     = os.path.join(BASE_DIR, "src", "tokenizer", "tfidf_vectorizer.pkl")
MAX_LEN_BILSTM = 200
MAX_LEN_GRU    = 250
MAX_LEN_CNN    = 250
THRESHOLD = 0.5

# ─── Custom Attention Layer ───────────────────────────────────────────────────
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            regularizer=regularizers.l2(1e-4),
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

# ─── App State ────────────────────────────────────────────────────────────────
app_state: dict = {}

CUSTOM_OBJECTS = {"AttentionLayer": AttentionLayer}
VOCAB_SIZE: dict[str, int] = {}


def _check_files() -> None:
    missing = [
        p for p in [MODEL_BILSTM, MODEL_GRU, MODEL_CNN, TOKENIZER_PATH, TFIDF_PATH]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError("File tidak ditemukan:\n" + "\n".join(missing))


def _get_embedding_vocab_size(model: tf.keras.Model) -> int:
    """Ambil input_dim dari layer Embedding pertama di model."""
    for layer in model.layers:
        if "embedding" in layer.name.lower():
            return layer.get_config()["input_dim"]
    return 50000  


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_files()

    app_state["bilstm"] = tf.keras.models.load_model(
        MODEL_BILSTM, custom_objects=CUSTOM_OBJECTS, compile=False
    )
    app_state["gru"] = tf.keras.models.load_model(
        MODEL_GRU, custom_objects=CUSTOM_OBJECTS, compile=False
    )
    app_state["cnn"] = tf.keras.models.load_model(
        MODEL_CNN, custom_objects=CUSTOM_OBJECTS, compile=False
    )

    VOCAB_SIZE["bilstm"] = _get_embedding_vocab_size(app_state["bilstm"])
    VOCAB_SIZE["gru"]    = _get_embedding_vocab_size(app_state["gru"])
    VOCAB_SIZE["cnn"]    = _get_embedding_vocab_size(app_state["cnn"])
    print(f"📐 Vocab sizes — bilstm:{VOCAB_SIZE['bilstm']} | gru:{VOCAB_SIZE['gru']} | cnn:{VOCAB_SIZE['cnn']}")

    with open(TOKENIZER_PATH, "rb") as f:
        app_state["tokenizer"] = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        app_state["tfidf"] = pickle.load(f)

    print("✅ Semua model & vectorizer berhasil dimuat.")
    yield
    app_state.clear()
    VOCAB_SIZE.clear()

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detection API",
    description=(
        "Deteksi berita palsu menggunakan ensemble 3 model: "
        "BiLSTM + Attention, GRU + Attention, dan CNN-BiLSTM + TF-IDF. "
        "Confidence score adalah rata-rata output sigmoid ketiga model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: Annotated[str, Field(min_length=1)]

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "Scientists discover new treatment for cancer."}]
        }
    }


class ModelScores(BaseModel):
    bilstm:     float
    gru:        float
    cnn_bilstm: float


class PredictResponse(BaseModel):
    text:         str
    label:        str   
    confidence:   float 
    is_fake:      bool
    model_scores: ModelScores


class BatchPredictRequest(BaseModel):
    texts: Annotated[list[str], Field(min_length=1, max_length=50)]

    model_config = {
        "json_schema_extra": {
            "examples": [{"texts": ["Breaking: aliens land in Jakarta!", "WHO releases new guidelines."]}]
        }
    }


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _pad(texts: list[str], maxlen: int, vocab_size: int) -> np.ndarray:
    """Tokenize, pad, lalu clip index agar tidak melebihi vocab embedding."""
    sequences = app_state["tokenizer"].texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, truncating="post", padding="post")
    return np.clip(padded, 0, vocab_size - 1)


def _tfidf(texts: list[str]) -> np.ndarray:
    return app_state["tfidf"].transform(texts).toarray().astype(np.float32)


def _predict_single(text: str) -> tuple[float, float, float]:
    """Prediksi satu teks, kembalikan (prob_bilstm, prob_gru, prob_cnn)."""
    pad_bilstm = _pad([text], MAX_LEN_BILSTM, VOCAB_SIZE["bilstm"])
    pad_gru    = _pad([text], MAX_LEN_GRU,    VOCAB_SIZE["gru"])
    pad_cnn    = _pad([text], MAX_LEN_CNN,    VOCAB_SIZE["cnn"])
    tfidf_feat = _tfidf([text])

    p_bilstm = float(app_state["bilstm"].predict(pad_bilstm, verbose=0).flatten()[0])
    p_gru    = float(app_state["gru"].predict(pad_gru,       verbose=0).flatten()[0])
    p_cnn    = float(app_state["cnn"].predict(
        {"input_sequence": pad_cnn, "input_tfidf": tfidf_feat}, verbose=0
    ).flatten()[0])

    return p_bilstm, p_gru, p_cnn


def _build_response(text: str, p_bilstm: float, p_gru: float, p_cnn: float) -> PredictResponse:
    avg     = float(np.mean([p_bilstm, p_gru, p_cnn]))
    is_fake = avg >= THRESHOLD
    return PredictResponse(
        text=text,
        label="HOAKS" if is_fake else "FAKTA",
        confidence=round(avg, 4),
        is_fake=is_fake,
        model_scores=ModelScores(
            bilstm=round(p_bilstm, 4),
            gru=round(p_gru, 4),
            cnn_bilstm=round(p_cnn, 4),
        ),
    )


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Fake News Detection API berjalan."}


@app.get("/health", tags=["Health"])
def health():
    loaded = all(k in app_state for k in ["bilstm", "gru", "cnn", "tokenizer", "tfidf"])
    return {"status": "ok" if loaded else "error", "models_loaded": loaded}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Prediksi satu teks: ensemble 3 model, confidence = rata-rata sigmoid."""
    try:
        p_bilstm, p_gru, p_cnn = _predict_single(request.text)
        return _build_response(request.text, p_bilstm, p_gru, p_cnn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """Prediksi beberapa teks sekaligus (maks. 50): ensemble 3 model."""
    try:
        results = [
            _build_response(text, *_predict_single(text))
            for text in request.texts
        ]
        return BatchPredictResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

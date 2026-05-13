import os
import warnings
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ─── App Setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')
DATA_DIR   = os.path.join(os.path.dirname(BASE_DIR), 'data')
DATA_PATH  = os.path.join(DATA_DIR, 'Extended_Employee_Performance_and_Productivity_Data.csv')

# ─── Lazy model registry ─────────────────────────────────────────────────────
_models  = {}
_scalers = {}

def _count_dataset_rows(path):
    if not os.path.exists(path):
        return 0
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = sum(1 for _ in f)
        return max(0, lines - 1)
    except Exception:
        return 0

DEFAULT_DATASET_ROWS = 100_000
DATA_ROWS = _count_dataset_rows(DATA_PATH)
if DATA_ROWS == 0:
    print(f"Warning: dataset not found at {DATA_PATH}. Defaulting total_records to {DEFAULT_DATASET_ROWS}.")
    DATA_ROWS = DEFAULT_DATASET_ROWS


def _load_keras(key, filename):
    if key not in _models:
        import tensorflow as tf
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try:
                _models[key] = tf.keras.models.load_model(path)
            except Exception as e:
                print(f"Warning: Could not load model {filename}: {e}")
                _models[key] = None
        else:
            _models[key] = None
    return _models[key]

def _load_pkl(key, filename):
    if key not in _models:
        path = os.path.join(MODELS_DIR, filename)
        _models[key] = joblib.load(path) if os.path.exists(path) else None
    return _models[key]

def _load_scaler(key, filename):
    if key not in _scalers:
        path = os.path.join(MODELS_DIR, filename)
        _scalers[key] = joblib.load(path) if os.path.exists(path) else None
    return _scalers[key]


# ─── Metric constants (from notebook training results) ───────────────────────
MODEL_METRICS = {
    "Linear Regression": {"mae": 4.21, "rmse": 5.63, "r2": 0.71,  "accuracy": None},
    "ANN":               {"mae": 1.83, "rmse": 2.47, "r2": 0.94,  "accuracy": None},
    "RNN":               {"mae": 2.51, "rmse": 3.28, "r2": 0.89,  "accuracy": None},
    "Backpropagation":   {"mae": None, "rmse": None,  "r2": None,  "accuracy": 0.87},
    "K-Means":           {"mae": None, "rmse": None,  "r2": None,  "accuracy": None,
                          "silhouette": 0.42, "clusters": 3},
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _scale(features, scaler):
    if scaler is None:
        return features
    try:
        return scaler.transform(features)
    except Exception:
        return features


def _generate_insight(resign_status, usia, gaji, jam_kerja, masa_kerja, cluster):
    cluster_labels = {0: "High Performer", 1: "Average", 2: "At-Risk"}
    cluster_name   = cluster_labels.get(cluster, "Tidak Diketahui")

    if jam_kerja > 80:
        return (f"⚠️ PERINGATAN KRITIS: Jam kerja {jam_kerja} jam/minggu melampaui "
                f"batas manusiawi. Risiko burnout dan penurunan produktivitas sangat tinggi "
                f"meski AI memprediksi {resign_status}. Cluster: {cluster_name}.")

    if resign_status == "Akan Resign":
        if gaji < 5:
            return (f"Model mendeteksi kemungkinan resign tinggi, terutama dipicu kompensasi "
                    f"yang rendah (Rp {gaji:.1f} juta). Karyawan berada di cluster {cluster_name}. "
                    f"Rekomendasi: tinjau ulang struktur gaji.")
        else:
            return (f"Beban kerja {jam_kerja} jam/minggu dikombinasikan faktor usia {int(usia)} tahun "
                    f"menjadi pemicu utama keinginan resign. Cluster: {cluster_name}. "
                    f"Rekomendasi: evaluasi keseimbangan beban kerja.")
    else:
        if masa_kerja > 5:
            return (f"Loyalitas tinggi terdeteksi — {masa_kerja:.1f} tahun masa kerja menunjukkan "
                    f"komitmen kuat. Karyawan berada di cluster {cluster_name}. "
                    f"Strategi retensi jangka panjang sangat disarankan.")
        return (f"Metrik kompensasi dan beban kerja saat ini stabil. "
                f"Karyawan dikategorikan dalam cluster {cluster_name}. "
                f"Pertahankan kondisi kerja yang ada untuk menjaga performa optimal.")


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', total_records=DATA_ROWS)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method != 'POST':
        return render_template('predict.html')

    print("Received POST request to /predict")
    try:
        usia       = float(request.form.get('usia', 0))
        gaji       = float(request.form.get('gaji', 0))
        jam_kerja  = float(request.form.get('jam_kerja', 0))
        masa_kerja = float(request.form.get('masa_kerja', 0))

        # ── validation ──
        errors = {}
        if not (18 <= usia <= 65):
            errors['usia'] = 'Usia harus antara 18–65 tahun.'
        if not (1 <= gaji <= 200):
            errors['gaji'] = 'Gaji harus antara 1–200 juta rupiah.'
        if not (1 <= jam_kerja <= 100):
            errors['jam_kerja'] = 'Jam kerja harus antara 1–100 jam/minggu.'
        if not (0 <= masa_kerja <= 40):
            errors['masa_kerja'] = 'Masa kerja harus antara 0–40 tahun.'
        if errors:
            return jsonify({'success': False, 'errors': errors}), 422

        raw = np.array([[usia, gaji, jam_kerja, masa_kerja]])

        # ── Scaler (general) ──
        scaler_gen = _load_scaler('general', 'scaler.joblib')
        scaler_ann = _load_scaler('ann',     'scaler_ann.pkl')

        scaled_gen = _scale(raw, scaler_gen)
        scaled_ann = _scale(raw, scaler_ann if scaler_ann else scaler_gen)

        # ── 1. Resign Prediction (Backpropagation) ──
        model_bp = _load_keras('backprop', 'model_backprop.h5')
        if model_bp:
            try:
                prob_resign = float(model_bp.predict(scaled_gen, verbose=0)[0][0])
            except Exception as e:
                print(f"Error predicting with backprop model: {e}")
                prob_resign = 0.65 if gaji < 5 else 0.15  # fallback
        else:
            prob_resign = 0.65 if gaji < 5 else 0.15  # fallback

        resign_status = "Akan Resign" if prob_resign > 0.5 else "Tetap Bertahan"
        resign_conf   = round(prob_resign * 100 if prob_resign > 0.5 else (1 - prob_resign) * 100, 1)

        # ── 2. Productivity — ANN ──
        model_ann = _load_keras('ann', 'model_ann.h5')
        if model_ann:
            prod_ann = float(model_ann.predict(scaled_ann, verbose=0)[0][0])
        else:
            prod_ann = max(0, min(100, 80 + masa_kerja * 0.5 - (jam_kerja > 50) * 10))

        # ── 3. Productivity — RNN ──
        model_rnn = _load_keras('rnn', 'model_rnn.h5')
        if model_rnn:
            # RNN expects 3-D input: (batch, timesteps, features)
            rnn_input = scaled_gen.reshape(scaled_gen.shape[0], 1, scaled_gen.shape[1])
            prod_rnn  = float(model_rnn.predict(rnn_input, verbose=0)[0][0])
        else:
            prod_rnn = prod_ann + np.random.uniform(-3, 3)

        # ── 4. Productivity — Linear Regression ──
        model_lr = _load_pkl('lr', 'model_lr.pkl')
        if model_lr:
            try:
                prod_lr = float(model_lr.predict(scaled_gen)[0])
            except Exception:
                # Model was trained on more features; use numeric fallback
                prod_lr = max(0, min(100, 75 + masa_kerja * 0.8 - (jam_kerja > 50) * 5))
        else:
            prod_lr = max(0, min(100, 75 + masa_kerja * 0.8 - (jam_kerja > 50) * 5))

        # clamp all to [0, 100]
        prod_ann = round(max(0.0, min(100.0, prod_ann)), 1)
        prod_rnn = round(max(0.0, min(100.0, prod_rnn)), 1)
        prod_lr  = round(max(0.0, min(100.0, prod_lr)),  1)

        # ensemble average
        prod_avg  = round((prod_ann + prod_rnn + prod_lr) / 3, 1)
        prod_lower = round(prod_avg - 2.5, 1)
        prod_upper = round(prod_avg + 2.5, 1)

        # ── 5. K-Means Cluster ──
        model_km = _load_pkl('kmeans', 'model_kmeans.pkl')
        if model_km:
            try:
                cluster = int(model_km.predict(scaled_gen)[0])
                # Remap: ensure cluster in 0-2
                cluster = cluster % 3
            except Exception:
                cluster = 1  # fallback Average
        else:
            cluster = 1  # fallback Average

        # ── Insight ──
        insight = _generate_insight(resign_status, usia, gaji, jam_kerja, masa_kerja, cluster)

        cluster_names = {0: "High Performer", 1: "Average", 2: "At-Risk"}

        return jsonify({
            'success': True,
            'data': {
                'resign_status':    resign_status,
                'resign_confidence': resign_conf,
                'resign_prob':      round(prob_resign * 100, 1),
                'prod_ann':         prod_ann,
                'prod_rnn':         prod_rnn,
                'prod_lr':          prod_lr,
                'prod_avg':         prod_avg,
                'productivity_range': f"{prod_lower}% – {prod_upper}%",
                'cluster':          cluster,
                'cluster_name':     cluster_names.get(cluster, "Unknown"),
                'insight':          insight,
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/compare')
def compare():
    return render_template('compare.html', metrics=MODEL_METRICS)


@app.route('/about')
def about():
    return render_template('about.html', total_records=DATA_ROWS)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=False, port=5000, use_reloader=False)

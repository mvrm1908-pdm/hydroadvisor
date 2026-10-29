from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

app = Flask(__name__)
CORS(app)

rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
with open(MODEL_DIR / "label_map.json") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}
with open(MODEL_DIR / "feature_names.json") as f:
    feature_names = json.load(f)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Low-confidence threshold: below this, the top-1 class probability is not
# treated as a confident single-class call. See thesis Section 4.6.1.
CONFIDENCE_FLOOR = 0.50


@app.route("/")
def home():
    return open(BASE_DIR / "index.html", encoding="utf-8").read()


def call_ollama(system_prompt, messages):
    try:
        conversation = system_prompt + "\n\n"
        for msg in messages:
            role = "ASSISTANT" if msg["role"] == "assistant" else "USER"
            conversation += f"{role}: {msg['content']}\n"
        conversation += "ASSISTANT:"

        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "mistral",
                "prompt": conversation,
                "stream": False,
                "temperature": 0.3,
                "num_predict": 200,
            },
            timeout=30,
        )

        if response.status_code == 200:
            return response.json().get("response", "No response")
        return f"Ollama error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Run ollama serve in another terminal."
    except Exception as e:
        return f"Error: {str(e)}"


def compute_volumetric_efficiency(actual_flow, theoretical_flow):
    if theoretical_flow and theoretical_flow > 0:
        return round((actual_flow / theoretical_flow) * 100, 2)
    return None


def compute_health_score(pred_proba, vol_eff, system_efficiency=None):
    """
    Weighted composite per thesis Table 3.4:
      system efficiency (SE)            40%
      inverse severe-leakage probability 35%
      volumetric efficiency ratio        25%

    system_efficiency is an optional field (matches SE.txt_mean in the
    training data). If the operator has not supplied it, volumetric
    efficiency is used as the best-available proxy for both the SE and
    volumetric-efficiency terms (SE integrates volumetric, mechanical and
    thermal losses into a single scalar; see thesis Section 2.3.4).
    """
    p_severe = pred_proba[2]
    inverse_severe_score = (1 - p_severe) * 100
    vol_eff_score = vol_eff if vol_eff is not None else 95
    se_score = system_efficiency if system_efficiency is not None else vol_eff_score

    health_score = (0.40 * se_score) + (0.35 * inverse_severe_score) + (0.25 * vol_eff_score)
    return int(round(min(100, max(0, health_score))))


def get_urgency_tier(health_score):
    """
    Four-tier maintenance decision logic per thesis Table 3.4.
    Urgency is derived from health_score ONLY, never from the raw
    predicted class label, so the displayed tier always agrees with the
    numeric score shown to the user.
    """
    if health_score >= 75:
        return "NORMAL", "NORMAL OPERATION"
    elif health_score >= 60:
        return "WATCH", "PLAN MAINTENANCE"
    elif health_score >= 40:
        return "CAUTION", "EXPEDITED MAINTENANCE"
    else:
        return "URGENT", "IMMEDIATE ACTION REQUIRED"


def build_leakage_label(pred_class, pred_proba):
    """
    Bare class name is only shown when the model is reasonably confident.
    Below CONFIDENCE_FLOOR the three-class distribution is near-tied, and
    asserting a single class name would misrepresent the classifier's
    actual output. See thesis Section 4.6.1.
    """
    base_label = {0: "No Leakage", 1: "Weak Leakage", 2: "Severe Leakage"}[pred_class]
    confidence = max(pred_proba)
    if confidence < CONFIDENCE_FLOOR:
        return f"{base_label} (Low Confidence, Borderline Classification)"
    return base_label


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    if not body:
        return jsonify({"error": "No data"}), 400

    user_inputs = body.get("sensor_values", {})
    asset_id = body.get("asset_id", "UNASSIGNED")
    iso_4406_code = user_inputs.get("iso_4406_code")
    iso_4406_level = iso_4406_code if iso_4406_code and iso_4406_code != "Not measured" else "Not measured"

    # Healthy-baseline (Class 0) defaults from thesis Table 4.3, used only when a
    # field is left blank. 0 is not used as a fallback for physical readings,
    # since an unmeasured pressure/temperature/efficiency channel is not the
    # same thing as a channel reading zero.
    g = lambda key, default: user_inputs[key] if user_inputs.get(key) is not None else default

    ps1_mean = g("ps1_mean", 160.5)
    ps2_mean = g("ps2_mean", ps1_mean)
    ps3_mean = g("ps3_mean", 10.5)
    ps4_mean = g("ps4_mean", 2.1)
    ps5_mean = g("ps5_mean", 8.5)
    ps6_mean = g("ps6_mean", ps1_mean)
    ps_values = [ps1_mean, ps2_mean, ps3_mean, ps4_mean, ps5_mean, ps6_mean]
    ps_multi_mean = sum(ps_values) / len(ps_values)
    ps_multi_var = sum((v - ps_multi_mean) ** 2 for v in ps_values) / len(ps_values)
    ps_multi_max = max(ps_values)

    ts1_mean = g("ts1_mean", 45.5)
    ts2_mean = g("ts2_mean", ts1_mean)
    ts_gradient = g("ts_gradient", ts2_mean - ts1_mean)  # override if real value known

    system_efficiency = user_inputs.get("system_efficiency")
    se_for_index = system_efficiency if system_efficiency is not None else 95.0
    degrad_index = g("degrad_index", ts_gradient * (1 - se_for_index / 100))  # override if real value known

    input_map = {
        "PS1.txt_mean": ps1_mean,
        "PS1.txt_var": g("ps1_var", 220.6),
        "PS1.txt_drop_rate": g("ps1_drop_rate", 0.000075),
        "PS1.txt_crest": g("ps1_crest", 1.194),
        "PS1.txt_skew": g("ps1_skew", 0),
        "PS2.txt_mean": ps2_mean,
        "PS2.txt_var": g("ps2_var", 220.6),
        "PS3.txt_mean": ps3_mean,
        "PS4.txt_mean": ps4_mean,
        "PS5.txt_mean": ps5_mean,
        "PS6.txt_mean": ps6_mean,
        "eps1_mean": g("eps1_mean", 2469.9),
        "ps_multi_mean": ps_multi_mean,
        "ps_multi_var": ps_multi_var,
        "ps_multi_max": ps_multi_max,
        "VS1.txt_rms": g("vs1_rms", 0.611),
        "VS1.txt_peak": g("vs1_peak", 1.6),
        "VS1.txt_crest": g("vs1_crest", 1.4),
        "VS1.txt_skew": g("vs1_skew", 0),
        "VS1.txt_kurtosis": g("vs1_kurtosis", 0.85),
        "VS1.txt_entropy": g("vs1_entropy", 0),
        "VS1.txt_trend": g("vs1_trend", 0),
        "VS1.txt_freq_dom": g("vs1_freq_dom", 0),
        "VS1.txt_spec_cent": g("vs1_spec_cent", 0),
        "VS1.txt_wav_e": g("vs1_wav_e", 0),
        "VS1.txt_wav_ent": g("vs1_wav_ent", 0),
        "TS1.txt_mean": ts1_mean,
        "TS1.txt_grad": g("ts1_grad", 0),
        "TS2.txt_mean": ts2_mean,
        "ts_gradient": ts_gradient,
        "TS2.txt_trend": g("ts2_trend", 0),
        "CE.txt_mean": g("ce_mean", 31.7),
        "CP.txt_mean": g("cp_mean", 2.0),
        "SE.txt_mean": se_for_index,
        "FS1.txt_mean": g("fs1_mean", 6.5),
        "degrad_index": degrad_index,
    }
    feature_vector = [input_map.get(fname, 0) for fname in feature_names]

    X = np.array(feature_vector).reshape(1, -1)
    pred_class = int(rf_model.predict(X)[0])
    pred_proba = rf_model.predict_proba(X)[0].tolist()

    actual_flow = user_inputs.get("actual_flow")
    theoretical_flow = user_inputs.get("theoretical_flow")
    vol_eff = compute_volumetric_efficiency(actual_flow, theoretical_flow) if actual_flow else None

    health_score = compute_health_score(pred_proba, vol_eff, system_efficiency)
    urgency, action = get_urgency_tier(health_score)
    leakage_label = build_leakage_label(pred_class, pred_proba)

    return jsonify({
        "asset_id": asset_id,
        "pump_leakage_class": pred_class,
        "pump_leakage_label": leakage_label,
        "confidence": round(max(pred_proba) * 100, 1),
        "health_score": health_score,
        "volumetric_efficiency": vol_eff,
        "iso_4406_level": iso_4406_level,
        "maintenance_action": action,
        "urgency": urgency,
        "sensor_inputs": user_inputs,
    })


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json()
    messages = body.get("messages", [])
    prediction_context = body.get("prediction_context", None)
    if not messages:
        return jsonify({"error": "No messages"}), 400

    if prediction_context:
        context = (
            f"PUMP: {prediction_context.get('pump_leakage_label')} | "
            f"Health: {prediction_context.get('health_score')}/100 | "
            f"Urgency: {prediction_context.get('urgency')}"
        )
        if messages[0]["role"] == "user":
            messages[0]["content"] = context + " | User: " + messages[0]["content"]

    system_prompt = "You are HydroAdvisor. Answer only about hydraulic pumps. Use real data. Keep responses short."
    reply = call_ollama(system_prompt, messages)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    print("\nStarting HydroAdvisor on http://localhost:5000")
    print("Make sure ollama serve is running\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

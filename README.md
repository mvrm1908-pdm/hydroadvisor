# HydroAdvisor: Hydraulic Pump Leakage Classification & Maintenance Support

**Kurzfassung (Deutsch):** Prototyp aus meiner Masterarbeit: Klassifikation der inneren Leckage von Hydraulikpumpen (keine / schwach / stark) auf Basis von Prüfstandsdaten, Ableitung einer Instandhaltungsempfehlung und lokal betriebener Chat zur Erläuterung. Validiert ausschließlich auf einem öffentlichen Labordatensatz, kein Serienprodukt.

Prototype from my M.Sc. thesis *"A Conceptual Framework for Predictive Maintenance of Hydraulic Pumps Using Sensor-Based Condition Monitoring"* (FOM Hochschule Essen, 2026, grade 2.0).

It classifies hydraulic pump **internal leakage** (none / weak / severe) from test-rig sensor data, converts the result into a maintenance action (NORMAL / WATCH / CAUTION / URGENT), and offers a locally hosted LLM chat for explanations.

> **Status: research prototype.** Trained and tested on one public laboratory dataset. Not validated on field data and not intended for production use.

## What it does
1. **Feature pipeline (notebook):** 36 features from pressure, flow, motor power, temperature, vibration and efficiency channels, each mapped to a physical degradation mechanism (internal leakage, abrasive wear, cavitation, contamination).
2. **Classifiers:** Random Forest (deployed) and SVM-RBF (comparison).
3. **Decision logic:** health score = 40 % system efficiency + 35 % (1 − P(severe leakage)) + 25 % volumetric efficiency → four urgency tiers.
4. **Web app:** Flask backend + HTML frontend; optional chat via a local Mistral 7B model (Ollama). No cloud API, runs offline.

## Results (UCI dataset, 1,449 stable cycles, 290-cycle hold-out test set)
| Model | Weighted F1 | MCC | Inference / sample |
|---|---|---|---|
| Random Forest | 0.993 | 0.990 | 0.14 ms |
| SVM (RBF) | 0.962 | 0.943 | 0.05 ms |

5-fold CV (RF): weighted F1 0.991 ± 0.005. Removing all vibration features did not change test F1.

**How to read these numbers:** the dataset comes from a single test rig with a fixed load cycle, and leakage is induced in discrete steps. Train and test cycles were split randomly, so neighbouring, near-identical cycles can appear in both sets. The scores therefore show that the pipeline works on this benchmark, not how it would perform on real machines. See *Limitations*.

## Run it
```bash
git clone https://github.com/mvrm1908-pdm/hydroadvisor.git
cd hydroadvisor
python -m venv .venv && .venv\Scripts\activate      # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python app/app.py                                   # open http://localhost:5000
```
Optional chat: install [Ollama](https://ollama.com), run `ollama pull mistral` and keep `ollama serve` running. Without it, classification still works.

To reproduce training: download the dataset (see `data/README.md`) and run `notebooks/hydraulic_pm_pipeline.ipynb`.

## Repository structure
```
app/         Flask backend (app.py), frontend (index.html), feature descriptions
models/      trained Random Forest (36 features), SVM + scaler, label map, feature names
notebooks/   full pipeline: preprocessing, feature engineering, training, evaluation
docs/figures thesis figures (filter effect, feature importance, CV, confusion matrices …)
data/        download instructions (dataset not included)
```

## Limitations (known, documented)
- **One lab dataset:** fixed duty cycle, constant temperature, artificially induced leakage. No field validation.
- **Random split** on sequential cycles probably makes test scores optimistic. A grouped / time-blocked split is planned.
- **Vibration channel is sampled at 1 Hz**, too slow for bearing or cavitation frequency analysis; vibration features carry little information here.
- **Manual input, no live data yet:** the app is currently an assessment tool with manual input. The model expects 36 features computed per 60-second load cycle, not raw sensor readings; empty fields are filled with healthy-baseline values, which lowers confidence. The architecture is designed so the input form can be replaced by an automated pipeline (data acquisition → cycle segmentation → feature extraction → prediction per cycle). This is not yet implemented, and deployment on a new machine requires retraining or recalibration with that machine's data.
- **The chat** receives only the classification label, health score and urgency tier. It does not retrieve documents (no RAG yet) and can give generic answers.
- Health-score weights and tier thresholds are design choices, not calibrated against maintenance cost data.

## Next steps
- Automated feature pipeline for live sensor data (e.g. OPC UA / MQTT input, cycle segmentation, per-cycle prediction)
- Grouped / time-blocked cross-validation to quantify the split effect
- Per-prediction explanations (SHAP)
- Retrieval over maintenance documentation for the chat, evaluated against expert answers
- Automated tests

## Data credit
Helwig, N., Pignanelli, E., & Schütze, A. (2015). *Condition Monitoring of Hydraulic Systems* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21 (CC BY 4.0)

## Author
Malli Vardhan Nasina, mechanical design engineer (hydraulic pumps, test rigs), M.Sc. Industry 4.0 & Management · [LinkedIn](https://www.linkedin.com/in/mallivardhan-prabha-a7647a25b)

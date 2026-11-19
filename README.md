# Stress-Predictor: Student Stress Assistant (ML + Lyzr Agent)

## Steps to run locally

1. Create and activate virtualenv:
   - `python -m venv venv`
   - `source venv/bin/activate` (mac/linux) or `venv\Scripts\activate` (Windows)

2. Install dependencies:
   - `pip install -r requirements.txt`

3. Train the synthetic model (generates data + saves model & scaler):
   - `python train.py`

4. (Optional) Set Lyzr environment variables if you want real Lyzr-agent integration:
   - `export LYZR_API_KEY="YOUR_KEY"`
   - `export LYZR_AGENT_URL="https://api.lyzr.ai/agent/YOUR_AGENT_ID"`
   - On Windows use `set` or PowerShell `$env:...`

   If you do NOT set those, the app uses a local fallback recommender for testing.

5. Run the Flask app:
   - `python app.py`
   - Open `http://127.0.0.1:5000` in your browser

## Files
- `train.py` — generates synthetic data and trains model
- `models/` — contains saved `stress_model.pkl` and `scaler.pkl`
- `app.py` — Flask app integrating ML + Lyzr
- `templates/chat.html` — frontend UI

# # app.py
# from flask import Flask, request, jsonify, render_template, session, redirect, url_for
# import pickle
# import requests
# import numpy as np
# import os

# app = Flask(__name__)
# app.secret_key = "change_this_secret_for_prod"

# # Load model & scaler
# MODEL_PATH = "models/stress_model.pkl"
# SCALER_PATH = "models/scaler.pkl"

# if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
#     raise FileNotFoundError("Model or scaler not found. Run train.py first to generate them.")

# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

# with open(SCALER_PATH, "rb") as f:
#     scaler = pickle.load(f)

# # Lyzr config - set environment variables or replace here (not recommended)
# LYZR_API_KEY = os.environ.get("sk-default-uKR4HaGfRuiARiBlL9O829KaOvRs8G3f", "")  # Set this in your environment for production
# LYZR_AGENT_URL = os.environ.get("https://agent-prod.studio.lyzr.ai/v3/inference/chat/", "")  # e.g. https://api.lyzr.ai/agent/YOUR_AGENT_ID

# # Simple rule-based fallback recommender (used if Lyzr credentials not provided)
# def fallback_recommender(stress_label, academic, sleep, app_usage, attendance, assignments_on_time):
#     recs = []
#     if stress_label == "High":
#         recs.append("You're showing high stress — prioritize sleep and short study sessions (25–50 min) with breaks.")
#         recs.append("Try to limit non-productive app usage to less than 2 hours/day and schedule focused work blocks.")
#     elif stress_label == "Medium":
#         recs.append("Medium stress: maintain routine sleep schedule and short exercise (20–30 min).")
#         recs.append("Break study into smaller tasks and reward completion to reduce pressure.")
#     else:
#         recs.append("Low stress: keep healthy habits. Continue consistent sleep and balanced app usage.")
#     # Small tailored tips
#     if sleep < 6:
#         recs.append("Consider adding 30-60 minutes more sleep — even small increases help recovery.")
#     if academic < 60:
#         recs.append("If academics are low, try weekly review sessions and prioritize difficult topics first.")
#     if app_usage > 6:
#         recs.append("High app usage detected. Try app-limiting tools (screen time) for productive windows.")
#     return " ".join(recs)

# # Home page
# @app.route("/")
# def index():
#     # clear session if new visit, but keep if continuing
#     if "history" not in session:
#         session["history"] = []
#     return render_template("chat.html")

# # Reset chat
# @app.route("/reset", methods=["POST"])
# def reset():
#     session.pop("history", None)
#     return jsonify({"status": "ok"})

# # Single-shot prediction + recommendation (form-style)
# @app.route("/recommend", methods=["POST"])
# def recommend():
#     try:
#         data = request.json
#         academic = float(data.get("academic", 0))
#         sleep = float(data.get("sleep", 0))
#         app_usage = float(data.get("appUsage", 0))
#         attendance = float(data.get("attendance", 85))
#         assignments_on_time = int(data.get("assignments_on_time", 1))

#         features = np.array([[academic, sleep, app_usage, attendance, assignments_on_time]])
#         scaled = scaler.transform(features)
#         pred = model.predict(scaled)[0]
#         stress_label = "Low" if pred == 0 else "Medium" if pred == 1 else "High"

#         # Build prompt for Lyzr (history optional)
#         prompt = f"Stress Level: {stress_label}, Academic: {academic}, Sleep: {sleep}, App usage: {app_usage}, Attendance: {attendance}, AssignmentsOnTime: {assignments_on_time}. Please suggest personalized, actionable recommendations for a student."

#         recommendation = ""
#         if LYZR_API_KEY and LYZR_AGENT_URL:
#             try:
#                 headers = {
#                     "Authorization": f"Bearer {LYZR_API_KEY}",
#                     "Content-Type": "application/json"
#                 }
#                 resp = requests.post(LYZR_AGENT_URL, json={"input": prompt}, headers=headers, timeout=10)
#                 resp.raise_for_status()
#                 jr = resp.json()
#                 # Attempt to locate response text - adapt if Lyzr uses different schema
#                 recommendation = jr.get("output") or jr.get("response") or jr.get("result") or str(jr)
#             except Exception as e:
#                 recommendation = f"(Lyzr call failed: {str(e)}). Using fallback recommendations. " + fallback_recommender(stress_label, academic, sleep, app_usage, attendance, assignments_on_time)
#         else:
#             recommendation = fallback_recommender(stress_label, academic, sleep, app_usage, attendance, assignments_on_time)

#         return jsonify({"stressLevel": stress_label, "recommendation": recommendation})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Chat endpoint — supports typing free text or entering numbers like "75,6,4"
# @app.route("/chat", methods=["POST"])
# def chat():
#     try:
#         payload = request.json
#         message = payload.get("message", "").strip()

#         if "history" not in session:
#             session["history"] = []

#         # If user sends three comma-separated numbers, treat them as academic,sleep,app_usage and run prediction
#         is_numeric_triplet = False
#         try:
#             parts = [float(x.strip()) for x in message.split(",")]
#             if len(parts) == 3:
#                 is_numeric_triplet = True
#                 academic, sleep, app_usage = parts
#                 # default other fields for quick prediction
#                 attendance = 85.0
#                 assignments_on_time = 1
#                 features = np.array([[academic, sleep, app_usage, attendance, assignments_on_time]])
#                 scaled = scaler.transform(features)
#                 pred = model.predict(scaled)[0]
#                 stress_label = "Low" if pred == 0 else "Medium" if pred == 1 else "High"
#                 system_msg = f"Predicted Stress Level: {stress_label} (Academic:{academic}, Sleep:{sleep}, App:{app_usage})"
#                 session["history"].append({"role": "system", "content": system_msg})
#                 # Build prompt to Lyzr
#                 prompt = "Conversation so far:\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in session["history"]])
#             else:
#                 session["history"].append({"role": "user", "content": message})
#                 prompt = "Conversation so far:\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in session["history"]])
#         except Exception:
#             session["history"].append({"role": "user", "content": message})
#             prompt = "Conversation so far:\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in session["history"]])

#         # Call Lyzr if key present
#         if LYZR_API_KEY and LYZR_AGENT_URL:
#             try:
#                 headers = {"Authorization": f"Bearer {LYZR_API_KEY}", "Content-Type": "application/json"}
#                 resp = requests.post(LYZR_AGENT_URL, json={"input": prompt}, headers=headers, timeout=10)
#                 resp.raise_for_status()
#                 jr = resp.json()
#                 bot_reply = jr.get("output") or jr.get("response") or jr.get("result") or str(jr)
#             except Exception as e:
#                 bot_reply = f"(Lyzr call failed: {e}). " + fallback_recommender("High",0,0,0,0,0)
#         else:
#             # Simple fallback: echo plus short tips based on last system message if exists
#             last_system = next((h for h in reversed(session["history"]) if h["role"]=="system"), None)
#             if last_system:
#                 # parse label from system message
#                 lab = "High" if "High" in last_system["content"] else "Medium" if "Medium" in last_system["content"] else "Low"
#                 bot_reply = fallback_recommender(lab,0,0,0,0,0)
#             else:
#                 bot_reply = "Hi — tell me your academic %, sleep hours, and app usage as `academic,sleep,appUsage` (e.g. `75,6,4`) or just ask a question."

#         session["history"].append({"role": "bot", "content": bot_reply})
#         return jsonify({"reply": bot_reply})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(port=5000, debug=True)











from flask import Flask, request, jsonify, render_template
import pickle
import requests
import numpy as np
import uuid

app = Flask(__name__)

# ---------- Load your ML Model ----------
model = pickle.load(open("models/stress_model.pkl", "rb"))   # Replace with your model path
scaler = pickle.load(open("models/scaler.pkl", "rb"))        # If you used scaling

# ---------- Lyzr Agent Config ----------
LYZR_API_KEY = "sk-default-uKR4HaGfRuiARiBlL9O829KaOvRs8G3f"
LYZR_AGENT_ID = "68dfbb5ad00b8734de4cfc4f"
LYZR_API_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"

# ---------- Homepage ----------
@app.route("/")
def index():
    return render_template("chat.html")  # Your frontend HTML

# ---------- Prediction + Lyzr Agent ----------
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        academic = float(data.get("academic", 0))
        sleep = float(data.get("sleep", 0))
        app_usage = float(data.get("app_usage", 0))
        attendance = float(data.get("attendance", 0))
        assignments_on_time = int(data.get("assignments_on_time", 1))

        # ---------- Step 1: Predict Stress ----------
        features = np.array([[academic, sleep, app_usage, attendance, assignments_on_time]])
        features_scaled = scaler.transform(features) if scaler else features
        stress_pred = model.predict(features_scaled)[0]

        # Convert numeric label -> string
        stress_label = "Low" if stress_pred == 0 else "Medium" if stress_pred == 1 else "High"

        # ---------- Step 2: Prepare message for Lyzr ----------
        message_text = (
            f"Predicted Stress Level: {stress_label}\n"
            f"Academic: {academic}, Sleep: {sleep}, App usage: {app_usage}, "
            f"Attendance: {attendance}, Assignments on time: {assignments_on_time}\n"
            "Please provide personalized recommendations for this student."
        )

        session_id = f"{LYZR_AGENT_ID}-{str(uuid.uuid4())}"  # Unique session for each user

        payload = {
            "user_id": "nikhilsiddartha21@gmail.com",
            "agent_id": LYZR_AGENT_ID,
            "session_id": session_id,
            "message": message_text
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": LYZR_API_KEY
        }

        response = requests.post(LYZR_API_URL, json=payload, headers=headers)
        agent_output = response.json()

        # Lyzr returns message in `response` field
        recommendation_text = agent_output.get("response", "No recommendation received.")

        return jsonify({
            "stressLevel": stress_label,
            "recommendation": recommendation_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)

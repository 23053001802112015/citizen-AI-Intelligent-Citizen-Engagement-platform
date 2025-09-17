Got it ✅
Your document describes the Citizen AI – Intelligent Citizen Engagement Platform with Flask, IBM Granite, Watson, and NLP features. I’ll now help you convert the project plan into Python code (main app.py + helper functions + project structure).

Here’s a starting point for the Flask backend (app.py) that aligns with your document:

# app.py
from flask import Flask, render_template, request, redirect, url_for, session
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
app.secret_key = "supersecretkey"  # For session handling

# -----------------------------
# IBM Granite Model Setup
# -----------------------------
model_path = "ibm-granite/granite-3.3-8b-instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# -----------------------------
# Helper Functions
# -----------------------------
def granite_generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def analyze_sentiment(text):
    # Simple rule-based sentiment analysis (replace with ML later)
    text = text.lower()
    if any(word in text for word in ["good", "great", "excellent", "happy"]):
        return "Positive"
    elif any(word in text for word in ["bad", "poor", "terrible", "sad"]):
        return "Negative"
    else:
        return "Neutral"

# In-memory storage
chat_history = []
feedback_list = []
concerns = []

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/chat")
def chat():
    return render_template("chat.html", question_response=None)

@app.route("/dashboard")
def dashboard():
    sentiment_counts = {
        "Positive": sum(1 for f in feedback_list if f["sentiment"] == "Positive"),
        "Neutral": sum(1 for f in feedback_list if f["sentiment"] == "Neutral"),
        "Negative": sum(1 for f in feedback_list if f["sentiment"] == "Negative"),
    }
    return render_template("dashboard.html", sentiments=sentiment_counts, concerns=concerns)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin":
            session["user"] = username
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

# -----------------------------
# Feature Routes
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.form.get("question")
    response = granite_generate_response(question)
    chat_history.append({"q": question, "a": response})
    return render_template("chat.html", question_response=response)

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    feedback = request.form.get("feedback")
    sentiment = analyze_sentiment(feedback)
    feedback_list.append({"text": feedback, "sentiment": sentiment})
    return render_template("chat.html", sentiment=sentiment)

@app.route("/concern", methods=["POST"])
def submit_concern():
    concern = request.form.get("concern")
    concerns.append(concern)
    return render_template("chat.html", concern_submitted=True)

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)


---

📂 Project Structure (as per your document):

CitizenAI/
│── app.py
│── requirements.txt
│── templates/
│   ├── index.html
│   ├── about.html
│   ├── services.html
│   ├── chat.html
│   ├── dashboard.html
│   ├── login.html
│── static/
    ├── css/
    │   └── styles.css
    ├── Images/
    └── Favicon/

🔹 This code covers:

Flask routes (/, /chat, /ask, /feedback, /concern, /dashboard, /login, /logout)

IBM Granite model integration (via Hugging Face)

Sentiment analysis (rule-based, can upgrade later)

In-memory storage for chat, feedback, concerns

Dynamic dashboard data



---

Do you want me to also generate the HTML templates (index.html, chat.html, dashboard.html, etc.) so you can run this Flask app end-to-end?


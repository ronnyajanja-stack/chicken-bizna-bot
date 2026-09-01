import os
import uuid
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
from rag_engine import retrieve_context

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

user_sessions = {}

SYSTEM_INSTRUCTION = """You are a warm, polite, and helpful customer support assistant for Chicken Bizna.

OPERATING GUIDELINES:
1. Grounded & Factual: Use the retrieved context as your source of truth. Do not invent prices, contact numbers, or specific inventory details not supported by the context.
2. Conversational & Flexible: If a customer enters a short or general keyword (such as "broiler", "kienyeji", or "feeds"), give a brief, friendly summary of what you offer for that category based on available information, then guide them on what they can ask next.
3. Warm & Helpful Fallbacks: If an exact detail is not mentioned in your knowledge base (such as specific live weight in kg or special delivery discounts), acknowledge it pleasantly, share any relevant general information available, and encourage them to reach out directly to the team or ask about pricing and availability.
4. Tone: Natural, welcoming, concise, and structured for fast reading."""

def get_session_history(session_id: str):
    if session_id not in user_sessions:
        user_sessions[session_id] = [
            {"role": "system", "content": SYSTEM_INSTRUCTION}
        ]
    return user_sessions[session_id]

@app.route("/")
def home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", "default_guest")

    try:
        # Retrieve context from ChromaDB
        context = retrieve_context(user_message)

        prompt_with_rag = f"""RETRIEVED KNOWLEDGE BASE SNIPPETS:
{context}

CUSTOMER QUESTION:
{user_message}"""

        history = get_session_history(session_id)
        history.append({"role": "user", "content": prompt_with_rag})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.4,  # Slight increase allows for warmer conversational wording while keeping facts grounded
        )
        bot_reply = response.choices[0].message.content

        history.append({"role": "assistant", "content": bot_reply})

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print("\n--- DETAILED ERROR TRACEBACK ---")
        traceback.print_exc()
        print("--------------------------------\n")
        return jsonify({"reply": "I'm having a brief moment of delay. Please feel free to ask again!"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=False, use_reloader=False, threaded=False)
import os
import uuid
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session, Response
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse
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

def generate_bot_reply(user_message: str, session_id: str) -> str:
    """Shared core RAG and OpenAI completion logic."""
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
        temperature=0.4,
    )
    bot_reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": bot_reply})
    return bot_reply

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
        bot_reply = generate_bot_reply(user_message, session_id)
        return jsonify({"reply": bot_reply})
    except Exception as e:
        print("\n--- DETAILED ERROR TRACEBACK (WEB) ---")
        traceback.print_exc()
        print("--------------------------------------\n")
        return jsonify({"reply": "I'm having a brief moment of delay. Please feel free to ask again!"}), 500

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Endpoint for Twilio WhatsApp incoming messages."""
    # Twilio sends form-encoded parameters: 'Body' (the text) and 'From' (e.g., 'whatsapp:+254754771381')
    user_message = request.values.get("Body", "").strip()
    sender_number = request.values.get("From", "unknown_user")

    resp = MessagingResponse()

    if not user_message:
        resp.message("Hello! Send us a question about our chicks, feeds, or pricing.")
        return Response(str(resp), mimetype="application/xml")

    try:
        # Use the sender's WhatsApp phone number as their unique session ID
        bot_reply = generate_bot_reply(user_message, sender_number)
        resp.message(bot_reply)
    except Exception as e:
        print("\n--- DETAILED ERROR TRACEBACK (WHATSAPP) ---")
        traceback.print_exc()
        print("------------------------------------------\n")
        resp.message("I'm having a brief moment of delay. Please feel free to ask again!")

    return Response(str(resp), mimetype="application/xml")

if __name__ == "__main__":
    app.run(port=5000, debug=False, use_reloader=False, threaded=False)
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
# Keep a stable secret key in production if set in env, otherwise fallback
app.secret_key = os.getenv("SECRET_KEY", "chicken_bizna_secret_default_key")

# Initialize OpenAI Client with a strict 15-second network timeout
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY is not set or empty in environment variables!")

client = OpenAI(
    api_key=api_key,
    timeout=15.0  # Prevents requests from hanging indefinitely
)

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
    try:
        context = retrieve_context(user_message)
    except Exception as e:
        print(f"Error retrieving RAG context: {e}")
        context = "No additional context found."

    prompt_with_rag = f"""RETRIEVED KNOWLEDGE BASE SNIPPETS:
{context}

CUSTOMER QUESTION:
{user_message}"""

    history = get_session_history(session_id)
    
    # Send system instruction + history + current query with context
    messages_to_send = list(history)
    messages_to_send.append({"role": "user", "content": prompt_with_rag})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_to_send,
        temperature=0.4,
    )
    
    bot_reply = response.choices[0].message.content
    
    # Store clean conversation history without the massive RAG blocks
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_reply})
    
    # Keep only last 10 exchanges to prevent token explosion
    if len(history) > 21:
        user_sessions[session_id] = [history[0]] + history[-20:]
        
    return bot_reply

@app.route("/")
def home():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a message to begin!"}), 400

    session_id = session.get("session_id", "default_guest")

    try:
        bot_reply = generate_bot_reply(user_message, session_id)
        return jsonify({"reply": bot_reply})
    except Exception as e:
        print("\n--- DETAILED ERROR TRACEBACK (WEB) ---")
        traceback.print_exc()
        print("--------------------------------------\n")
        return jsonify({
            "reply": f"Service temporarily delayed: {str(e)[:120]}"
        }), 200  # Returns 200 so the UI displays the exact error instead of crashing silently

@app.route("/webhook", methods=["POST"])
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Endpoint for Twilio WhatsApp incoming messages."""
    user_message = request.values.get("Body", "").strip()
    sender_number = request.values.get("From", "unknown_user")

    resp = MessagingResponse()

    if not user_message:
        resp.message("Hello! Send us a question about our chicks, feeds, or pricing.")
        return Response(str(resp), mimetype="application/xml")

    try:
        bot_reply = generate_bot_reply(user_message, sender_number)
        resp.message(bot_reply)
    except Exception as e:
        print("\n--- DETAILED ERROR TRACEBACK (WHATSAPP) ---")
        traceback.print_exc()
        print("------------------------------------------\n")
        resp.message("I'm having a brief moment of delay. Please feel free to ask again!")

    return Response(str(resp), mimetype="application/xml")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
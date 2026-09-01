import os
import threading
import requests
from dotenv import load_dotenv
from flask import Flask, request
from google import genai
from rag_engine import retrieve_context

load_dotenv()

app = Flask(__name__)

# =========================
# CONFIGURATION
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN")
META_GRAPH_API_VERSION = os.getenv("META_GRAPH_API_VERSION", "v21.0")

# =========================
# GEMINI CLIENT
# =========================
client = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# CHAT MEMORY
# =========================
user_sessions = {}

def get_or_create_chat(user_id: str):
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    return user_sessions[user_id]


# =========================
# SEND WHATSAPP MESSAGE
# =========================
def send_whatsapp_message(to: str, message: str):
    url = f"https://graph.facebook.com/{META_GRAPH_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": message
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        print(f"Meta Send Status: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except Exception as err:
        print(f"Failed to send WhatsApp message: {err}")
        return None


# =========================
# GEMINI RESPONSE GENERATION
# =========================
def process_message_async(sender: str, user_message: str):
    """Runs in the background to prevent Meta 3-second webhook timeouts."""
    try:
        chat_history = get_or_create_chat(sender)

        # 1. Retrieve knowledge base context
        context = retrieve_context(user_message)

        # 2. Format system and conversation history
        history_formatted = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-6:]])

        prompt = f"""You are a helpful customer support assistant for Chicken Bizna.

STRICT OPERATING RULES:
1. Answer ONLY using the facts from the retrieved context below.
2. If the answer is NOT explicitly found in the context, reply: "I'm sorry, I don't have that specific information in my records. Please call us directly for assistance."
3. Keep responses clear, professional, and concise for WhatsApp.

RETRIEVED CONTEXT:
{context}

RECENT CONVERSATION HISTORY:
{history_formatted}

CURRENT CUSTOMER QUESTION:
{user_message}"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text

        # 3. Update conversation history
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": answer})

        # 4. Deliver response back to user
        send_whatsapp_message(sender, answer)

    except Exception as e:
        print(f"Error processing message from {sender}: {e}")
        send_whatsapp_message(sender, "I'm having trouble retrieving details right now. Please try again in a moment.")


# =========================
# META WEBHOOK VERIFICATION
# =========================
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == META_VERIFY_TOKEN:
        print("Webhook verified successfully.")
        return challenge, 200

    print("Webhook verification failed.")
    return "Forbidden", 403


# =========================
# RECEIVE WHATSAPP MESSAGES
# =========================
@app.route("/webhook", methods=["POST"])
def receive_webhook():
    data = request.get_json() or {}

    try:
        entry = data.get("entry", [])
        for entry_item in entry:
            changes = entry_item.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                messages = value.get("messages", [])

                for message in messages:
                    if message.get("type") != "text":
                        continue

                    sender = message.get("from")
                    user_message = message.get("text", {}).get("body", "").strip()

                    if sender and user_message:
                        print(f"Received message from {sender}: {user_message}")
                        
                        # Process AI retrieval in a separate thread to immediately return 200 OK to Meta
                        threading.Thread(
                            target=process_message_async,
                            args=(sender, user_message),
                            daemon=True
                        ).start()

    except Exception as e:
        print(f"Webhook parsing error: {e}")

    # Immediately acknowledge Meta
    return "EVENT_RECEIVED", 200


# =========================
# HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return "WhatsApp AI Bot is running!"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=False,
        use_reloader=False
    )
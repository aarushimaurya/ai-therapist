import uuid
from pathlib import Path
from typing import Optional

import edge_tts
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# The frontend is a sibling directory of this file: backend/main.py -> ../frontend
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Allows the browser-based frontend (a different origin) to call this API.
# allow_origins=["*"] is fine for local development; a deployed app should
# list its real frontend URL(s) instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation store: conversation_id -> list of message dicts.
conversations: dict[str, list[dict[str, str]]] = {}

# Groq() reads GROQ_API_KEY from the environment automatically.
client = Groq()

MODEL = "llama-3.3-70b-versatile"

# A warm, natural-sounding free neural voice (Microsoft Edge's online TTS
# service via the edge-tts library). "-8%" rate gives a calmer, less rushed
# delivery, matching the mellow tone the therapist is going for.
TTS_VOICE = "en-US-AriaNeural"
TTS_RATE = "-8%"

THERAPIST_SYSTEM_PROMPT = """You are a warm, empathetic AI therapist. Your job is to listen, \
not to lecture. The user should be doing most of the talking, not you.

Guidelines:
- Keep every response SHORT: 1-3 sentences, never more. No bullet lists, no multi-part \
explanations.
- Respond with a brief reflection of what they said, or a single open-ended question \
to draw them out — not both at length, and never more than one question per reply.
- Don't jump to advice or solutions unless the user explicitly asks for it.
- Validate emotions before problem-solving.
- Never diagnose conditions or prescribe medication.
- Remind the user, when relevant, that you are an AI and not a licensed therapist, \
and encourage them to reach out to a qualified professional for serious or ongoing concerns.
"""


class ChatRequest(BaseModel):
    message: str
    # Optional: omitted on the first message, then echoed back by the client
    # on every following message to continue the same conversation.
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str


class SpeakRequest(BaseModel):
    text: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # Find the existing conversation, or start a new one.
    if request.conversation_id is None:
        conversation_id = str(uuid.uuid4())
        history = [{"role": "system", "content": THERAPIST_SYSTEM_PROMPT}]
    else:
        conversation_id = request.conversation_id
        history = conversations.get(conversation_id)
        if history is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    # Add the new user message to the running history.
    history.append({"role": "user", "content": request.message})

    # Send the WHOLE history so the model has the full context.
    # max_tokens is kept low as a hard backstop so replies stay short even
    # if the model ignores the brevity instructions in the system prompt.
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=150,
        messages=history,
    )
    reply = response.choices[0].message.content

    # Save the assistant's reply so it's remembered on the next turn.
    history.append({"role": "assistant", "content": reply})
    conversations[conversation_id] = history

    return ChatResponse(reply=reply, conversation_id=conversation_id)


@app.post("/speak")
async def speak(request: SpeakRequest) -> Response:
    communicate = edge_tts.Communicate(request.text, voice=TTS_VOICE, rate=TTS_RATE)
    audio_chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.extend(chunk["data"])
    return Response(content=bytes(audio_chunks), media_type="audio/mpeg")


# Serves frontend/index.html at "/" (and any other file in that folder).
# Mounted LAST so it only catches requests that didn't match /chat or
# /speak above. Serving the frontend from this same origin (instead of
# opening index.html as a file://... URL) is what lets the browser
# remember the microphone permission permanently.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

# 🔑 Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBaBgpGBnBi9ZxbNjjdsujdODQdMkwaebY"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chat(BaseModel):
    message: str

# ✅ Gemini 2.5 Flash (FAST + LOW LATENCY)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template("""
You are Solar Chat, an AI assistant for space weather.

ONLY answer questions related to:
- solar storms
- geomagnetic storms
- satellite impact
- space weather risks
- forecasts

If question is NOT related to space weather, say:
"I only answer space weather questions."

Use very simple language.
Keep answers short and clear.

Question: {question}
""")

chain = prompt | llm

@app.post("/chat")
def solar_chat(chat: Chat):
    try:
        response = chain.invoke({"question": chat.message})
        return {"reply": response.content}
    except Exception as e:
        return {"reply": f"❌ Error: {str(e)}"}

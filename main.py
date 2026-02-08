from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# 🔹 Load environment variables
load_dotenv()

# 🔹 Debug check (keep for now)
print("GOOGLE_API_KEY loaded:", bool(os.getenv("GOOGLE_API_KEY")))

app = FastAPI()

# 🔹 CORS (frontend friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Request schema
class Chat(BaseModel):
    message: str

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SolarChat backend is running 🚀"
    }


# 🔹 Gemini 2.5 Flash (explicit API key pass = no crash)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# 🔹 Prompt
prompt = ChatPromptTemplate.from_template("""
You are Solar Chat, an AI assistant for space weather.

ONLY answer questions related to:
- solar storms
- geomagnetic storms
- satellite impact
- space weather risks
- forecasts

If the question is NOT related to space weather, say:
"I only answer space weather questions."

Use very simple language.
Keep answers short and clear.

Question: {question}
""")

# 🔹 Chain
chain = prompt | llm

# 🔹 API endpoint
@app.post("/chat")
def solar_chat(chat: Chat):
    try:
        response = chain.invoke({"question": chat.message})
        return {"reply": response.content}
    except Exception as e:
        return {"reply": f"❌ Error: {str(e)}"}

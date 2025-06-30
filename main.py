from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)
app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Request body model
class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(req: TextRequest):
    prompt = f"Summarize the following text:\n{req.text}"
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions="You are an expert in summarising. your task is to summarise everything i give you, summarise it very shortly",
            input=prompt,
        )
        summary = response.output_text
        return {"summary": summary}
    except Exception as e:
        return {"summary": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

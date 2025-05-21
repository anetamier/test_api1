from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")

app = FastAPI(title="Сервис анализа тональности текста")

templates = Jinja2Templates(directory=templates_dir)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_api_key():
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="API ключ не найден")
    return api_key

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    sentiment: str

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_model=TextResponse)
async def analyze_text(request: TextRequest, api_key: str = Depends(get_api_key)):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    system_content = 'Определи тональность текста. Ответь одним словом: "положительно", "отрицательно" или "нейтрально".'
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": request.text}
    ]
    
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=messages,
            temperature=0
        )
        return TextResponse(sentiment=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обращении к модели: {str(e)}")


# if __name__ == "__main__":
#     uvicorn.run("app:app", port=80, reload=True)  
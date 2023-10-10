from bot import PDFIndexer, Bot

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

templates = Jinja2Templates(directory="./chatapp")

app = FastAPI()

bot, c_index = None, None


class ChatModel(BaseModel):
    query: str


try:
    app.mount(
        "/static",
        StaticFiles(directory="../frontend", html=True),
        name="static",
    )

    templates = Jinja2Templates(directory="../frontend")
except RuntimeError:
    pass


@app.get("/admin")
def root(request: Request):
    return templates.TemplateResponse(
        "admin.html",
        context={"request": request},
    )


@app.post("/index")
def root(request: Request, file: UploadFile):
    vdb = PDFIndexer(bfile=file).index()

    global bot, c_index
    bot = Bot(vdb=vdb)
    c_index = file.filename

    return HTMLResponse("<h2>Indexing successful 🚀</h2><a href='/'>Start chatting!</a>")


@app.get("/")
def chat(request: Request):
    return templates.TemplateResponse(
        "index.html", context={"request": request, "c_index": c_index}
    )


@app.post("/chat")
def chat(cm: ChatModel):
    query = cm.query
    if bot == None:
        return "No index found"

    return bot.reply(question=query)

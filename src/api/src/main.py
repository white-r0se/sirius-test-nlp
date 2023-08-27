from fastapi import FastAPI
from pydantic import BaseModel
from interface_model import InterfaceModel
from interface_model import __version__
__version__ = "0.0.1"

app = FastAPI()
interface = InterfaceModel()

class TextInput(BaseModel):
    text: str

class TextOutput(BaseModel):
    reply: str
    history: str

@app.get("/")
def read_root():
    return {"status": "OK", "version": __version__}

@app.post("/predict", response_model=TextOutput)
def predict(input: TextInput):
    reply, history = interface.predict(input.text)
    return {"reply": reply, "history": history}

@app.get("/clear_history")
def clear_history():
    interface.clear_history()
    return {"status": "OK"}
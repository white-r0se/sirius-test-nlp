from fastapi import FastAPI
from pydantic import BaseModel
from interface_model import InterfaceModel
from interface_model import __version__

app = FastAPI()
interface = InterfaceModel()
print("Model is loaded")


class TextInput(BaseModel):
    text: str


class TextOutput(BaseModel):
    reply: str
    history: str


@app.get("/")
def read_root():
    """Root endpoint"""
    return {"status": "OK", "version": __version__}


@app.post("/predict", response_model=TextOutput)
def predict(input: TextInput):
    """Predict endpoint"""
    reply, history = interface.predict(input.text)
    return {"reply": reply, "history": history}


@app.get("/clear_history")
def clear_history():
    """Clear history endpoint"""
    interface.clear_history()
    return {"status": "OK"}

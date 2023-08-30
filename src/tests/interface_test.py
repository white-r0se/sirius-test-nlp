import pytest
import transformers
from api.src.interface_model import InterfaceModel, PeftModel, PeftConfig


@pytest.fixture
def interface_model():
    return InterfaceModel(setup_model=False)


def test_setup_model(interface_model):
    interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
    interface_model.setup_model()
    assert isinstance(interface_model.config, PeftConfig)
    assert isinstance(
        interface_model.tokenizer,
        transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast,
    )
    assert isinstance(interface_model.model, PeftModel)
    assert isinstance(interface_model.chat_history, str)


def test_predict(interface_model):
    interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
    interface_model.setup_model()
    reply, history = interface_model.predict("Sample text")
    assert isinstance(reply, str)
    assert isinstance(history, str)


def test_predict_with_too_long_history(interface_model):
    interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
    interface_model.chat_history = " ".join(["Sample history"] * 1024)
    reply, history = interface_model.predict("Sample text")
    assert len(history.split()) < 2048


def test_history_saves(interface_model):
    interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
    interface_model.predict("Sample text")
    assert "@@ПЕРВЫЙ@@ Sample text @@ВТОРОЙ@@ " in interface_model.chat_history


def test_clear_history(interface_model):
    interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
    interface_model.predict("Sample text")
    interface_model.clear_history()
    assert interface_model.chat_history == "@@ПЕРВЫЙ@@ "

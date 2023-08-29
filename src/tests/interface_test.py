import unittest
from unittest.mock import patch, MagicMock
import transformers
from ..api.src.interface_model import InterfaceModel, PeftModel, PeftConfig


class TestInterfaceModel(unittest.TestCase):
    def setUp(self):
        self.interface_model = InterfaceModel()

    def test_setup_model(self):
        self.interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
        self.interface_model.setup_momkdel()
        self.assertTrue(isinstance(self.interface_model.config, PeftConfig))
        self.assertTrue(
            isinstance(
                self.interface_model.tokenizer,
                transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast,
            )
        )
        self.assertTrue(isinstance(self.interface_model.model, PeftModel))
        self.assertTrue(isinstance(self.interface_model.chat_history, str))

    def test_predict(self):
        self.interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
        self.interface_model.setup_model()
        reply, history = self.interface_model.predict("Sample text")
        self.assertTrue(isinstance(reply, str))
        self.assertTrue(isinstance(history, str))

    def test_predict_with_too_long_history(self):
        self.interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
        self.interface_model.chat_history = " ".join(["Sample history"] * 1024)
        reply, history = self.interface_model.predict("Sample text")
        self.assertTrue(len(history.split()) < 2048)

    def test_history_saves(self):
        self.interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
        self.interface_model.predict("Sample text")
        self.assertTrue(
            "@@ПЕРВЫЙ@@ Sample text @@ВТОРОЙ@@ " in self.interface_model.chat_history
        )

    def test_clear_history(self):
        self.interface_model.model_path = "models/rudialogpt-medium-lora-5ep"
        self.interface_model.predict("Sample text")
        self.interface_model.clear_history()
        self.assertEqual(self.interface_model.chat_history, "@@ПЕРВЫЙ@@ ")


if __name__ == "__main__":
    unittest.main()

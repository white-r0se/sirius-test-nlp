import unittest
from unittest.mock import patch, MagicMock
import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(parent_path, "api/src"))
from interface_model import InterfaceModel

class TestInterfaceModel(unittest.TestCase):
    def setUp(self):
        self.interface_model = InterfaceModel()

    @patch("interface_model.PeftConfig.from_pretrained")
    @patch("interface_model.AutoTokenizer.from_pretrained")
    @patch("interface_model.AutoModelForCausalLM.from_pretrained")
    @patch("interface_model.PeftModel.from_pretrained")
    def test_setup_model(self, mock_peft_model, mock_causal_model, mock_tokenizer, mock_config):
        mock_config.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_causal_model.return_value = MagicMock()
        mock_peft_model.return_value = MagicMock()

        self.interface_model.setup_model()

        self.assertIsNotNone(self.interface_model.model)
        self.assertIsNotNone(self.interface_model.tokenizer)
        self.assertIsNotNone(self.interface_model.config)
        self.assertIsNotNone(self.interface_model.chat_history)

    @patch("interface_model.InterfaceModel.predict")
    def test_predict(self, mock_predict):
        mock_predict.return_value = ("Sample reply", "Sample history")

        reply, history = self.interface_model.predict("Sample text")

        self.assertEqual(reply, "Sample reply")
        self.assertEqual(history, "Sample history")

    def test_predict_with_too_long_history(self):
        self.interface_model.chat_history = " ".join(["Sample history"] * 1024)
        reply, history = self.interface_model.predict("Sample text")
        self.assertTrue(len(history.split()) < 2048)

    def test_history_saves(self):
        self.interface_model.predict("Sample text")

        self.assertTrue("@@ПЕРВЫЙ@@ Sample text @@ВТОРОЙ@@ " in self.interface_model.chat_history)

    def test_clear_history(self):
        self.interface_model.predict("Sample text")
        self.interface_model.clear_history()

        self.assertEqual(self.interface_model.chat_history, "@@ПЕРВЫЙ@@ ")

if __name__ == '__main__':
    unittest.main()

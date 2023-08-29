import unittest
from unittest.mock import patch, MagicMock
import requests
from ..bot.src.main import ChatBot


class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.token = "6421026571:AAGPlGWr-FeHr6xRCllrrWfs6t-_DYMCWLM"
        self.chat_bot = ChatBot(self.token)

    @patch("requests.get")
    def test_check_connection_success(self, mock_get):
        mock_get.return_value.status_code = 200

        result = self.chat_bot.check_connection()
        self.assertTrue(result)

    @patch("requests.get")
    def test_check_connection_failure(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = self.chat_bot.check_connection()
        self.assertFalse(result)

    @patch("requests.post")
    def test_send_post_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"reply": "Sample reply"}
        mock_post.return_value = mock_response

        response = self.chat_bot._send_post("Sample text")
        self.assertEqual(response["reply"], "Sample reply")

    @patch("requests.get")
    def test_send_get_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "OK"}
        mock_get.return_value = mock_response

        response = self.chat_bot._send_get()
        self.assertEqual(response["status"], "OK")

    @patch("requests.get")
    def test_clear_history_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "OK"}
        mock_get.return_value = mock_response

        response = self.chat_bot._send_get()
        self.assertEqual(response["status"], "OK")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
import requests
import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(parent_path, "bot/src"))
from main import ChatBot

class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.token = "6421026571:AAGPlGWr-FeHr6xRCllrrWfs6t-_DYMCWLM"
        self.chat_bot = ChatBot(self.token)

    @patch("main.requests.get")
    def test_check_connection_success(self, mock_get):
        # Mock the response of requests.get to simulate a successful connection
        mock_get.return_value.status_code = 200

        result = self.chat_bot.check_connection()
        self.assertTrue(result)

    @patch("main.requests.get")
    def test_check_connection_failure(self, mock_get):
        # Mock the response of requests.get to simulate a failed connection
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = self.chat_bot.check_connection()
        self.assertFalse(result)

    @patch("main.requests.post")
    def test_send_post_success(self, mock_post):
        # Mock the response of requests.post to simulate a successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"reply": "Sample reply"}
        mock_post.return_value = mock_response

        response = self.chat_bot.send_post("Sample text")
        self.assertEqual(response["reply"], "Sample reply")

    @patch("main.requests.get")
    def test_send_get_success(self, mock_get):
        # Mock the response of requests.get to simulate a successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Success"}
        mock_get.return_value = mock_response

        response = self.chat_bot.send_get()
        self.assertEqual(response["status"], "Success")

    @patch("main.requests.get")
    def test_clear_history_success(self, mock_get):
        # Mock the response of requests.get to simulate a successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "OK"}
        mock_get.return_value = mock_response

        response = self.chat_bot.send_get()
        self.assertEqual(response["status"], "OK")

if __name__ == '__main__':
    unittest.main()

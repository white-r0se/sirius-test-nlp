import os
import pytest
from unittest.mock import patch, MagicMock
import requests
from src.bot.run_bot import ChatBot

BOT_TOKEN = os.getenv("BOT_TOKEN")


@pytest.fixture
def chat_bot():
    return ChatBot(BOT_TOKEN)


@patch("requests.get")
def test_check_connection_success(mock_get, chat_bot):
    mock_get.return_value.status_code = 200

    assert chat_bot.check_connection()


@patch("requests.get")
def test_check_connection_failure(mock_get, chat_bot):
    mock_get.side_effect = requests.exceptions.ConnectionError()

    assert not chat_bot.check_connection()


@patch("requests.post")
def test_send_post_success(mock_post, chat_bot):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"reply": "Sample reply"}
    mock_post.return_value = mock_response

    response = chat_bot._send_post("Sample text")
    assert response["reply"] == "Sample reply"


@patch("requests.get")
def test_send_get_success(mock_get, chat_bot):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "OK"}
    mock_get.return_value = mock_response

    response = chat_bot._send_get()
    assert response["status"] == "OK"


@patch("requests.get")
def test_clear_history_success(mock_get, chat_bot):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "OK"}
    mock_get.return_value = mock_response

    response = chat_bot._send_get()
    assert response["status"] == "OK"

import telebot
import requests

class ChatBot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        self.url = "http://sirius-test-nlp-app-1"

    def check_connection(self):
        try:
            response = requests.get(self.url, timeout=10)
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.ConnectionError:
            return False

    def send_post(self, text):
        url = self.url + "/predict"
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=1000)
        if response.status_code == 200:
            return response.json()
        else:
            return {"reply": "Произошла ошибка при обращении к боту"}

    def send_get(self):
        url = self.url + "/clear_history"
        response = requests.get(url)
        if response.status_code == 200:
            return {"status": "OK"}
        else:
            return {"status": "Произошла ошибка при обращении к боту"}      

    def start(self):
        @self.bot.message_handler(commands=['help'])
        def send_help(message):
            self.bot.reply_to(message, "Запуск бота: /start\nОчистка чата: /clear_history")

        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            markup = telebot.types.ReplyKeyboardMarkup()
            itembtn1 = telebot.types.KeyboardButton('Привет!')
            itembtn2 = telebot.types.KeyboardButton('/help')
            itembtn3 = telebot.types.KeyboardButton('/clear_history')
            markup.add(itembtn1, itembtn2, itembtn3)
            self.bot.reply_to(message, "Привет! Я чатбот. О чем хочешь поговорить?", reply_markup=markup)

        @self.bot.message_handler(commands=['clear_history'])
        def clear_history(message):
            if self.check_connection():
                response = self.send_get()
                if response["status"] == "OK":
                    self.bot.reply_to(message, "История чата очищена")
                else:
                    self.bot.reply_to(message, "Произошла ошибка при обращении к боту")
            else:
                self.bot.reply_to(message, "Произошла ошибка при обращении к боту (ошибка подключения)")

        @self.bot.message_handler(func=lambda m: True)
        def reply(message):
            if self.check_connection():
                self.bot.send_chat_action(message.chat.id, 'typing')
                response = self.send_post(message.text)
                if "reply" in response:
                    reply = response["reply"]
                    self.bot.reply_to(message, reply)
                else:
                    self.bot.reply_to(message, "Произошла ошибка при обращении к боту")
            else:
                self.bot.reply_to(message, "Произошла ошибка при обращении к боту (ошибка подключения)")

    def run(self):
        self.start()
        self.bot.polling()

if __name__ == '__main__':
    with open('/run/secrets/token', 'r') as f:
        BOT_TOKEN = f.read()
    chat_bot = ChatBot(token=BOT_TOKEN)
    chat_bot.run()

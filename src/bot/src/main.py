# import os
import telebot
import requests

# BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = telebot.TeleBot("6421026571:AAGPlGWr-FeHr6xRCllrrWfs6t-_DYMCWLM")

def send_post(text):
    url = "http://sirius-test-nlp-app-1/predict"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, timeout=1000)
    if response.status_code == 200:
        return response.json()
    else:
        return {"reply": "Произошла ошибка при обращении к боту"}

def send_get():
    url = "http://sirius-test-nlp-app-1/clear_history"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"status": "Произошла ошибка при обращении к боту"}

@bot.message_handler(commands=['help'])
def send_help(message):
    bot.reply_to(message, "Запуск бота: /start\Очистка чата: /clear_history")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    markup = telebot.types.ReplyKeyboardMarkup()
    itembtn1 = telebot.types.KeyboardButton('Привет!')
    itembtn2 = telebot.types.KeyboardButton('/help')
    itembtn3 = telebot.types.KeyboardButton('/clear_history')
    markup.add(itembtn1, itembtn2, itembtn3)
    bot.reply_to(message, "Привет! Я чатбот. О чем хочешь поговорить?", reply_markup=markup)

@bot.message_handler(commands=['clear_history'])
def clear_history(message):
    send_get()    
    bot.reply_to(message, "История чата очищена")

@bot.message_handler(func=lambda m: True)
def reply(message):
    bot.send_chat_action(message.chat.id, 'typing')
    reply = send_post(message.text)["reply"]
    bot.reply_to(message, reply)

def main():
    bot.polling()

if __name__ == '__main__':
    main()
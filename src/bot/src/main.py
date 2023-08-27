# import os
import telebot
import requests

# BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = telebot.TeleBot("6421026571:AAGPlGWr-FeHr6xRCllrrWfs6t-_DYMCWLM")

def send_post_predict(text):
    # send post request to api, i have no idea how to do it
    # but I have network "mynet" between my docker containers
    # so I can send request to api container
    url = "http://sirius-test-nlp-app-1:8080/predict"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def send_get_clear_history():
    url = "http://sirius-test-nlp-app-1:8080/clear_history"
    response = requests.get(url)
    return response.json()

@bot.message_handler(commands=['help'])
def send_help(message):
    bot.reply_to(message, "Для очистки истории чата нажмите /clear_history")

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
    send_get_clear_history()
    bot.reply_to(message, "История чата очищена")

@bot.message_handler(func=lambda m: True)
def reply(message):
    bot.send_chat_action(message.chat.id, 'typing')
    reply = send_post_predict(message.text)["reply"]
    print(history)
    bot.reply_to(message, reply)

def main():
    bot.polling()

if __name__ == '__main__':
    main()
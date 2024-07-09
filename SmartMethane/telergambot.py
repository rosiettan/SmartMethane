import telebot
import time

bot = telebot.TeleBot("5660265930:AAFqNKrHj2gVvK0YN2vzAKjYQ5JxpjFjLDg", parse_mode=None)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "A personal assistant 'SmartMetan' welcomes you!")


@bot.message_handler(func=lambda m: True)
def echo_all(message):
    if "how" in message.text.lower() and "you" in message.text.lower():
        bot.send_message(message.chat.id, "I'm fine, thank you")
    elif "name" in message.text.lower():
        bot.send_message(message.chat.id, "My name is Rosie")
    elif message.text == "count":
        for i in range(5,0,-1):
            bot.send_message(message.chat.id, f"Counting....{i}")
            time.sleep(1) # никогда нельзя использовать в производстве, так как ухудшает работу
        bot.send_message(message.chat.id, "Go!")
    else:
        bot.send_message(message.chat.id, "I don't understand you")


print("Bot started")
bot.infinity_polling()
print("Bot stopped")
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes
)

from sermons import sermons

BOT_TOKEN = "8701198641:AAG_G6BPznjstUSxXFBJHl-WUx0ujevVRpc"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """
Welcome to Apostle Omo's A.I

Tell me what kind of message you need and I’ll recommend teachings for you.
"""
    )

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.lower()

    found_sermons = []

    for sermon in sermons:
        for keyword in sermon["keywords"]:
            if keyword in user_message:
                found_sermons.append(sermon)

    if found_sermons:
        response = "Here are some recommended messages:\n\n"

        for sermon in found_sermons:
            response += f"{sermon['title']}\n"
            response += f"{sermon['link']}\n\n"

    else:
        response = "Sorry, I couldn't find a matching message yet."

    await update.message.reply_text(response)

app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

print("Apostle Omo's A.I is running...")

app.run_polling(
    poll_interval=3,
    timeout=30,
    bootstrap_retries=5
)
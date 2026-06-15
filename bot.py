from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes
)

import os
import random
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sermons import sermons

BOT_TOKEN = os.getenv("BOT_TOKEN")

WELCOME_MESSAGE = """
🙏 Welcome to Apostle Omo's A.I.

Describe what you are going through, and I will recommend relevant teachings from Apostle Omorogbe Aimiuwu.

You can ask things like:

• I need favour in my business
• I feel emotionally down
• I need restoration after failure
• I need breakthrough in my career
• I want to grow spiritually
• I need help with commitment in prayer

Simply send your message in your own words.
"""


def find_matching_sermons(user_message: str):
    """
    Returns sermons with the highest matching score.
    Longer keyword phrases receive higher scores.
    """

    cleaned_message = re.sub(r"[^\w\s]", " ", user_message.lower())
    cleaned_message = re.sub(r"\s+", " ", cleaned_message).strip()

    scored_sermons = []

    for sermon in sermons:
        score = 0

        for keyword in sermon["keywords"]:
            keyword = keyword.lower().strip()

            if keyword in cleaned_message:
                score += len(keyword.split())

        if score > 0:
            scored_sermons.append((score, sermon))

    if not scored_sermons:
        return []

    highest_score = max(score for score, _ in scored_sermons)

    best_matches = [
        sermon
        for score, sermon in scored_sermons
        if score == highest_score
    ]

    random.shuffle(best_matches)

    return best_matches[:3]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE)


async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    if not user_message:
        return

    matched_sermons = find_matching_sermons(user_message)

    if matched_sermons:
        response = "📖 Here are some recommended messages for you:\n\n"

        seen_links = set()

        for sermon in matched_sermons:
            if sermon["link"] in seen_links:
                continue

            seen_links.add(sermon["link"])

            response += f"✨ {sermon['title']}\n"
            response += f"{sermon['link']}\n\n"

    else:
        response = (
            "🙏 I couldn’t find an exact match for that.\n\n"
            "Try describing it differently like:\n"
            "• I need breakthrough in my career\n"
            "• I feel stuck in life\n"
            "• I need favour in business\n"
            "• I need spiritual growth\n"
        )

    await update.message.reply_text(response)


app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

print("Apostle Omo's A.I is running...")

app.run_polling(
    poll_interval=3,
    timeout=30,
    bootstrap_retries=5,
)

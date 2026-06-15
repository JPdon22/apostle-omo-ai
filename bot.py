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


# =========================
# BOT TOKEN
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")


# =========================
# AI MODEL (LOAD FIRST)
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp")


# =========================
# SERMON STORE + EMBEDDINGS
# =========================
sermon_store = sermons.copy()

for sermon in sermon_store:
    text = sermon["title"] + " " + sermon["description"]
    sermon["embedding"] = model.encode(text)


# =========================
# WELCOME MESSAGE
# =========================
WELCOME_MESSAGE = """
🙏 Welcome to Apostle Omo's A.I.

Describe what you're going through and I’ll recommend relevant teachings.

Examples:
• I need favour in business
• I feel stuck in life
• I need breakthrough
• I feel spiritually down
• I need restoration
"""


# =========================
# HYBRID MATCHING ENGINE
# =========================
def find_matching_sermons(user_message: str):
    clean_msg = re.sub(r"[^\w\s]", " ", user_message.lower())
    msg_tokens = set(clean_msg.split())

    user_vec = model.encode(user_message)

    scored = []

    for sermon in sermon_store:

        # ---------------- KEYWORD SCORE ----------------
        keyword_score = 0

        for keyword in sermon["keywords"]:
            kw = keyword.lower()

            if kw in clean_msg:
                keyword_score += 2
            else:
                kw_tokens = set(kw.split())
                keyword_score += len(kw_tokens & msg_tokens) * 0.3

        keyword_score = min(keyword_score / 5, 1)

        # ---------------- SEMANTIC SCORE ----------------
        semantic_score = cosine_similarity(
            [user_vec],
            [sermon["embedding"]]
        )[0][0]

        # ---------------- FINAL SCORE ----------------
        final_score = (0.45 * keyword_score) + (0.55 * semantic_score)

        scored.append((final_score, sermon))

    if not scored:
        return []

    scored.sort(reverse=True, key=lambda x: x[0])

    top_score = scored[0][0]

    if top_score < 0.35:
        return []

    best = [s for score, s in scored if abs(score - top_score) < 0.03]

    random.shuffle(best)

    return best[:3]


# =========================
# START COMMAND
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE)


# =========================
# MESSAGE HANDLER
# =========================
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
            "• I feel stuck in life\n"
            "• I need breakthrough\n"
            "• I need favour in business\n"
            "• I need spiritual growth\n"
        )

    await update.message.reply_text(response)


# =========================
# APP SETUP
# =========================
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

print("Apostle Omo's A.I is running...")

app.run_polling(
    poll_interval=3,
    timeout=30,
    bootstrap_retries=5
)

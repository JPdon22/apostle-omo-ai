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

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sermons import sermons


# =========================
# BOT TOKEN
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")


# =========================
# LOAD AI MODEL (MUST BE FIRST)
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
🙏 Welcome to Apostle Omo's A.I

Tell me what you're going through and I will recommend relevant messages.

Examples:
- I need breakthrough
- I feel stuck in life
- I need favour in business
- I need spiritual growth
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

        # 🔧 Reduced divisor so partial matches contribute more
        keyword_score = min(keyword_score / 2, 1)

        # ---------------- SEMANTIC SCORE ----------------
        semantic_score = cosine_similarity(
            [user_vec],
            [sermon["embedding"]]
        )[0][0]

        # ---------------- FINAL SCORE ----------------
        # 🔧 Increased semantic weight since natural language rarely
        # matches keywords exactly
        final_score = (0.2 * keyword_score) + (0.8 * semantic_score)

        scored.append((final_score, sermon))

    scored.sort(reverse=True, key=lambda x: x[0])

    # 🔍 DEBUG: print top 5 scores so you can tune the threshold
    print(f"\n--- Scores for: '{user_message}' ---")
    for score, sermon in scored[:5]:
        print(f"{score:.3f} - {sermon['title']}")
    print("---------------------------------\n")

    top_score = scored[0][0]

    # 🔧 Lowered threshold (tune further based on debug output above)
    if top_score < 0.12:
        return []

    best = [s for score, s in scored if abs(score - top_score) < 0.05]

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
        response = "📖 Here are recommended messages for you:\n\n"

        seen = set()

        for sermon in matched_sermons:
            if sermon["link"] in seen:
                continue

            seen.add(sermon["link"])

            response += f"✨ {sermon['title']}\n"
            response += f"{sermon['link']}\n\n"

    else:
        response = (
            "🙏 I couldn’t find a strong match yet.\n\n"
            "Try saying:\n"
            "• I need breakthrough\n"
            "• I feel stuck in life\n"
            "• I need favour in business\n"
            "• I need spiritual growth"
        )

    await update.message.reply_text(response)


# =========================
# APP SETUP
# =========================
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

print("Apostle Omo's A.I is running...")

app.run_polling()

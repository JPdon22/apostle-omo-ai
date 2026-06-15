from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)

import os
import random
import re
import logging
import sqlite3
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sermons import sermons


# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =========================
# BOT TOKEN
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    logger.critical("BOT_TOKEN environment variable is not set. Exiting.")
    raise SystemExit("BOT_TOKEN environment variable is not set.")


# =========================
# DATABASE SETUP (for analytics / logging)
# =========================
DB_PATH = "bot_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            message TEXT,
            top_score REAL,
            matched_titles TEXT,
            timestamp TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sermon_title TEXT,
            rating TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_query(user_id, username, message, top_score, matched_titles):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO queries (user_id, username, message, top_score, matched_titles, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, username, message, top_score, matched_titles, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log query: {e}")

def log_feedback(user_id, sermon_title, rating):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO feedback (user_id, sermon_title, rating, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, sermon_title, rating, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")


# =========================
# LOAD AI MODEL (MUST BE FIRST)
# =========================
try:
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load model: {e}")
    raise SystemExit(f"Failed to load model: {e}")


# =========================
# VALIDATE SERMON DATA
# =========================
REQUIRED_FIELDS = ["title", "description", "keywords", "link"]

def validate_sermons(raw_sermons):
    if not raw_sermons:
        raise ValueError("The 'sermons' list is empty. Add at least one sermon entry.")

    seen_links = {}
    cleaned = []

    for i, sermon in enumerate(raw_sermons):
        missing = [f for f in REQUIRED_FIELDS if f not in sermon]
        if missing:
            raise ValueError(
                f"Sermon entry #{i} ('{sermon.get('title', 'UNKNOWN')}') "
                f"is missing required field(s): {missing}"
            )

        if not isinstance(sermon["keywords"], list) or not sermon["keywords"]:
            raise ValueError(
                f"Sermon entry #{i} ('{sermon['title']}') must have a non-empty 'keywords' list."
            )

        if not isinstance(sermon["title"], str) or not sermon["title"].strip():
            raise ValueError(f"Sermon entry #{i} has an empty or invalid 'title'.")

        if not isinstance(sermon["link"], str) or not sermon["link"].strip():
            raise ValueError(f"Sermon entry #{i} ('{sermon['title']}') has an empty or invalid 'link'.")

        link = sermon["link"].strip()
        if link in seen_links:
            logger.warning(
                f"Duplicate link detected: '{link}' used by both "
                f"'{seen_links[link]}' and '{sermon['title']}'. Check sermons.py for a data-entry mistake."
            )
        else:
            seen_links[link] = sermon["title"]

        cleaned.append(sermon)

    return cleaned


try:
    sermon_store = validate_sermons(sermons.copy())
    logger.info(f"Loaded and validated {len(sermon_store)} sermon entries.")
except ValueError as e:
    logger.critical(f"Sermon data validation failed: {e}")
    raise SystemExit(f"Sermon data validation failed: {e}")


# =========================
# SERMON EMBEDDINGS (BATCHED)
# =========================
logger.info("Generating sermon embeddings (batched)...")
texts = [s["title"] + " " + s["description"] for s in sermon_store]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

for sermon, emb in zip(sermon_store, embeddings):
    sermon["embedding"] = emb

logger.info("Sermon embeddings ready.")


# =========================
# MESSAGES
# =========================
WELCOME_MESSAGE = """
🙏 Welcome to Apostle Omo's A.I

Tell me what you're going through and I will recommend relevant messages.

Examples:
• I need breakthrough
• I feel stuck in life
• I need favour in business
• I need spiritual growth

You can also type /help anytime for guidance.
"""

HELP_MESSAGE = """
ℹ️ How to use this bot:

Just describe what you're going through in your own words, e.g.:
• "I feel anxious about my finances"
• "I need healing"
• "I'm struggling in my marriage"

I'll search for sermons that match what you shared.

Commands:
/start – Restart and see the welcome message
/help – Show this help message
"""

NO_MATCH_MESSAGE = (
    "🙏 I couldn't find a strong match yet.\n\n"
    "Try saying:\n"
    "• I need breakthrough\n"
    "• I feel stuck in life\n"
    "• I need favour in business\n"
    "• I need spiritual growth\n\n"
    "Or type /help for more guidance."
)

UNSUPPORTED_MESSAGE = (
    "🙏 I can only understand text messages right now. "
    "Please type out what you're going through, and I'll recommend a message for you."
)

GREETING_RESPONSE = (
    "🙏 Hello! I'm here to recommend sermons based on what you're going through.\n\n"
    "Just tell me how you're feeling or what you need, for example:\n"
    "• I feel stuck in life\n"
    "• I need a breakthrough\n"
    "• I'm struggling in my marriage\n\n"
    "Type /help anytime for more guidance."
)

# Quick greetings/small-talk/help phrases that shouldn't go through the matcher
GREETING_KEYWORDS = {
    "hi", "hello", "hey", "yo", "sup", "hiya", "howdy",
    "good morning", "good afternoon", "good evening", "good day",
    "morning", "evening", "afternoon",
    "what can you do", "what do you do", "how does this work",
    "how do you work", "what is this", "who are you",
    "what are you", "help me", "how can you help",
    "what can you help with", "are you a bot", "are you real",
    "thanks", "thank you", "thx", "ok", "okay", "cool", "nice",
    "test", "testing"
}


# =========================
# HYBRID MATCHING ENGINE
# =========================
def find_matching_sermons(user_message: str, exclude_links=None):
    if not sermon_store:
        return [], 0.0

    exclude_links = exclude_links or set()

    clean_msg = re.sub(r"[^\w\s]", " ", user_message.lower())
    msg_tokens = set(clean_msg.split())

    user_vec = model.encode(user_message)

    scored = []

    for sermon in sermon_store:
        if sermon["link"] in exclude_links:
            continue

        # ---------------- KEYWORD SCORE ----------------
        keyword_score = 0

        for keyword in sermon["keywords"]:
            kw = keyword.lower()

            if kw in clean_msg:
                keyword_score += 2
            else:
                kw_tokens = set(kw.split())
                keyword_score += len(kw_tokens & msg_tokens) * 0.3

        keyword_score = min(keyword_score / 2, 1)

        # ---------------- SEMANTIC SCORE ----------------
        semantic_score = cosine_similarity(
            [user_vec],
            [sermon["embedding"]]
        )[0][0]

        # ---------------- FINAL SCORE ----------------
        final_score = (0.2 * keyword_score) + (0.8 * semantic_score)

        scored.append((final_score, sermon))

    if not scored:
        return [], 0.0

    scored.sort(reverse=True, key=lambda x: x[0])

    top_score = scored[0][0]

    if top_score < 0.12:
        return [], top_score

    best = [s for score, s in scored if abs(score - top_score) < 0.05]

    random.shuffle(best)

    return best[:3], top_score


# =========================
# GREETING / SMALL-TALK DETECTION
# =========================
def is_greeting_or_smalltalk(user_message: str) -> bool:
    clean_check = re.sub(r"[^\w\s]", "", user_message.lower()).strip()
    clean_check = re.sub(r"\s+", " ", clean_check)

    if not clean_check or len(clean_check) < 3:
        return True

    if clean_check in GREETING_KEYWORDS:
        return True

    for phrase in GREETING_KEYWORDS:
        if (
            clean_check == phrase
            or clean_check.startswith(phrase + " ")
            or clean_check.endswith(" " + phrase)
        ):
            return True

    return False


# =========================
# COMMAND HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(WELCOME_MESSAGE)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_MESSAGE)


# =========================
# BUILD RESPONSE WITH FEEDBACK BUTTONS
# =========================
def build_sermon_response(matched_sermons):
    response = "📖 Here are recommended messages for you:\n\n"
    seen = set()
    buttons = []

    for sermon in matched_sermons:
        if sermon["link"] in seen:
            continue
        seen.add(sermon["link"])

        response += f"✨ {sermon['title']}\n{sermon['link']}\n\n"

        buttons.append([
            InlineKeyboardButton(f"👍 Helpful: {sermon['title'][:25]}", callback_data=f"fb_up|{sermon['title']}"),
            InlineKeyboardButton("👎", callback_data=f"fb_down|{sermon['title']}")
        ])

    buttons.append([InlineKeyboardButton("🔄 Show me more like this", callback_data="more")])

    return response.strip(), InlineKeyboardMarkup(buttons)


# =========================
# MESSAGE HANDLER
# =========================
async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    if not user_message or not user_message.strip():
        return

    user = update.effective_user

    # Catch greetings, small talk, and "what can you do?" style questions
    if is_greeting_or_smalltalk(user_message):
        await update.message.reply_text(GREETING_RESPONSE)
        return

    matched_sermons, top_score = find_matching_sermons(user_message)

    matched_titles = ", ".join(s["title"] for s in matched_sermons) if matched_sermons else "NONE"
    log_query(user.id, user.username or "unknown", user_message, float(top_score), matched_titles)
    logger.info(f"User {user.id} ({user.username}): '{user_message}' -> score={top_score:.3f}, matches={matched_titles}")

    if matched_sermons:
        # Save context for "show me more" follow-ups
        context.user_data["last_message"] = user_message
        context.user_data["shown_links"] = {s["link"] for s in matched_sermons}

        response, markup = build_sermon_response(matched_sermons)
        await update.message.reply_text(response, reply_markup=markup)
    else:
        await update.message.reply_text(NO_MATCH_MESSAGE)


# =========================
# CALLBACK HANDLER (feedback + "show more")
# =========================
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user = update.effective_user
    data = query.data

    await query.answer()

    if data.startswith("fb_up|") or data.startswith("fb_down|"):
        rating, title = data.split("|", 1)
        rating = "up" if rating == "fb_up" else "down"
        log_feedback(user.id, title, rating)
        logger.info(f"User {user.id} feedback: {rating} for '{title}'")
        await query.message.reply_text("🙏 Thank you for your feedback!")
        return

    if data == "more":
        last_message = context.user_data.get("last_message")
        shown_links = context.user_data.get("shown_links", set())

        if not last_message:
            await query.message.reply_text(
                "🙏 Please send a new message describing what you're going through, "
                "and I'll find relevant sermons."
            )
            return

        matched_sermons, top_score = find_matching_sermons(last_message, exclude_links=shown_links)

        if matched_sermons:
            context.user_data["shown_links"] |= {s["link"] for s in matched_sermons}
            response, markup = build_sermon_response(matched_sermons)
            await query.message.reply_text(response, reply_markup=markup)
        else:
            await query.message.reply_text(
                "🙏 No more matches for that topic. "
                "Feel free to describe what else you're going through."
            )
        return


# =========================
# UNSUPPORTED MESSAGE TYPES
# =========================
async def handle_unsupported(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(UNSUPPORTED_MESSAGE)


# =========================
# ERROR HANDLER
# =========================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error: {context.error}", exc_info=context.error)


# =========================
# APP SETUP
# =========================
def main():
    init_db()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    app.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_unsupported))

    app.add_error_handler(error_handler)

    logger.info("Apostle Omo's A.I is running...")
    app.run_polling()


if __name__ == "__main__":
    main()

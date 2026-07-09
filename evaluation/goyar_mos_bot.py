#!/usr/bin/env python3
"""
Goyar -- Telegram-based MOS/SMOS/intelligibility annotation bot
(paper Appendix "Annotation Interface and Procedure").

Used to collect the subjective evaluation reported in Section 5: for each
item, a rater is shown the reference text, then listens to the reference
speaker's recording followed by the synthesized clip, and rates the
synthesized sample on three criteria via inline buttons:

    naturalness   -- 1-5 MOS ("does it sound robotic?")
    similarity    -- 1-5 SMOS ("does it sound like the reference speaker?")
    text_match    -- 1-5 in 0.5 increments, intelligibility MOS
                      ("does the audio match the written text?")

Every rating is written to disk immediately (not just on final submit), and
a user's position/answers are checkpointed so a rater can close Telegram and
resume later without losing progress.

Setup:
    pip install -r evaluation/requirements.txt
    export GOYAR_BOT_TOKEN=...          # from @BotFather -- required, no default
    export GOYAR_SAMPLES_CSV=samples.csv   # optional, see schema below

Input CSV schema (one row per item to rate), default columns read:
    transcript        -- reference text shown to the rater
    reference_audio    -- path to the reference speaker's recording
    generated_audio     -- path to the synthesized clip
    speaker_id, Gender, age, accent  -- optional, recorded alongside ratings
    speaker_type        -- optional, e.g. "unseen" (recorded alongside ratings)

Usage:
    python evaluation/goyar_mos_bot.py
"""

import json
import logging
import os
import random
from datetime import datetime
from typing import Dict

import pandas as pd
import telebot
from telebot import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("GOYAR_BOT_TOKEN")
if not BOT_TOKEN:
    raise SystemExit(
        "GOYAR_BOT_TOKEN is not set. Create a bot via @BotFather on Telegram and "
        "export GOYAR_BOT_TOKEN=<token> before running this script."
    )
bot = telebot.TeleBot(BOT_TOKEN)

# File paths (all overridable via env vars; defaults keep everything local to cwd).
SAMPLES_CSV = os.environ.get("GOYAR_SAMPLES_CSV", "samples.csv")
MOS_RESULTS_CSV = os.environ.get("GOYAR_MOS_RESULTS_CSV", "MOS.csv")
USERS_CSV = os.environ.get("GOYAR_USERS_CSV", "USERS.csv")
USER_PROGRESS_JSON = os.environ.get("GOYAR_USER_PROGRESS_JSON", "user_progress.json")

# Global state
audio_samples = []
user_sessions = {}


class UserSession:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.username = ""
        self.name = ""
        self.age = ""
        self.gender = ""
        self.current_audio_index = 0
        self.audio_samples = []
        self.scores = {}
        self.audio_played = {}
        self.message_seen = {}
        self.registration_complete = False
        self.evaluation_started = False

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "username": self.username,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "current_audio_index": self.current_audio_index,
            "audio_samples": self.audio_samples,
            "scores": self.scores,
            "audio_played": self.audio_played,
            "message_seen": self.message_seen,
            "registration_complete": self.registration_complete,
            "evaluation_started": self.evaluation_started,
        }

    @classmethod
    def from_dict(cls, data):
        session = cls(data["user_id"])
        session.username = data.get("username", "")
        session.name = data.get("name", "")
        session.age = data.get("age", "")
        session.gender = data.get("gender", "")
        session.current_audio_index = data.get("current_audio_index", 0)
        session.audio_samples = data.get("audio_samples", [])
        session.scores = data.get("scores", {})
        session.audio_played = data.get("audio_played", {})
        session.message_seen = data.get("message_seen", {})
        session.registration_complete = data.get("registration_complete", False)
        session.evaluation_started = data.get("evaluation_started", False)
        return session


def load_user_progress():
    global user_sessions
    if os.path.exists(USER_PROGRESS_JSON):
        try:
            with open(USER_PROGRESS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                for user_id, session_data in data.items():
                    user_sessions[int(user_id)] = UserSession.from_dict(session_data)
        except Exception as exc:
            logger.error(f"Error loading user progress: {exc}")


def save_user_progress():
    """Atomically saves user progress, keeping a .backup of the last good write."""
    try:
        data = {str(uid): session.to_dict() for uid, session in user_sessions.items()}
        if os.path.exists(USER_PROGRESS_JSON):
            backup_file = f"{USER_PROGRESS_JSON}.backup"
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(USER_PROGRESS_JSON, backup_file)
        with open(USER_PROGRESS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error(f"Error saving user progress: {exc}")
        backup_file = f"{USER_PROGRESS_JSON}.backup"
        if os.path.exists(backup_file) and not os.path.exists(USER_PROGRESS_JSON):
            os.rename(backup_file, USER_PROGRESS_JSON)
            logger.info("Restored progress from backup")


def load_audio_samples():
    """Loads and shuffles the rating queue from SAMPLES_CSV."""
    global audio_samples
    try:
        df = pd.read_csv(SAMPLES_CSV)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        audio_samples = df.to_dict("records")
        logger.info(f"Loaded {len(audio_samples)} audio samples from {SAMPLES_CSV}")
    except Exception as exc:
        logger.error(f"Error loading audio samples from {SAMPLES_CSV}: {exc}")
        audio_samples = []


def get_user_session(user_id: int) -> UserSession:
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    return user_sessions[user_id]


def create_rating_keyboard(question_type: str, current_score: float = 0):
    """Inline keyboard for a rating question. text_match uses 0.5 increments
    (intelligibility MOS, Appendix); the other two questions use whole 1-5."""
    if question_type == "text_match":
        keyboard = types.InlineKeyboardMarkup(row_width=5)
        scores = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        buttons = [
            types.InlineKeyboardButton(f"{'✅' if score == current_score else ''}{score}", callback_data=f"rate_{question_type}_{score}")
            for score in scores
        ]
        keyboard.row(*buttons[:5])
        keyboard.row(*buttons[5:])
    else:
        keyboard = types.InlineKeyboardMarkup(row_width=5)
        buttons = [
            types.InlineKeyboardButton(f"{'✅' if i == current_score else ''}{i}", callback_data=f"rate_{question_type}_{i}")
            for i in range(1, 6)
        ]
        keyboard.row(*buttons)
    return keyboard


def create_navigation_keyboard(session: UserSession):
    keyboard = types.InlineKeyboardMarkup()
    current = session.current_audio_index
    total = len(session.audio_samples)

    keyboard.row(types.InlineKeyboardButton(f"📊 پیشرفت ({current + 1}/{total})", callback_data="show_progress"))
    keyboard.row(types.InlineKeyboardButton("🎯 رفتن به صدای مشخص", callback_data="go_to_audio"))

    current_audio_id = str(current)
    current_completed = all(f"{current_audio_id}_{q}" in session.scores for q in ("naturalness", "similarity", "text_match"))

    if current_completed:
        keyboard.row(types.InlineKeyboardButton(f"✏️ ویرایش نمرات صدا {current + 1}", callback_data=f"edit_audio_{current}"))
        if current < total - 1:
            keyboard.row(types.InlineKeyboardButton("➡️ رفتن به صدای بعدی", callback_data="next_audio"))

    completed_audios = [
        i
        for i in range(current)
        if all(f"{i}_{q}" in session.scores for q in ("naturalness", "similarity", "text_match"))
    ]
    if completed_audios:
        keyboard.row(types.InlineKeyboardButton("📝 ویرایش صداهای قبلی", callback_data="show_completed_list"))

    if len(session.scores) >= total * 3:
        keyboard.row(types.InlineKeyboardButton("✅ ثبت نهایی همه امتیازها", callback_data="submit_final"))

    return keyboard


@bot.message_handler(commands=["start"])
def start_command(message):
    user_id = message.from_user.id
    session = get_user_session(user_id)
    session.username = message.from_user.username or ""

    welcome_text = """
🎵 **ربات ارزیابی سیستم تبدیل متن به گفتار فارسی** 🎵

خوش آمدید! این ربات برای ارزیابی کیفیت نمونه‌های صوتی سیستم تبدیل متن به گفتار فارسی استفاده می‌شود.

**آنچه شما انجام خواهید داد:**
1️⃣ به فایل‌های صوتی به همراه متن آنها گوش دهید
2️⃣ هر صدا را بر اساس ۳ جنبه امتیازدهی کنید:
   • **طبیعی بودن**: صدا چقدر طبیعی به نظر می‌رسد؟ (مقیاس ۱ تا ۵)
   • **شباهت گوینده**: صدا چقدر شبیه گوینده اصلی است؟ (مقیاس ۱ تا ۵)
   • **تطبیق با متن**: صدا چقدر با متن مطابقت دارد؟ (مقیاس ۱ تا ۵ با ۰.۵ فاصله)

**دستورالعمل‌ها:**
📱 از دکمه‌های درون‌خطی برای امتیازدهی و ناوبری استفاده کنید
🔊 قبل از امتیازدهی به هر فایل صوتی گوش دهید
✏️ همیشه می‌توانید امتیازهای خود را ویرایش کنید
💾 پیشرفت شما به صورت خودکار ذخیره می‌شود

بیایید با اطلاعات پایه‌ای درباره شما شروع کنیم.
    """

    keyboard = types.InlineKeyboardMarkup()
    if session.registration_complete:
        keyboard.row(types.InlineKeyboardButton("🎯 ادامه ارزیابی", callback_data="continue_eval"))
        keyboard.row(types.InlineKeyboardButton("📝 بروزرسانی پروفایل", callback_data="start_registration"))
    else:
        keyboard.row(types.InlineKeyboardButton("📝 شروع ثبت‌نام", callback_data="start_registration"))

    bot.send_message(message.chat.id, welcome_text, parse_mode="Markdown", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: call.data == "start_registration")
def start_registration(call):
    bot.edit_message_text("لطفاً نام خود را وارد کنید:", call.message.chat.id, call.message.message_id)
    bot.register_next_step_handler(call.message, get_name)


def get_name(message):
    session = get_user_session(message.from_user.id)
    session.name = message.text.strip()

    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(
        types.InlineKeyboardButton("👨 مرد", callback_data="gender_male"),
        types.InlineKeyboardButton("👩 زن", callback_data="gender_female"),
        types.InlineKeyboardButton("🤷 سایر", callback_data="gender_other"),
    )
    bot.send_message(message.chat.id, f"ممنون {session.name}! لطفاً جنسیت خود را انتخاب کنید:", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: call.data.startswith("gender_"))
def handle_gender(call):
    session = get_user_session(call.from_user.id)
    session.gender = call.data.split("_")[1]
    gender_map = {"male": "مرد", "female": "زن", "other": "سایر"}

    bot.edit_message_text(
        f"جنسیت: {gender_map.get(session.gender, session.gender)}\n\nلطفاً سن خود را وارد کنید:",
        call.message.chat.id,
        call.message.message_id,
    )
    bot.register_next_step_handler(call.message, get_age)


def get_age(message):
    session = get_user_session(message.from_user.id)
    try:
        age = int(message.text.strip())
        if not (10 <= age <= 100):
            raise ValueError
    except ValueError:
        bot.send_message(message.chat.id, "لطفاً سنی معتبر بین ۱۰ تا ۱۰۰ سال وارد کنید:")
        bot.register_next_step_handler(message, get_age)
        return

    session.age = str(age)
    session.registration_complete = True
    save_user_to_csv(session)

    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(types.InlineKeyboardButton("🎯 شروع ارزیابی", callback_data="start_evaluation"))

    gender_map = {"male": "مرد", "female": "زن", "other": "سایر"}
    confirmation = f"""
✅ **ثبت‌نام تکمیل شد!**

**اطلاعات شما:**
👤 نام: {session.name}
👫 جنسیت: {gender_map.get(session.gender, session.gender)}
📅 سن: {session.age}

آماده برای شروع ارزیابی صوتی هستید؟
    """
    bot.send_message(message.chat.id, confirmation, parse_mode="Markdown", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: call.data in ["start_evaluation", "continue_eval"])
def start_evaluation(call):
    session = get_user_session(call.from_user.id)
    if not session.registration_complete:
        bot.answer_callback_query(call.id, "لطفاً ابتدا ثبت‌نام را تکمیل کنید!")
        return

    if not session.audio_samples:
        session.audio_samples = audio_samples.copy()
        random.seed(call.from_user.id)  # consistent per-user shuffle
        random.shuffle(session.audio_samples)
        logger.info(f"Assigned {len(session.audio_samples)} audio samples to user {call.from_user.id}")

    session.evaluation_started = True
    save_user_progress()

    if call.data == "continue_eval":
        bot.edit_message_text(
            f"ادامه ارزیابی... صدای فعلی: {session.current_audio_index + 1}/{len(session.audio_samples)}",
            call.message.chat.id,
            call.message.message_id,
        )
    else:
        bot.edit_message_text("شروع ارزیابی...", call.message.chat.id, call.message.message_id)

    show_current_audio(call.message.chat.id, session)


def show_current_audio(chat_id: int, session: UserSession):
    if session.current_audio_index >= len(session.audio_samples):
        show_completion(chat_id, session)
        return

    current_audio = session.audio_samples[session.current_audio_index]
    audio_id = f"{session.current_audio_index}"
    session.message_seen[audio_id] = datetime.now().isoformat()

    transcript = current_audio.get("transcript", "متن در دسترس نیست")
    audio_info = f"""
🎵 **صدای شماره {session.current_audio_index + 1} از {len(session.audio_samples)}**

📝 **متن:** {transcript}

🔊 لطفاً به هر دو فایل صوتی گوش دهید، سپس آن‌ها را در سه جنبه زیر امتیازدهی کنید:
    """
    bot.send_message(chat_id, audio_info, parse_mode="Markdown")

    reference_audio_file = current_audio.get("reference_audio", "")
    if reference_audio_file and os.path.exists(reference_audio_file):
        try:
            with open(reference_audio_file, "rb") as audio:
                bot.send_voice(chat_id, audio, caption="🎯 **صدای مرجع (اصلی)**", parse_mode="Markdown")
        except Exception as exc:
            logger.error(f"Error sending reference audio: {exc}")
            bot.send_message(chat_id, f"⚠️ فایل صوتی مرجع یافت نشد: {reference_audio_file}")

    generated_audio_file = current_audio.get("generated_audio", "")
    if generated_audio_file and os.path.exists(generated_audio_file):
        try:
            with open(generated_audio_file, "rb") as audio:
                keyboard = create_navigation_keyboard(session)
                bot.send_voice(chat_id, audio, caption="🤖 **صدای تولید شده**", parse_mode="Markdown", reply_markup=keyboard)
            session.audio_played[audio_id] = datetime.now().isoformat()
        except Exception as exc:
            logger.error(f"Error sending generated audio: {exc}")
            bot.send_message(
                chat_id, f"⚠️ فایل صوتی تولید شده یافت نشد: {generated_audio_file}", reply_markup=create_navigation_keyboard(session)
            )
    else:
        bot.send_message(chat_id, "⚠️ مسیر فایل صوتی در دسترس نیست", reply_markup=create_navigation_keyboard(session))

    show_rating_questions(chat_id, session)


def show_rating_questions(chat_id: int, session: UserSession):
    audio_id = f"{session.current_audio_index}"
    questions = [
        (
            "naturalness",
            "🎭 **طبیعی بودن صدا**: صدای تولید شده چقدر طبیعی به نظر می‌رسد؟\n\n"
            "۱ = کاملاً غیرطبیعی و ماشینی\n"
            "۲ = نسبتاً غیرطبیعی با نقص‌های واضح\n"
            "۳ = متوسط، قابل قبول اما نه کاملاً طبیعی\n"
            "۴ = تقریباً طبیعی با کمی نقص\n"
            "۵ = کاملاً طبیعی و انسان‌مانند",
        ),
        (
            "similarity",
            "👤 **شباهت گوینده**: صدای تولید شده چقدر شبیه صدای مرجع (اصلی) است؟\n\n"
            "۱ = کاملاً متفاوت، مثل شخص دیگری\n"
            "۲ = نسبتاً متفاوت، شباهت کمی دارد\n"
            "۳ = متوسط، تا حدودی شبیه است\n"
            "۴ = خیلی شبیه با اختلاف جزئی\n"
            "۵ = دقیقاً مثل همان گوینده",
        ),
        (
            "text_match",
            "📝 **تطبیق با متن**: صدای تولید شده چقدر با متن نوشته شده مطابقت دارد؟\n\n"
            "۱ = کاملاً با متن مطابقت ندارد\n"
            "۱.۵ = خیلی کم مطابقت دارد\n"
            "۲ = کمی مطابقت دارد اما اختلافات زیادی وجود دارد\n"
            "۲.۵ = تا حدودی مطابقت دارد\n"
            "۳ = متوسط، مطابقت قابل قبول\n"
            "۳.۵ = خوب مطابقت دارد\n"
            "۴ = خیلی خوب مطابقت دارد\n"
            "۴.۵ = تقریباً کامل مطابقت دارد\n"
            "۵ = کاملاً و دقیقاً با متن مطابقت دارد",
        ),
    ]

    for q_type, q_text in questions:
        current_score = session.scores.get(f"{audio_id}_{q_type}", 0)
        bot.send_message(chat_id, q_text, parse_mode="Markdown", reply_markup=create_rating_keyboard(q_type, current_score))


@bot.callback_query_handler(func=lambda call: call.data.startswith("rate_"))
def handle_rating(call):
    session = get_user_session(call.from_user.id)

    parts = call.data.split("_")
    if len(parts) < 3:
        return
    question_type = "_".join(parts[1:-1])
    try:
        score = float(parts[-1])
    except ValueError:
        bot.answer_callback_query(call.id, "خطا در امتیاز!")
        return

    audio_id = f"{session.current_audio_index}"
    session.scores[f"{audio_id}_{question_type}"] = score
    save_individual_mos_result(session, question_type, score)
    save_user_progress()

    try:
        bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=create_rating_keyboard(question_type, score))
        bot.answer_callback_query(call.id, f"✅ امتیاز ثبت شد: {score}/5")
    except Exception:
        bot.answer_callback_query(call.id, f"✅ امتیاز ذخیره شد: {score}/5")

    questions_answered = sum(1 for q in ("naturalness", "similarity", "text_match") if f"{audio_id}_{q}" in session.scores)
    if questions_answered == 3:
        bot.send_message(
            call.message.chat.id,
            f"✅ صدای {session.current_audio_index + 1} تکمیل شد! می‌توانید نمرات را ویرایش کنید یا به صدای بعدی بروید.",
            reply_markup=create_navigation_keyboard(session),
        )


@bot.callback_query_handler(func=lambda call: call.data == "go_to_audio")
def handle_go_to_audio(call):
    session = get_user_session(call.from_user.id)
    total_audios = len(session.audio_samples)
    current_index = session.current_audio_index

    keyboard = types.InlineKeyboardMarkup(row_width=5)
    buttons = []
    for i in range(total_audios):
        is_completed = all(f"{i}_{q}" in session.scores for q in ("naturalness", "similarity", "text_match"))
        text = f"🔵{i + 1}" if i == current_index else (f"✅{i + 1}" if is_completed else f"⭕{i + 1}")
        buttons.append(types.InlineKeyboardButton(text, callback_data=f"jump_to_audio_{i}"))
        if len(buttons) == 5 or i == total_audios - 1:
            keyboard.row(*buttons)
            buttons = []
    keyboard.row(types.InlineKeyboardButton("🔙 برگشت", callback_data="back_to_current"))

    legend_text = f"""
🎯 **انتخاب صدا برای رفتن:**

🔵 = صدای فعلی ({current_index + 1})
✅ = تکمیل شده
⭕ = تکمیل نشده

روی شماره صدا کلیک کنید:
    """
    bot.send_message(call.message.chat.id, legend_text, parse_mode="Markdown", reply_markup=keyboard)
    bot.answer_callback_query(call.id, "انتخاب صدا مورد نظر")


@bot.callback_query_handler(func=lambda call: call.data.startswith("jump_to_audio_"))
def handle_jump_to_audio(call):
    session = get_user_session(call.from_user.id)
    audio_index = int(call.data.split("_")[3])
    if audio_index >= len(session.audio_samples):
        bot.answer_callback_query(call.id, "خطا: صدای نامعتبر!")
        return

    session.current_audio_index = audio_index
    save_user_progress()
    bot.answer_callback_query(call.id, f"رفتن به صدای {audio_index + 1}")
    show_current_audio(call.message.chat.id, session)


@bot.callback_query_handler(func=lambda call: call.data.startswith("edit_audio_"))
def handle_edit_audio(call):
    session = get_user_session(call.from_user.id)
    audio_index = int(call.data.split("_")[2])
    if audio_index >= len(session.audio_samples):
        bot.answer_callback_query(call.id, "خطا: صدای نامعتبر!")
        return

    session.current_audio_index = audio_index
    save_user_progress()
    bot.answer_callback_query(call.id, f"ویرایش صدای {audio_index + 1}...")
    show_current_audio(call.message.chat.id, session)


@bot.callback_query_handler(func=lambda call: call.data == "next_audio")
def handle_next_audio(call):
    session = get_user_session(call.from_user.id)
    if session.current_audio_index < len(session.audio_samples) - 1:
        session.current_audio_index += 1
        save_user_progress()
        bot.answer_callback_query(call.id, f"رفتن به صدای {session.current_audio_index + 1}...")
        show_current_audio(call.message.chat.id, session)
    else:
        bot.answer_callback_query(call.id, "🎉 تبریک! همه صداها ارزیابی شدند!")
        show_completion(call.message.chat.id, session)


@bot.callback_query_handler(func=lambda call: call.data == "show_completed_list")
def handle_show_completed_list(call):
    session = get_user_session(call.from_user.id)
    completed_audios = []
    for i in range(len(session.audio_samples)):
        naturalness = session.scores.get(f"{i}_naturalness")
        similarity = session.scores.get(f"{i}_similarity")
        text_match = session.scores.get(f"{i}_text_match")
        if naturalness and similarity and text_match:
            completed_audios.append({"index": i, "naturalness": naturalness, "similarity": similarity, "text_match": text_match})

    if not completed_audios:
        bot.answer_callback_query(call.id, "هیچ صدای تکمیل شده‌ای وجود ندارد!")
        return

    keyboard = types.InlineKeyboardMarkup()
    for audio in completed_audios:
        text = f"صدا {audio['index'] + 1}: طبیعی={audio['naturalness']}, شباهت={audio['similarity']}, متن={audio['text_match']}"
        keyboard.row(types.InlineKeyboardButton(text, callback_data=f"edit_audio_{audio['index']}"))
    keyboard.row(types.InlineKeyboardButton("🔙 برگشت", callback_data="back_to_current"))

    bot.send_message(
        call.message.chat.id,
        "📝 **صداهای تکمیل شده برای ویرایش:**\n\nروی هر صدا کلیک کنید تا نمرات آن را ویرایش کنید:",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    bot.answer_callback_query(call.id, "لیست صداهای تکمیل شده نمایش داده شد")


@bot.callback_query_handler(func=lambda call: call.data == "back_to_current")
def handle_back_to_current(call):
    session = get_user_session(call.from_user.id)
    bot.answer_callback_query(call.id, f"برگشت به صدای فعلی ({session.current_audio_index + 1})")
    show_current_audio(call.message.chat.id, session)


@bot.callback_query_handler(func=lambda call: call.data == "show_progress")
def show_progress(call):
    session = get_user_session(call.from_user.id)
    total_audios = len(session.audio_samples)
    total_questions = total_audios * 3
    answered_questions = len(session.scores)

    scores_text = f"""
📊 **پیشرفت و نمرات شما**

🎵 صدا: {session.current_audio_index + 1}/{total_audios}
✅ سوالات پاسخ داده شده: {answered_questions}/{total_questions}
📈 پیشرفت: {(answered_questions / total_questions) * 100:.1f}%

📝 **نمرات ثبت شده:**
"""
    for audio_idx in range(session.current_audio_index + 1):
        naturalness = session.scores.get(f"{audio_idx}_naturalness", "❌")
        similarity = session.scores.get(f"{audio_idx}_similarity", "❌")
        text_match = session.scores.get(f"{audio_idx}_text_match", "❌")
        transcript = ""
        if audio_idx < len(session.audio_samples):
            full = session.audio_samples[audio_idx].get("transcript", "")
            transcript = full[:50] + "..." if len(full) > 50 else full
        scores_text += f"\n🎵 **صدا {audio_idx + 1}:** {transcript}\n   🎭 طبیعی بودن: {naturalness}/5\n   👤 شباهت گوینده: {similarity}/5\n   📝 تطبیق با متن: {text_match}/5\n"

    bot.send_message(call.message.chat.id, scores_text, parse_mode="Markdown")
    bot.answer_callback_query(call.id, "✅ جزئیات پیشرفت ارسال شد")


@bot.callback_query_handler(func=lambda call: call.data == "submit_final")
def submit_final_scores(call):
    session = get_user_session(call.from_user.id)
    logger.info(f"Final evaluation completed for user {session.user_id}; results already saved incrementally.")

    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(types.InlineKeyboardButton("🔄 شروع ارزیابی جدید", callback_data="restart_eval"))

    completion_text = f"""
🎉 **ارزیابی کامل شد!**

ممنون {session.name} بابت بازخورد ارزشمندتان!

**آمار شما:**
✅ کل صداهای امتیازدهی شده: {len(session.audio_samples)}
📊 کل امتیازهای ثبت شده: {len(session.scores)}

پاسخ‌های شما ذخیره شده و به بهبود سیستم تبدیل متن به گفتار ما کمک خواهد کرد.
    """
    bot.edit_message_text(completion_text, call.message.chat.id, call.message.message_id, parse_mode="Markdown", reply_markup=keyboard)


def save_individual_mos_result(session: UserSession, question_type: str, score: float):
    """Writes each rating to MOS_RESULTS_CSV immediately (not just on final submit)."""
    if session.current_audio_index >= len(session.audio_samples):
        return

    audio_sample = session.audio_samples[session.current_audio_index]
    audio_id = str(session.current_audio_index)
    result = {
        "user_id": session.user_id,
        "username": session.username,
        "user_name": session.name,
        "user_age": session.age,
        "user_gender": session.gender,
        "audio_index": session.current_audio_index,
        "speaker_id": audio_sample.get("speaker_id", ""),
        "speaker_gender": audio_sample.get("Gender", ""),
        "speaker_age": audio_sample.get("age", ""),
        "accent": audio_sample.get("accent", ""),
        "generated_audio": audio_sample.get("generated_audio", ""),
        "reference_audio": audio_sample.get("reference_audio", ""),
        "transcript": audio_sample.get("transcript", ""),
        "speaker_type": audio_sample.get("speaker_type", ""),
        "question_type": question_type,
        "rating_score": score,
        "audio_played": session.audio_played.get(audio_id, ""),
        "message_seen": session.message_seen.get(audio_id, ""),
        "rating_timestamp": datetime.now().isoformat(),
    }

    try:
        if os.path.exists(MOS_RESULTS_CSV):
            existing_df = pd.read_csv(MOS_RESULTS_CSV)
            mask = (
                (existing_df["user_id"] == session.user_id)
                & (existing_df["audio_index"] == session.current_audio_index)
                & (existing_df["question_type"] == question_type)
            )
            combined_df = pd.concat([existing_df[~mask], pd.DataFrame([result])], ignore_index=True)
        else:
            combined_df = pd.DataFrame([result])
        combined_df.to_csv(MOS_RESULTS_CSV, index=False)
    except Exception as exc:
        logger.error(f"Error saving individual MOS result: {exc}")


def save_user_to_csv(session: UserSession):
    user_data = {
        "user_id": session.user_id,
        "username": session.username,
        "name": session.name,
        "age": session.age,
        "gender": session.gender,
        "registration_date": datetime.now().isoformat(),
    }
    df = pd.DataFrame([user_data])
    if os.path.exists(USERS_CSV):
        existing_df = pd.read_csv(USERS_CSV)
        existing_df = existing_df[existing_df["user_id"] != session.user_id]
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)


def show_completion(chat_id: int, session: UserSession):
    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(types.InlineKeyboardButton("✅ ثبت نتایج", callback_data="submit_final"))
    keyboard.row(types.InlineKeyboardButton("📝 بررسی امتیازها", callback_data="review_scores"))

    completion_text = f"""
🎊 **همه صداها ارزیابی شدند!**

آفرین {session.name}! شما همه {len(session.audio_samples)} نمونه صوتی را امتیازدهی کردید.

**خلاصه:**
📊 سوالات پاسخ داده شده: {len(session.scores)}
🎵 صداهای تکمیل شده: {len(session.audio_samples)}

لطفاً امتیازهای خود را بررسی کنید و زمانی که آماده بودید، ثبت نهایی کنید.
    """
    bot.send_message(chat_id, completion_text, parse_mode="Markdown", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: call.data in ["ignore", "review_scores", "edit_scores"])
def handle_misc_callbacks(call):
    if call.data == "ignore":
        bot.answer_callback_query(call.id)
    elif call.data == "review_scores":
        session = get_user_session(call.from_user.id)
        total_questions = len(session.audio_samples) * 3
        bot.answer_callback_query(call.id, f"امتیازها: {len(session.scores)}/{total_questions} تکمیل شده", show_alert=True)
    elif call.data == "edit_scores":
        bot.answer_callback_query(call.id, "برای ویرایش امتیازها، می‌توانید روی دکمه‌های امتیازدهی کلیک کنید", show_alert=True)


@bot.callback_query_handler(func=lambda call: call.data == "restart_eval")
def restart_evaluation(call):
    session = get_user_session(call.from_user.id)
    session.current_audio_index = 0
    session.scores = {}
    session.audio_played = {}
    session.message_seen = {}
    session.audio_samples = []
    save_user_progress()

    bot.edit_message_text("ارزیابی جدید شروع می‌شود...", call.message.chat.id, call.message.message_id)
    start_evaluation(call)


@bot.message_handler(commands=["help"])
def help_command(message):
    help_text = """
🆘 **Help & Commands**

**Available Commands:**
/start - Start or restart the evaluation
/help - Show this help message
/progress - Show your current progress
/reset - Reset your evaluation (WARNING: deletes all progress!)

**How to Use:**
1. Complete registration (name, age, gender)
2. Listen to each audio clip carefully
3. Rate on 3 aspects: naturalness (1-5), speaker similarity (1-5), text match (1-5, 0.5 increments)

Progress is saved automatically after every rating.
    """
    bot.send_message(message.chat.id, help_text, parse_mode="Markdown")


@bot.message_handler(commands=["progress"])
def progress_command(message):
    if message.from_user.id not in user_sessions:
        bot.send_message(message.chat.id, "Please start the evaluation first using /start")
        return
    show_progress_details(message.chat.id, user_sessions[message.from_user.id])


def show_progress_details(chat_id: int, session: UserSession):
    if not session.evaluation_started:
        bot.send_message(chat_id, "Evaluation not started yet. Use /start to begin.")
        return

    total_audios = len(session.audio_samples)
    total_questions = total_audios * 3
    answered_questions = len(session.scores)
    completion_pct = (answered_questions / total_questions) * 100 if total_questions > 0 else 0
    progress_bar = "█" * int(completion_pct // 10) + "░" * (10 - int(completion_pct // 10))

    progress_text = f"""
📈 **Detailed Progress Report**

👤 **User:** {session.name}

📊 Current Audio: {session.current_audio_index + 1}/{total_audios}
Questions Answered: {answered_questions}/{total_questions}
Completion: {completion_pct:.1f}%

{progress_bar} {completion_pct:.1f}%
    """
    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(types.InlineKeyboardButton("🎯 Continue Evaluation", callback_data="continue_eval"))
    bot.send_message(chat_id, progress_text, parse_mode="Markdown", reply_markup=keyboard)


@bot.message_handler(commands=["reset"])
def reset_user_progress(message):
    keyboard = types.InlineKeyboardMarkup()
    keyboard.row(
        types.InlineKeyboardButton("✅ بله، همه را پاک کن", callback_data="confirm_reset"),
        types.InlineKeyboardButton("❌ انصراف", callback_data="cancel_reset"),
    )
    warning_text = """
⚠️ **هشدار: حذف پیشرفت**

این عمل همه پیشرفت شما را پاک خواهد کرد. آیا مطمئن هستید؟
    """
    bot.send_message(message.chat.id, warning_text, parse_mode="Markdown", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: call.data in ["confirm_reset", "cancel_reset"])
def handle_reset_confirmation(call):
    if call.data == "confirm_reset":
        user_sessions.pop(call.from_user.id, None)
        save_user_progress()
        bot.edit_message_text(
            "✅ پیشرفت شما پاک شد. از /start برای شروع مجدد استفاده کنید.", call.message.chat.id, call.message.message_id
        )
    else:
        bot.edit_message_text("❌ عملیات لغو شد. پیشرفت شما حفظ شده است.", call.message.chat.id, call.message.message_id)


def initialize_csv_files():
    if not os.path.exists(MOS_RESULTS_CSV):
        headers = [
            "user_id", "username", "user_name", "user_age", "user_gender",
            "audio_index", "speaker_id", "speaker_gender", "speaker_age", "accent",
            "generated_audio", "reference_audio", "transcript", "speaker_type",
            "question_type", "rating_score", "audio_played", "message_seen", "rating_timestamp",
        ]
        pd.DataFrame(columns=headers).to_csv(MOS_RESULTS_CSV, index=False)
    if not os.path.exists(USERS_CSV):
        pd.DataFrame(columns=["user_id", "username", "name", "age", "gender", "registration_date"]).to_csv(USERS_CSV, index=False)


def main():
    print("Starting Goyar MOS evaluation bot...")
    load_audio_samples()
    if not audio_samples:
        logger.error(f"No audio samples loaded from {SAMPLES_CSV} -- check the file exists and has the expected columns.")
        return

    load_user_progress()
    logger.info(f"Loaded progress for {len(user_sessions)} users")
    initialize_csv_files()

    logger.info(f"Loaded {len(audio_samples)} audio samples. Bot is running...")
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        save_user_progress()
        logger.info("Progress saved. Bot shutdown complete.")


if __name__ == "__main__":
    main()

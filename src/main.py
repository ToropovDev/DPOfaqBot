import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from src.config import BOT_TOKEN
from src.faq_model import faq_model
from src.texts import START_TEXT

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        text=START_TEXT.format(update.effective_user.first_name),
    )


async def handle_faq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    result = faq_model.get_best_match(
        query=user_text,
    )

    logger.info(f"user_text: {user_text}")
    logger.info(f"question: {result['question']}")
    logger.info(f"score: {result['score']}")
    logger.info(f"answer: {result['answer']}")
    logger.info(f"generated_answer: {result['generated_answer']}")

    await update.message.reply_text(
        text=result["generated_answer"],
    )


def main():
    logger.info("Запуск бота...")

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_faq))

    application.run_polling()

    logger.info("Бот остановлен")


if __name__ == "__main__":
    main()

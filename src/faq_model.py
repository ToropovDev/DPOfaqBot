import logging
from typing import Optional

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from src import texts
from src.config import EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME

logger = logging.getLogger(__name__)


class FAQModel:
    def __init__(self):
        logger.info("Initializing FAQ model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        logger.info("Loading FAQ data...")
        df = pd.read_parquet("data/faq_embeddings.parquet")
        self.question_list = df["user_question"].tolist()
        self.question_embeddings = torch.tensor(
            np.stack(df["question_embedding"].values)
        ).to(torch.float32)

        logger.info("Initializing generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)

        logger.info("Loading generation model...")
        self.generation_model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL_NAME
        )

        logger.info("Initializing pipeline...")
        self.generator = pipeline(
            "text-generation",
            model=self.generation_model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

        self.faq_path = "data/faq_full.parquet"

    def get_best_match(self, query: str) -> dict[str, Optional[str]]:
        logger.info(f"Processing query: {query}")

        query_embedding = self.embedding_model.encode(
            query, convert_to_tensor=True
        ).cpu()
        cos_scores = util.cos_sim(query_embedding, self.question_embeddings)[0]
        best_score_idx = torch.argmax(cos_scores)

        if cos_scores[best_score_idx] < 0.6:
            return {
                "answer": "Извините, я не могу ответить на этот вопрос.",
                "question": None,
                "score": float(cos_scores[best_score_idx]),
                "generated_answer": "Извините, я не могу ответить на этот вопрос.",
            }

        best_question = self.question_list[best_score_idx]

        df_full = pd.read_parquet(self.faq_path)
        matched_row = df_full[df_full["user_question"] == best_question].iloc[0]
        original_answer = matched_row["assistant_answer"]

        prompt = self._build_prompt(query, original_answer)
        generated_answer = self._generate_response(prompt)

        return {
            "answer": original_answer,
            "question": best_question,
            "score": float(cos_scores[best_score_idx]),
            "generated_answer": generated_answer.strip(),
        }

    @staticmethod
    def _build_prompt(user_query: str, context_answer: str) -> str:
        return texts.PROMPT.format(user_query, context_answer).strip()

    def _generate_response(self, prompt: str) -> str:
        response = self.generator(
            prompt,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            return_full_text=False,
        )
        text = response[0]["generated_text"]

        if "Ответ:" not in text:
            return text

        return text.split("Ответ:")[-1]


faq_model = FAQModel()

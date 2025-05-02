import json
import logging

from sentence_transformers import SentenceTransformer, util
import torch

from src.config import MODEL_NAME


logger = logging.getLogger(__name__)

class FAQModel:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

        with open("src/data/faq.json", "r") as file:
            self.faq = json.load(file)

        self.question_list = list(self.faq.keys())
        self.question_embeddings = self.model.encode(
            self.question_list,
            convert_to_tensor=True,
        )

    def get_best_answer(self, query: str):
        logger.info(f'Processing query: {query}')

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.question_embeddings)[0]
        best_score_idx = torch.argmax(cos_scores)

        if cos_scores[best_score_idx] < 0.6:
            return "Извините, я не могу ответить на этот вопрос."

        best_question = self.question_list[best_score_idx]
        best_answer = self.faq[best_question]

        return best_answer


faq_model = FAQModel()

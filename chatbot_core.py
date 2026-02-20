import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import random
import warnings

warnings.filterwarnings("ignore")

# Load JSON responses
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

json_responses = load_json("solmate_response.json")


class EmotionSupportBot:
    def __init__(self):
        self.model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # 🔑 MODEL → JSON EMOTION MAPPING
        self.emotion_map = {
            "joy": "joy",
            "sadness": "sadness",
            "anger": "anger",
            "fear": "fear",
            "love": "love",
            "surprise": "surprise"
        }

    def process(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        confidence = probs.max().item()
        model_emotion = self.model.config.id2label[probs.argmax().item()]

        # Map model emotion to JSON emotion
        final_emotion = self.emotion_map.get(model_emotion, "joy")

        # Pick response from JSON
        response = random.choice(json_responses[final_emotion])

        return final_emotion, confidence, response


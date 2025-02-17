from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline("sentiment-analysis", model="sismetanin/xlm_roberta_large-ru-sentiment-sentirueval2016", from_pt=True)

    def predict(self, text):
        result = self.model(text)[0]
        return result["label"], result["score"]

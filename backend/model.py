from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline(model="tired-racoon/tonika_sentim", trust_remote_code=True)

    def predict(self, text):
        result = self.model(text)[0]
        return result["label"], result["score"]

analyzer = SentimentAnalyzer()

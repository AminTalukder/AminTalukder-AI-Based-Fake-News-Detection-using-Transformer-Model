from flask import Flask, request, render_template
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Set the model path
MODEL_PATH = "E:/Deploy_ML/saved_model"

try:
    model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model/tokenizer: {e}")
    exit()


# Function to predict if news is fake or real
def predict_news(news_text):
    inputs = tokenizer(
        news_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    logits = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    predicted_class = np.argmax(probabilities, axis=1)[0]

    result = {
        "class": "True News" if predicted_class == 1 else "Fake News",
        "confidence": f"{probabilities[0][predicted_class] * 100:.2f}%"
    }
    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_heading = request.form["news_heading"]
        news_summary = request.form["news_summary"]
        combined_text = f"{news_heading} {news_summary}"
        prediction = predict_news(combined_text)

        return render_template(
            "index.html",
            prediction=prediction["class"],
            confidence=prediction["confidence"]
        )


if __name__ == "__main__":
    app.run(debug=True)

#APP.PY
from flask import Flask, render_template, request
import re
from transformers import pipeline
from googletrans import Translator

app = Flask(__name__)

# =========================
# 🔹 TRANSLATOR (NEW)
# =========================
translator = Translator()

def translate_to_english(text):
    try:
        translated = translator.translate(text, dest='en')
        print("🌐 Original:", text)
        print("🌐 Translated:", translated.text)
        return translated.text
    except Exception as e:
        print("❌ Translation Error:", e)
        return text


# =========================
# 🔹 CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    return text.strip()


# =========================
# 🔹 LOAD MODEL
# =========================
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)


# =========================
# 🔹 MAP TO 5 CLASSES
# =========================
def map_to_5_classes(label, score):

    if label == "LABEL_0":
        base = "negative"
    elif label == "LABEL_1":
        base = "neutral"
    elif label == "LABEL_2":
        base = "positive"
    else:
        base = "neutral"

    if base == "negative":
        return "More Negative" if score > 0.75 else "Negative"
    elif base == "neutral":
        return "Neutral"
    elif base == "positive":
        return "More Positive" if score > 0.75 else "Positive"

    return "Neutral"


# =========================
# 🔹 ROUTES
# =========================
@app.route('/')
def index():
    return render_template('home.html')


@app.route('/home')
def home():
    return render_template('index.html')


# =========================
# 🔹 PREDICTION
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    user_input = request.form.get('text', '').strip()

    if not user_input:
        return render_template('result.html',
                               text="",
                               sentiment="None",
                               result="No text provided",
                               suggestion="Please enter some text.",
                               labels=[],
                               values=[])

    # =========================
    # 🌐 TRANSLATION (NEW)
    # =========================
    translated_text = translate_to_english(user_input)

    # =========================
    # 🔥 CRITICAL CHECK
    # =========================
    critical_keywords = [
        "die", "suicide", "kill myself",
        "end my life", "better to die",
        "no reason to live"
    ]

    if any(word in translated_text.lower() for word in critical_keywords):
        final_sentiment = "More Negative"

        labels = ["More Negative", "Negative", "Neutral", "Positive", "More Positive"]
        values = [1, 0, 0, 0, 0]

        result = "🚨 Critical Mental Health Alert"
        suggestion = """
        💡 Immediate Help Needed:<br>
        - Talk to someone you trust<br>
        - Contact a mental health professional<br>
        - Do not stay alone<br>
        - You matter 💙
        """

        return render_template(
            'result.html',
            text=user_input,
            sentiment=final_sentiment,
            result=result,
            suggestion=suggestion,
            labels=labels,
            values=values
        )

    # =========================
    # 🔹 MODEL PREDICTION
    # =========================
    cleaned = clean_text(translated_text)

    pred = sentiment_model(cleaned)[0]
    print("🤖 MODEL OUTPUT:", pred)

    label = pred['label']
    score = float(pred['score'])

    final_sentiment = map_to_5_classes(label, score)

    # =========================
    # 🔹 GRAPH DATA
    # =========================
    labels = ["More Negative", "Negative", "Neutral", "Positive", "More Positive"]
    values = [0, 0, 0, 0, 0]

    if final_sentiment == "More Negative":
        values[0] = score
    elif final_sentiment == "Negative":
        values[1] = score
    elif final_sentiment == "Neutral":
        values[2] = score
    elif final_sentiment == "Positive":
        values[3] = score
    elif final_sentiment == "More Positive":
        values[4] = score

    values = [round(v if v != 0 else (1 - score) / 4, 3) for v in values]

    # =========================
    # 🔹 RESULT + SUGGESTION
    # =========================
    if final_sentiment == "More Negative":
        result = "🚨 Highly Negative Emotion Detected"
        suggestion = """
        💡 Immediate Support Needed:<br>
        - Talk to a friend or family<br>
        - Seek professional help<br>
        - Practice relaxation<br>
        """

    elif final_sentiment == "Negative":
        result = "⚠️ Negative Emotion Detected"
        suggestion = """
        💡 Suggestions:<br>
        - Take breaks<br>
        - Listen to music<br>
        - Do activities you enjoy<br>
        """

    elif final_sentiment == "Neutral":
        result = "😐 Neutral Emotion"
        suggestion = """
        💡 You're stable:<br>
        - Stay consistent<br>
        - Maintain balance<br>
        """

    elif final_sentiment == "Positive":
        result = "😊 Positive Emotion"
        suggestion = """
        ✨ Keep Going:<br>
        - Stay productive<br>
        - Share positivity<br>
        """

    elif final_sentiment == "More Positive":
        result = "🌟 Highly Positive Emotion"
        suggestion = """
        🚀 Excellent State:<br>
        - Inspire others<br>
        - Keep growing<br>
        """

    else:
        result = "Unknown"
        suggestion = "No suggestion available"

    return render_template(
        'result.html',
        text=user_input,
        sentiment=final_sentiment,
        result=result,
        suggestion=suggestion,
        labels=labels,
        values=values
    )


# =========================
# 🔹 MODEL VIEW
# =========================
@app.route('/models_view')
def models_view():
    images = [
        'model.png',
        'compare.png',
        'line.png',
        'output.png',
        'output_non.png'
    ]
    return render_template('models.html', images=images)


# =========================
# 🔹 RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)
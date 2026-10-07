from flask import Flask, render_template, request, jsonify
from knowledge_base import knowledge_base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

questions = list(knowledge_base.keys())
answers = list(knowledge_base.values())

vectorizer = TfidfVectorizer()
knowledge_vectors = vectorizer.fit_transform(questions)


def get_response(question):
    question = question.lower().strip()

    if not question:
        return "Please enter a question."

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        knowledge_vectors
    )[0]

    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    if best_score < 0.15:
        return (
            "Sorry, I don't have information about that yet. "
            "Please ask about CodeAlpha, its tasks, Flask, AWS, "
            "Python, GitHub, or cloud computing."
        )

    return answers[best_match_index]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "")

    response = get_response(question)

    return jsonify({
        "response": response
    })


if __name__ == "__main__":
    app.run(debug=True)
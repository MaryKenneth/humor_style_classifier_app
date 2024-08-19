#import flask module
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer


# instance of flask application 
app = Flask(__name__)

# Load the model
with open('ali_xgboost_humour_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Define label mappings
label_map = {
    0: "Self-enhancing",
    1: "Self-deprecating",
    2: "Affiliative",
    3: "Aggressive",
    4: "Neutral"
}

# Label Explanation Mapping 
explanation_map = {
    0: "This style involves making jokes that make yourself feel good or help you cope with difficult situations. It's often lighthearted and positive, focusing on finding humor in life without putting others down.",
    1: "This style involves making jokes at your own expense. While it can be funny, it often highlights your own flaws or shortcomings.",
    2: "This style is all about sharing jokes that bring people together and make everyone laugh. It's friendly, inclusive, and helps build connections with others.",
    3: "This style includes jokes that might hurt or put others down, even if it's meant to be funny. It can come across as mean or sarcastic.",
    4: "This style is just general humor that doesn't strongly fall into any of the other categories. It's neither overly positive nor negative, and it's usually inoffensive and mild."
}

# Label Encouragement
encouragement_map = {
    0: "Great job! This kind of humor helps you stay positive and enjoy life more.",
    1: "It's okay to laugh at yourself sometimes, but be kind to yourself. Too much self-deprecating humor can hurt your self-esteem.",
    2: "Awesome! This kind of humor brings people closer and spreads positivity.",
    3: "Try to avoid humor that could hurt others. Positive humor makes everyone feel good.",
    4: "Nice! Neutral humor is safe and can be enjoyed by most people."
}

# Load the embedding model once globally
embedding_model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

def ALI(sentences):
    dataset_embedding = embedding_model.encode(sentences, normalize_embeddings=True)
    return dataset_embedding

def single_predict(example):
    # Embedding 
    embedding = ALI(example)
    embedding = np.expand_dims(embedding, axis=0)

    # Predict probabilities using the model
    proba = model_data.predict_proba(embedding)
    
    # Get the index of the highest probability
    pred_index = np.argmax(proba)
    
    # Return the predicted index and the probabilities
    return pred_index, proba[0]

    # Predict using the model
    #pred = model_data.predict(embedding)
    # Since `pred` is an array, extract the first (and likely only) element
    #pred = int(pred[0])  # Convert NumPy array to a Python integer
    #return pred

# home route that returns below text when root url is accessed
@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/classify_joke", methods=["POST"])
def classify_joke():
    data = request.json
    joke = data['joke']

    # Make predictions
    #prediction = single_predict(joke)
    prediction, proba = single_predict(joke)

    # Get the prediction probability for the predicted class
    probability = proba[prediction]  # Probability for the predicted class

 
    # Map numeric predictions to string labels
    label = label_map.get(prediction, "Unknown")
    explanation = explanation_map.get(prediction, "Unknown")
    encouragement = encouragement_map.get(prediction, "Unknown")

    
    return jsonify({
        'style': f"{label}",
        'explain': f"{explanation}",
        'encourage': f"{encouragement}",
        'probability': f"Model Confidence: {probability:.2f}"
    })


if __name__ == "__main__":
    app.run(debug=True)
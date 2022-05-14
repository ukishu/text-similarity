import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template


def cos_sim(sentence1_emb, sentence2_emb):
    """
    Cosine similarity between two columns of sentence embeddings
    
    Args:
      sentence1_emb: sentence1 embedding column
      sentence2_emb: sentence2 embedding column
    
    Returns:
      The row-wise cosine similarity between the two columns.
      For instance if sentence1_emb=[a,b,c] and sentence2_emb=[x,y,z]
      Then the result is [cosine_similarity(a,x), cosine_similarity(b,y), cosine_similarity(c,z)]
    """
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)

app = Flask(__name__)

@app.route('/',methods=['POST'])
def predict():
    if request.method == 'POST':
        # inputs
        text1 = request.form['text1']
        text2 = request.form['text2']

        # Load the pre-trained model
        model = SentenceTransformer('stsb-mpnet-base-v2')

        # Generate Embeddings
        text1_emb = model.encode(text1, show_progress_bar=True)
        text2_emb = model.encode(text2, show_progress_bar=True)

        # Cosine Similarity
        cosine_score = np.dot(text1_emb,text2_emb)/(np.linalg.norm(text1_emb)*np.linalg.norm(text2_emb))

        # Scaling Values to 0 and 1
        sim_score=2/(1+np.exp(2*(1-cosine_score)))
        output = round(sim_score, 3)

        return render_template('index.html', prediction_text='similarity score: {}'.format(output))

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
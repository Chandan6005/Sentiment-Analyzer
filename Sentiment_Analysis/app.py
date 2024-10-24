from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model and the TF-IDF vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        prediction = model.predict(vect)
        output = 'Positive' if prediction[0] == 1 else 'Negative'
        return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
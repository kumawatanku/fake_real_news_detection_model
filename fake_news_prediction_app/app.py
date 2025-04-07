from flask import Flask,render_template,request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')    
nltk.download('omw-1.4')

with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

def preprocess_text(text):
    text=text.lower()
    text=text.translate(str.maketrans('', '',string.punctuation))
    tokens=text.split()
    tokens=[word for word in tokens if word not in stopwords.words('english')]
    lemmatizer=WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_text=request.form['news']
    cleaned=preprocess_text(input_text)
    vectorized=vectorizer.transform([cleaned])
    prediction=model.predict(vectorized)[0]
    result="🟢 Real News" if prediction==1 else "🔴 Fake News"
    return render_template("index.html",prediction=result,original=input_text)

if __name__=='__main__':
    app.run(debug=True)
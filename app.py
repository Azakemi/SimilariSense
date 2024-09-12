from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics import pairwise
import pickle

app = Flask(__name__)

# Load the model and vectorizer at the start
with open('product_classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load product data
file_path2 = 'Tata_Steel_datatrain.csv'
data2 = pd.read_csv(file_path2)

product_names = data2['Product_name'].tolist()
product_categories = data2['category'].tolist()
product_subcategories = data2['sub_category'].tolist()
product_features = vectorizer.transform(product_names)


def find_similar_items(query, product_features, product_names, product_categories, product_subcategories):
    query_features = vectorizer.transform([query])
    similarities = pairwise.cosine_similarity(query_features, product_features).flatten()
    similar_indices = similarities.argsort()[-4:][::-1]
    top_similar_items = [(product_names[i], product_categories[i], product_subcategories[i], similarities[i]) for i in
                         similar_indices]
    return top_similar_items


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    similar_items = find_similar_items(query, product_features, product_names, product_categories,
                                       product_subcategories)

    results = []
    for item, category, subcategory, score in similar_items:
        results.append({
            'product_name': item,
            'category': category,
            'subcategory': subcategory,
            'similarity': f"{score * 100:.2f}%"
        })

    return render_template('index.html', query=query, results=results)


if __name__ == '__main__':
    app.run(debug=True)




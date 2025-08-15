from flask import Flask, request, render_template
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load data and train model once
df = pd.read_csv("customers.csv")
X = df[["AnnualIncome", "SpendingScore"]]
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

@app.route('/', methods=['GET', 'POST'])
def home():
    cluster = None
    if request.method == 'POST':
        income = float(request.form['income'])
        score = float(request.form['score'])
        cluster = kmeans.predict([[income, score]])[0]
    return render_template('index.html', cluster=cluster)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('salary_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    data = {}
    if request.method == 'POST':
        data = {
            'age': int(request.form['age']),
            'workclass': request.form['workclass'],
            'education': request.form['education'],
            'education_num': int(request.form['education_num']),
            'marital_status': request.form['marital_status'],
            'occupation': request.form['occupation'],
            'relationship': request.form['relationship'],
            'race': request.form['race'],
            'sex': request.form['sex'],
            'hours_per_week': int(request.form['hours_per_week']),
            'native_country': request.form['native_country']
        }
        input_df = pd.DataFrame([data])
        for col in input_df.select_dtypes(include='object').columns:
            input_df[col] = pd.factorize(input_df[col])[0]
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        pred = model.predict(input_df)[0]
        prediction = 'Over 50K' if pred == 1 else '50K or below'
    return render_template('index.html', prediction=prediction, data=data)

if __name__ == '__main__':
    app.run(debug=True)

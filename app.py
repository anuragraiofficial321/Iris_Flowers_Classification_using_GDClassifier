from flask import Flask, jsonify, request, render_template
import numpy as np
from joblib import load

# Load the model from the file
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    lr = load('model.pkl')
    to_predict_list = request.form.to_dict()
    
    # Form validation to check for empty fields
    errors = []
    for key, value in to_predict_list.items():
        if not value:
            errors.append(f"Please enter {key.replace('_', ' ').title()}.")

    if errors:
        return render_template('index.html', errors=errors)
    
    # Convert values to float and make prediction
    to_predict_list = list(map(float, to_predict_list.values()))
    to_predict_list = np.array(to_predict_list).reshape(1, -1)
    print(to_predict_list)
    
    prediction = lr.predict(to_predict_list)
    prediction_result = f"{list(prediction)[0]}"
    return render_template('prediction.html', prediction=prediction_result)



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

from flask import Flask, jsonify, request
from sklearn.datasets import load_iris
import pickle
app = Flask(__name__)
import json

iris = load_iris()


@app.route('/test_getip')
def test_get_ip():
    ip_address = request.remote_addr
    return jsonify({'ip': ip_address})

@app.route('/test_predictlogistic', methods=['POST'])
def test_predictlogistic():
    data = json.loads(request.data)
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    X_new = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    y_pred = model.predict([X_new])
    print(y_pred)

    return jsonify({'prediction': iris.target_names[y_pred[0]]})

@app.route('/test_predictcnn', methods=['POST'])
def test_predictcnn():
    image = request.json['image']
    # Load the saved model from file
    with open('cnn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(image).argmax()
    return jsonify({'prediction': int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)

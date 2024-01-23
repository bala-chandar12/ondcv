from flask import Flask, jsonify,request
from flask_cors import CORS
from main import predic

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/data')
def get_data():
    data = {'message': 'This is sample data'}
    return jsonify(data)
@app.route('/data', methods=['POST'])
def post_data():
    req_data = request.get_json()
    print(req_data["url"])
    res=predic(req_data["url"])

    # Assuming the request sends a JSON object
    # Do something with the received data
    received_data = {'received_data': res}
    return jsonify(received_data)

if __name__ == '__main__':
    app.run()

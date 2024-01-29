from flask import Flask, jsonify,request
from flask_cors import CORS
from main import predic
import requests
import time

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
def download(url):
    response = requests.get(url)
    with open("C:/Users/nagar/PycharmProjects/ondc/content/local_image.jpg", "wb") as f:
        f.write(response.content)


@app.route('/data')
def get_data():
    data = {'message': 'This is sample data'}
    return jsonify(data)
@app.route('/data', methods=['POST'])
def post_data():
    req_data = request.get_json()
    print(req_data["url"])
    download(req_data["url"])
    time.sleep(2)


    res=predic("C:/Users/nagar/PycharmProjects/ondc/content/local_image.jpg")
   


    # Assuming the request sends a JSON object
    # Do something with the received data
    received_data = {'received_data': res}
    return jsonify(received_data)

if __name__ == '__main__':
    app.run()

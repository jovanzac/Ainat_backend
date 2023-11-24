from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route("/real_time", methods=["POST"])
def real_time_data() :
    if request.method == "POST" :
        response = request.json
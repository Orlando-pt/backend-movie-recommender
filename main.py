from flask import Flask, request
from flask_cors import CORS, cross_origin

from colab_filtering import CF

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
cf_module = CF()

@app.route("/")
@cross_origin()
def hello():
    return "Hello, World!"

@app.route("/movies")
@cross_origin()
def get_movies():
    response = app.response_class(
        response=cf_module.get_movies,
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/recommendation", methods=['POST'])
@cross_origin()
def get_recommendations():
    print(request.json)
    if request.method == 'POST':
        cf_module.add_user_ratings(
            request.json['movies'],
            request.json['ratings']
        )

        cf_module.build_model()
        cf_module.train_model()

        recommendations = cf_module.get_user_recommendations('cosine')

        response = app.response_class(
            response=recommendations,
            status=200,
            mimetype='application/json'
        )
        return response

@app.errorhandler(500)
@cross_origin()
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)

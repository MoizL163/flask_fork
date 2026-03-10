from flask import Blueprint, jsonify, request
from flask_restful import Api, Resource

from model.titanic import TitanicModel


titanic_api = Blueprint("titanic_api", __name__, url_prefix="/api/titanic")
api = Api(titanic_api)


class TitanicAPI:
    class _Predict(Resource):
        def post(self):
            passenger = request.get_json(silent=True) or {}
            try:
                model = TitanicModel.get_instance()
                response = model.predict(passenger)
                return jsonify(response)
            except Exception as err:
                return jsonify({"error": f"Prediction failed: {str(err)}"}), 500

        def get(self):
            return jsonify(
                {
                    "message": "Titanic endpoint ready. Send POST JSON to this route.",
                    "required_keys": [
                        "name",
                        "pclass",
                        "sex",
                        "age",
                        "sibsp",
                        "parch",
                        "fare",
                        "embarked",
                        "alone",
                    ],
                }
            )


api.add_resource(TitanicAPI._Predict, "/predict")

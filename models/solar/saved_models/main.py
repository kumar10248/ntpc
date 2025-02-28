from flask import Flask, request, jsonify
from datetime import datetime
import time
import requests
import pickle 

app = Flask(__name__)


forcasted_generated = 0

api_url = "https://api.open-meteo.com/v1/forecast?latitude=20.5937&longitude=78.9629&hourly=temperature_2m,wind_speed_10m,direct_radiation,shortwave_radiation_instant&daily=daylight_duration&timezone=auto&forecast_days=14"

@app.route('/retrain',methods=["POST"])
def retrain():
    data = request.get_json()
    return jsonify({"received": data})


@app.route('/predict',methods=["POST"])
def predict():
    """ It will return the power generation for next 14 days by taking the data from the api and predicting it from the model"""
    data = request.get_json()
    latitude = data["latitude"]
    longitude = data["longitude"]
    print(latitude,longitude)
    if(not latitude or not longitude):
        return jsonify({"Message":"No Data found"})
    print("Being Called")
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,wind_speed_10m,direct_radiation,shortwave_radiation_instant&daily=daylight_duration&timeformat=unixtime&timezone=auto&forecast_days=14")
    
    print(response.json())
    global forcasted_generated
    model = pickle.load(open('model.pkl', 'rb')) 
    res = model.predict(response.json())


    # response.json() - send this data to the pickel file and then store the record
    return jsonify(res.tolist())


if __name__ == '__main__':
    app.run()

    # while(True):
    #     now = datetime.now()
    #     automaticallyCall()
    #     time.sleep(36)

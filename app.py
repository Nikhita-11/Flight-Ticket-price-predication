from flask import Flask, request, render_template, jsonify
# Alternatively can use Django, FastAPI, or anything similar
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__, static_folder='templates')
app = application

@app.route('/', methods = ['POST', "GET"])

def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("form.html")
    else: 
        data = CustomData(
            airline = request.form.get('airline'),
            source_city = request.form.get("source_city"), 
            destination_city = request.form.get("destination_city"), 
            departure_time = request.form.get("departure_time"),
            arrival_time = request.form.get("arrival_time"), 
            stops = request.form.get("stops"), 
            Class = request.form.get("class"),
            duration = float(request.form.get("duration")),
            days_left = int(request.form.get("days_left"))
        ) 
    new_data = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(new_data)

    results = round(pred[0],2)

    return render_template("form.html", final_result = results)

if __name__ == "__main__": 
    app.run(host = "0.0.0.0", debug= True)

#http://127.0.0.1:5000/ in browser
from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialise the Flask app
app = Flask(__name__)

# Use pickle to load in the pre-trained model
filename = "models/model.sav"
model = pickle.load(open(filename, "rb"))


# Set up the main route
@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        # Extract the input from the form
        temperature = request.form.get("temperature")
        humidity = request.form.get("humidity")
        windspeed = request.form.get("windspeed")

        # Create DataFrame based on input
        input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                       columns=['temperature', 'humidity', 'windspeed'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        # Given that the prediction is stored in an array we simply extract by indexing
        prediction = model.predict(input_variables)[0]

        # We now pass on the input from the from and the prediction to the index page
        return render_template("index.html",
                               original_input={'Temperature': temperature,
                                               'Humidity': humidity,
                                               'Windspeed': windspeed},
                               result=prediction
                               )
    # If the request method is GET
    return render_template("index.html")
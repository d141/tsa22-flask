from flask import Flask, render_template, request, Response
import pickle
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Initialise the Flask app
app = Flask(__name__)
app.debug = True

# Use pickle to load in the pre-trained model
filename = "models/model.sav"
model = pickle.load(open(filename, "rb"))

#@app.route('/draw_plot/', methods=['GET', 'POST'])
def draw_plot(df):
    fig = create_figure(df)
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return pngImageB64String

def create_figure(df):
    fig, ax = plt.subplots(figsize = (6,4))
    fig.patch.set_facecolor('#E8E5DA')
    x = df.index
    y = df.mean_va
    ax.plot(x, y, color = "#304C89")
    plt.xticks(rotation = 30, size = 5)
    plt.ylabel("Water Level", size = 5)
    return fig


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        # Extract the input from the form
        site = request.form.get("site")
        period = request.form.get("period")
        input_to_filename = {"black_rock": "data/black_rock_reservoir"}
        data = pd.read_csv(input_to_filename[site], sep="\t")
        data['date'] = data['year_nu'].astype(str) + ' ' + data['month_nu'].astype(str) + ' 1'
        data.set_index('date', inplace=True)
        data = data.iloc[1:, :]
        data.index = pd.to_datetime(data.index, format='%Y %m %d')  # drop null values
        data.dropna(inplace=True)
        df = data.drop(columns=['agency_cd', 'site_no', 'parameter_cd', 'ts_id', 'year_nu', 'month_nu'])
        df['mean_va'] = df['mean_va'].astype(float)
        # Create DataFrame based on input
        # input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
        #                                columns=['temperature', 'humidity', 'windspeed'],
        #                                dtype=float,
        #                                index=['input'])

        # Get the model's prediction
        # Given that the prediction is stored in an array we simply extract by indexing
        prediction = float(df.sample(1).mean_va)
        image = draw_plot(df)
        # We now pass on the input from the from and the prediction to the index page
        return render_template("index.html",
                               original_input={'Site': site,
                                               'Months Forecasted': period},
                               result=prediction,
                               image=image
                               )
    # If the request method is GET
    return render_template('index.html')

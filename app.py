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
import pandas as pd
import statsmodels.api as sm
from fbprophet import Prophet

# Initialise the Flask app
app = Flask(__name__)
app.debug = True

# Use pickle to load in the pre-trained model
filename = "models/model.sav"
model = pickle.load(open(filename, "rb"))


# @app.route('/draw_plot/', methods=['GET', 'POST'])
def draw_plot_current(df, site):
    fig = create_figure_current(df, site)
    # Convert plot to PNG image
    png_image = io.BytesIO()
    FigureCanvas(fig).print_png(png_image)

    # Encode PNG image to base64 string
    png_image_b64_string = "data:image/png;base64,"
    png_image_b64_string += base64.b64encode(png_image.getvalue()).decode('utf8')

    return png_image_b64_string


def create_figure_current(df, site):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#E8E5DA')
    x = df.index
    y = df.value
    ax.plot(x, y, color="#304C89")
    plt.title(f"Volume of water in acre-feet at {site}")
    plt.xticks(rotation=30, size=10)
    plt.ylabel("Acre-feet", size=10)
    return fig

def draw_plot_decomp(df, site):
    fig = create_figure_decomp(df, site)
    # Convert plot to PNG image
    png_image = io.BytesIO()
    FigureCanvas(fig).print_png(png_image)

    # Encode PNG image to base64 string
    png_image_b64_string = "data:image/png;base64,"
    png_image_b64_string += base64.b64encode(png_image.getvalue()).decode('utf8')

    return png_image_b64_string

def create_figure_decomp(df, site):
    result = sm.tsa.seasonal_decompose(df['value'], model='additive')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    plt.title(f"Volume of water in acre-feet at {site}")
    fig.patch.set_facecolor('#E8E5DA')
    fig.suptitle(f"Decomposition plots for {site}")
    result.trend.plot(ax=ax1, title="Trend", color="#304C89").xaxis.label.set_visible(False)
    result.resid.plot(ax=ax2, title="Residuals", color="#304C89").xaxis.label.set_visible(False)
    result.seasonal.plot(ax=ax3, title="Seasonal Component", color="#304C89").xaxis.label.set_visible(False)
    return fig

def draw_plot_pred(data, site):
    fig = create_figure_pred(data, site)
    # Convert plot to PNG image
    png_image = io.BytesIO()
    FigureCanvas(fig).print_png(png_image)

    # Encode PNG image to base64 string
    png_image_b64_string = "data:image/png;base64,"
    png_image_b64_string += base64.b64encode(png_image.getvalue()).decode('utf8')

    return png_image_b64_string

def create_figure_pred(data, site):
    df = pd.DataFrame()
    df['ds'] = data['dateTime'].dt.tz_localize(None)
    df['y'] = data['value']
    model = Prophet()
    model.fit(df)
    forecast = model.make_future_dataframe(periods=36, freq='MS')
    forecast = model.predict(forecast)
    plt.figure(figsize=(18, 6))
    model.plot(forecast, xlabel='Date', ylabel='Mean Water Level')
    plt.title('FB Prophet - Black Rock Reservoir Water Level')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    plt.title(f"Volume of water in acre-feet at {site}")
    fig.patch.set_facecolor('#E8E5DA')
    fig.suptitle(f"Decomposition plots for {site}")
    result.trend.plot(ax=ax1, title="Trend", color="#304C89").xaxis.label.set_visible(False)
    result.resid.plot(ax=ax2, title="Residuals", color="#304C89").xaxis.label.set_visible(False)
    result.seasonal.plot(ax=ax3, title="Seasonal Component", color="#304C89").xaxis.label.set_visible(False)
    return fig

def adfuller_test(sales):
    result=sm.tsa.stattools.adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    summary = []
    for value,label in zip(result,labels):
        summary.append(f"{label} : {str(value)}")
    if result[1] <= 0.05:
        summary.append("P value is less than 0.05 that means we can reject the null hypothesis(Ho). Therefore we can conclude that data has no unit root and is stationary \n")
    else:
        summary.append("Weak evidence against null hypothesis that means time series has a unit root which indicates that it is non-stationary \n")
    return summary


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        # Extract the input from the form
        site = request.form.get("site")
        period = request.form.get("period")
        input_to_filename = {'Lake Havasu': 'LAKE HAVASU NEAR PARKER DAM, AZ-CA',
                             'Lake Tahoe': 'LAKE TAHOE A TAHOE CITY CA',
                             'Tulloch Reservoir': 'TULLOCH RES NR KNIGHTS FERRY CA',
                             'El Capitan Reservoir': 'EL CAPITAN RES NR LAKESIDE CA',
                             'Independence Lake': 'INDEPENDENCE LK NR TRUCKEE CA',
                             'San Antonio Reservoir': 'SAN ANTONIO RESERVOIR NR SUNOL CA',
                             'San Vicente Reservoir': 'SAN VICENTE RES NR LAKESIDE CA',
                             'Salton Sea': 'SALTON SEA NR WESTMORLAND CA'}
        data = pd.read_parquet('data/lakes_in_ca.parquet')
        data = data[data['siteName'] == input_to_filename[site]]
        data = data.sort_values(by="dateTime")
        data = data.set_index(data.dateTime)
        #data = data.drop(columns=['dateTime', 'siteName'])
        data['value'] = data['value'].fillna(value=None, method='backfill', axis=None, limit=None, downcast=None)
        # Get the model's prediction
        # Given that the prediction is stored in an array we simply extract by indexing
        latest_reading = data.tail(1).value.values[0]
        stationarity_summary = adfuller_test(data['value'])
        current_image = draw_plot_current(data, site)
        decomp_image = draw_plot_decomp(data, site)
        # We now pass on the input from the from and the prediction to the index page
        return render_template("index.html",
                               original_input={'Site': site,
                                               'Months Forecasted': period},
                               stationarity_summary=stationarity_summary,
                               latest_reading=latest_reading,
                               current_image=current_image,
                               decomp_image=decomp_image
                               )
    # If the request method is GET
    return render_template('index.html')

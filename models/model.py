import pandas as pd
from sklearn import svm
import pickle

# Read the data
df = pd.read_csv('https://raw.githubusercontent.com/sg-tarek/HTML-CSS-and-JavaScript/main/Model%20Deployment/data.csv', index_col='Unnamed: 0')

# Fit the data to an ML model
x = df[['temperature', 'humidity', 'windspeed']]
y = df['count']

regr = svm.SVR()
regr.fit(x, y)

# Save the model
filename = 'model.sav'
pickle.dump(regr, open(filename, 'wb'))
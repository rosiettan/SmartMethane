from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('methane_detection_model2.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    country = request.form.get('Country')
    sector = request.form.get('Sector')
    year = int(request.form.get('Year'))
    print(country)
    print(sector)
    print(year)

    data = pd.read_csv("Dataset.csv")
    mean_value = data.mean()
    data.fillna(value=mean_value, inplace=True)

    def filter_data(data, country, sector):
        # Filter the dataset based on country and sector
        filtered_data = data[(data['Country'] == country) & (data['Sector'] == sector) & (data['Year'] <= year)]
        # Drop the target variable column
        filtered_data = filtered_data.drop(columns=['Value'])
        # Get the indices of the filtered rows
        indices = filtered_data.index.tolist()
        return indices, filtered_data

    ind, df = filter_data(data, country, sector)
    period = round(df["Year"].nunique() / 12, 1)
    print(period)
    X = pd.get_dummies(data, columns=["Country", "Sector"])
    X = X.iloc[:, :205]
    p = model.predict(X)

    prediction = list()
    for i in ind:
        prediction.append(p[i])

    print(prediction)

    # Calculate the concentration and classification
    concentration = np.mean(prediction) / 1000
    if concentration < 0.0003:
        return render_template('index.html',
                               concentration='The classification is NORMAL. \n The predicted methane concentration is '
                                             'about = {} for period = {} years\n'.format(abs(concentration), period),
                               bhai="Your prediction")

    elif 0.0003 <= concentration <= 0.0005:
        return render_template('index.html',
                               concentration='The classification is MEDIUM. \n The predicted methane concentration is '
                                             'about = {} for period = {} years\n'.format(abs(concentration), period),
                               bhai="Your prediction")

    elif concentration > 0.0005:
        return render_template('index.html',
                               concentration='The classification is DANGEROUS. \n The predicted methane concentration '
                                             'is about = {} for period = {} years\n'.format(abs(concentration), period),
                               bhai="Your prediction")


if __name__ == '__main__':
    app.run(debug=True)

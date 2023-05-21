"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""
# Test
# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import OrdinalEncoder

def _preprocess_data(data):

    enc = OrdinalEncoder()

    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)

    # Load the dictionary as a Pandas DataFrame.
    test_df = pd.DataFrame.from_dict([feature_vector_dict])
    test_df['Valencia_pressure'] = test_df['Valencia_pressure'].fillna(test_df['Valencia_pressure'].mean())
    test_df.Valencia_wind_deg = enc.fit_transform(test_df[['Valencia_wind_deg']])
    test_df.Seville_pressure = enc.fit_transform(test_df[['Seville_pressure']])

    #Convert time column to datetime
    test_df['time'] = pd.to_datetime(test_df['time'])

    # Extract date and time components
    test_df['Day'] = test_df['time'].dt.day
    test_df['Month'] = test_df['time'].dt.month   
    test_df['Year'] = test_df['time'].dt.year
    test_df['Start_hour'] = test_df['time'].dt.hour

    # Make a copy of the time column for later use
    test_time = test_df['time'].copy()

    # Drop unwanted columns from both train_df and test_df
    columns_to_drop = ['Unnamed: 0','Bilbao_weather_id','Madrid_weather_id','Barcelona_weather_id','Seville_weather_id', 'Valencia_temp_max','Seville_temp_max', 'Valencia_temp_min', 'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp_min']
    test_df.drop(columns_to_drop, axis=1, inplace=True)

    # Drop the time column from train_df
    columns_to_replace = ['Seville_rain_3h', 'Madrid_rain_1h', 'Barcelona_rain_3h', 'Valencia_snow_3h', 'Seville_rain_1h', 'Bilbao_snow_3h', 'Barcelona_rain_1h', 'Madrid_clouds_all', 'Seville_clouds_all', 'Bilbao_clouds_all', 'Bilbao_rain_1h']

    for column in columns_to_replace:
        mean_value = test_df[column].mean()
        test_df[column] = test_df[column].replace(0, mean_value)   

    predict_vector = test_df[['Madrid_wind_speed','Madrid_humidity','Madrid_clouds_all','Madrid_temp','Madrid_pressure']]
    
    return predict_vector


def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()

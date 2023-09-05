import pickle
import pandas as pd
import json


def predict_price(config):
    ##loading the model from the saved file
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    return model.predict(df)
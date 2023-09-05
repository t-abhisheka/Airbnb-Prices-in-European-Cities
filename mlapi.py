from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi import HTTPException

app = FastAPI()

class ScoreingItem(BaseModel):
    room_shared : int
    person_capacity : float
    host_is_superhost : int
    multi : int
    biz : int
    cleanliness_rating : float
    guest_satisfaction_overall : float
    bedrooms : int
    dist : float
    metro_dist : float
    attr_index : float
    attr_index_norm : float
    rest_index : float
    rest_index_norm : float
    lng : float
    lat : float
    room_type_private_room : int
    room_type_shared_room  : int
    city_athens  : int
    city_barcelona  : int
    city_berlin  : int
    city_budapest  : int
    city_lisbon  : int
    city_london  : int
    city_paris  : int
    city_rome  : int
    city_vienna  : int
    time_weekends  : int

pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as f:
        model = pickle.load(f)


@app.post('/')

async def scoring_endpoint(item: ScoreingItem):
    try:
        df = pd.DataFrame([item.dict()])
        yhat = model.predict(df)
        return {"prediction": yhat.tolist()}
    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred: {str(e)}")
        # Raise an HTTP exception with a 500 status code
        raise HTTPException(status_code=500, detail="Internal Server Error")
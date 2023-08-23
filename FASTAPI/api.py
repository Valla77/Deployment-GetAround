import pandas as pd
import os
import json
import mlflow
import uvicorn
import gc
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse


description = """
Welcome to my rental price predictor API !\n
Share your car's attributes, and an adept Machine Learning model, developed using GetAround data, will suggest a daily rental rate based on the information you provide. 

**Use the endpoint `/predict` to estimate the daily rental price of your car !**
"""

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Use this endpoint for getting predictions"
    }
]

app = FastAPI(
    title="💸 Getaround API : Car Rental Price Predictor",
    description=description,
    version="0.1",
    contact={
        "name": "getaround",
        "url": "https://fr.getaround.com/?utm_medium=sem&utm_source=google&utm_campaign=6622016394&utm_content=79706791860&gclid=Cj0KCQjwuZGnBhD1ARIsACxbAVh14FuCiyY15b97W_BJLeekh7dHUzw98ZBU5KSX20Xn_OcFOXMg2M4aAmkCEALw_wcB"},
    openapi_tags=tags_metadata
)

class Car(BaseModel):
    model_key: Literal['Citroën','Peugeot','PGO','Renault','Audi','BMW','Mercedes','Opel','Volkswagen','Ferrari','Mitsubishi','Nissan','SEAT','Subaru','Toyota','other'] 
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: Literal['diesel','petrol','other']
    paint_color: Literal['black','grey','white','red','silver','blue','beige','brown','other']
    car_type: Literal['convertible','coupe','estate','hatchback','sedan','subcompact','suv','van']
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

    # Redirect automatically to /docs (without showing this endpoint in /docs)
@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')

# Make predictions
@app.post("/predict", tags=["Predictions"])
async def predict(cars: List[Car]):
    # clean unused memory
    gc.collect(generation=2)

    # Read input data
    car_features = pd.DataFrame(jsonable_encoder(cars))

    # Log model from mlflow
    logged_model = 'runs:/99e02d56d67048c3b50599aa24da22c6/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict and format response
    prediction = loaded_model.predict(car_features)
    response = {"prediction": prediction.tolist()}
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, debug=True, reload=True) 

    # Redirect automatically to /docs (without showing this endpoint in /docs)
# @app.get("/", include_in_schema=False)
# async def docs_redirect():
#     return RedirectResponse(url='/docs')

# # Make predictions
# @app.post("/predict", tags=["Predictions"])
# async def predict(cars: List[Car]):
#     # clean unused memory
#     gc.collect(generation=2)

#     # Read input data
#     car_features = pd.DataFrame(jsonable_encoder(cars))

#     # Load model as a PyFuncModel.
#     logged_model = 'runs:/e7a29c0aa6db4302b9a75bb16193ae82/model'
#     loaded_model = mlflow.pyfunc.load_model(logged_model)

#     # Predict and format response
#     prediction = loaded_model.predict(car_features)
#     response = {"prediction": prediction.tolist()}
#     return response

# if __name__=="__main__":
#     uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)

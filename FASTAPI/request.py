import requests

response = requests.post("https://getaround-fastapi-val-189215b0cd0a.herokuapp.com/", json=
  {
  "model_key": "Citroën",
  "mileage": 13929,
  "engine_power": 317,
  "fuel": "petrol",
  "paint_color": "grey",
  "car_type": "convertible",
  "private_parking_available": True,
  "has_gps": True,
  "has_air_conditioning": False,
  "automatic_car": False,
  "has_getaround_connect": False,
  "has_speed_regulator": True,
  "winter_tires": True
  })
print(response.json())
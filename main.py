from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

app = FastAPI()

model_path = hf_hub_download(
    repo_id="LesibaG/house-price-random-forest",
    filename="house_price_model.pkl"
)

model = joblib.load(model_path)

# Define the house information required
class HouseInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Homepage
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict_price(house: HouseInput):

    # Convert user input to a Pandas DataFrame
    house_data = pd.DataFrame([house.model_dump()])

    # Generate prediction
    prediction = model.predict(house_data)

    # Convert to dollars and round
    price = round(float(prediction[0] * 100000), 2)

    # Return prediction as JSON
    return {"predicted_price": price}

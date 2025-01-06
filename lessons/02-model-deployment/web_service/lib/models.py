from pydantic import BaseModel


class InputData(BaseModel):
    PULocationID: int = 264
    DOLocationID: int = 264
    passenger_count: int = 1


class PredictionOut(BaseModel):
    trip_duration_prediction: float

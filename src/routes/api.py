from fastapi import APIRouter
from typing import List, Dict
from src.controllers.ExtractorController import ExtractorController
from src.controllers.PredictionController import PredictionController
from src.models.schemas import TextRequest, BatchTextRequest

# Initialize router
router = APIRouter(prefix="/prediction", tags=["Prediction"])

# Initialize controllers
extractor_controller = ExtractorController()
prediction_controller = PredictionController()

# Routes
@router.post("/from-text", response_model=Dict)
async def predict_from_text(
    request: TextRequest
):
    """
    Extract features from text and predict churn.
    """
    # Extract features from text
    customer_data = extractor_controller.extract_features(request.text)
    
    # Make prediction
    prediction = prediction_controller.predict_new(customer_data)
    
    return prediction

@router.post("/from-text-with-probability", response_model=Dict)
async def predict_from_text_with_probability(
    request: TextRequest
):
    """
    Extract features from text and predict churn with probability.
    """
    # Extract features from text
    customer_data = extractor_controller.extract_features(request.text)
    
    # Make prediction with probability
    prediction = prediction_controller.predict_with_probability(customer_data)
    
    return prediction

@router.post("/batch", response_model=List[Dict])
async def predict_batch(
    request: BatchTextRequest,
):
    """
    Process a batch of customer descriptions and predict churn for each.
    """
    # Extract features for each text
    customer_data_list = []
    for text in request.texts:
        customer_data = extractor_controller.extract_features(text)
        customer_data_list.append(customer_data)
    
    # Make batch prediction
    predictions = prediction_controller.predict_batch(customer_data_list)
    
    return predictions
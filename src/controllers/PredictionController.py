import numpy as np
import pandas as pd
import logging
from typing import Dict, List
from fastapi import HTTPException
from src.models.schemas import CustomerData
from src.helpers.config import settings

# Configure logger
logger = logging.getLogger(__name__)

class PredictionController:
    """
    Controller class for making churn predictions using the trained model.
    """
    
    def __init__(self):
        """Initialize the PredictionController."""
        self.pipe = settings.pipe
        self.classifier = settings.classifier
        self.columns = settings.columns
        self.dtypes = settings.dtypes
    
    def _prepare_input_data(self, data: CustomerData) -> pd.DataFrame:
        """
        Prepare input data for prediction.
        """
        try:
            # Concatenate all features from pydantic model
            input_data = np.array([
                data.CreditScore, 
                data.Geography, 
                data.Gender, 
                data.Age, 
                data.Tenure, 
                data.Balance, 
                data.NumOfProducts, 
                data.HasCrCard, 
                data.IsActiveMember, 
                data.EstimatedSalary
            ])
            
            # Convert to DataFrame with correct data types
            X_new = pd.DataFrame([input_data], columns=self.columns)
            X_new = X_new.astype(self.dtypes)
            
            logger.debug(f"Prepared input data for prediction: {X_new.shape}")
            return X_new
            
        except Exception as e:
            logger.error(f"Error preparing input data: {str(e)}")
            raise
    
    def _format_response(self, data: CustomerData, prediction: int, probability: float = None) -> Dict:
        """
        Format prediction response.
        """
        response = {
            'Age': data.Age,
            'CreditScore': data.CreditScore,
            'Geography': data.Geography,
            'Gender': data.Gender,
            'Tenure': data.Tenure,
            'Balance': data.Balance,
            'NumOfProducts': data.NumOfProducts,
            'HasCrCard': data.HasCrCard,
            'IsActiveMember': data.IsActiveMember,
            'EstimatedSalary': data.EstimatedSalary,
            'Prediction': 'Exit' if prediction == 1 else 'Not Exit'
        }
        
        # Add probability if provided
        if probability is not None:
            response['Probability'] = round(float(probability), 4)
        
        return response
    
    def predict_new(self, data: CustomerData) -> Dict:
        """
        Makes a churn prediction for new customer data.
        """
        logger.info("Making new prediction")
        
        try:
            # Prepare input data
            X_new = self._prepare_input_data(data)
            
            # Apply preprocessing pipeline
            X_processed = self.pipe.transform(X_new)
            
            # Make prediction
            y_pred = self.classifier.predict(X_processed)[0]
            
            # Format and return response
            response = self._format_response(data, y_pred)
            logger.info(f"Prediction result: {response['Prediction']}")
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_batch(self, data_list: List[CustomerData]) -> List[Dict]:
        """
        Makes churn predictions for a batch of customer data.
        """
        logger.info(f"Making batch prediction for {len(data_list)} customers")
        
        try:
            results = []
            for i, data in enumerate(data_list):
                logger.debug(f"Processing item {i+1}/{len(data_list)}")
                prediction = self.predict_new(data)
                results.append(prediction)
            
            logger.info(f"Successfully completed batch prediction for {len(data_list)} customers")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    def predict_with_probability(self, data: CustomerData) -> Dict:
        """
        Makes a churn prediction with probability for new customer data.
        """
        logger.info("Making prediction with probability")
        
        try:
            # Prepare input data
            X_new = self._prepare_input_data(data)
            
            # Apply preprocessing pipeline
            X_processed = self.pipe.transform(X_new)
            
            # Make prediction with probability
            y_pred = self.classifier.predict(X_processed)[0]
            y_prob = self.classifier.predict_proba(X_processed)[0][1]  # Probability of class 1 (Exit)
            
            # Format and return response
            response = self._format_response(data, y_pred, y_prob)
            logger.info(f"Prediction result: {response['Prediction']} with probability {response['Probability']}")
            return response
            
        except Exception as e:
            logger.error(f"Prediction with probability failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
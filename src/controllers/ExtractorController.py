import re
import json
import logging
from typing import Optional
from fastapi import HTTPException
from src.models.schemas import CustomerData
from src.helpers.config import settings

# Configure logger
logger = logging.getLogger(__name__)

class ExtractorController:
    """
    Controller class for extracting customer data from natural language text using OpenAI's models.
    """
    
    def __init__(self):
        """Initialize the ExtractorController."""
        self.client = settings.client
        self.model = settings.openai_model
    
    def _generate_prompt(self, text: str) -> str:
        """Generate a prompt for the OpenAI model to extract customer data from text."""
        
        return f"""
        Extract the following fields from the text and provide them in JSON format: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
        
        Example:
        Text: "Jane Smith is a 35-year-old female from Canada with a credit score of 650. She has been with the bank for 3 years, has a balance of 2000.0 USD, holds 1 product, owns a credit card, is an active member, and earns an estimated salary of 75000.0 USD."
        JSON: {{
            "CreditScore": 650,
            "Geography": "Canada",
            "Gender": "Female",
            "Age": 35,
            "Tenure": 3,
            "Balance": 2000.0,
            "NumOfProducts": 1,
            "HasCrCard": true,
            "IsActiveMember": true,
            "EstimatedSalary": 75000.0
        }}
        
        Text: "{text}"
        JSON:
        """
    
    def _extract_last_json_from_output(self, output: str) -> Optional[str]:
        """Extract the last JSON object from the OpenAI output."""
        
        json_matches = re.findall(r'\{.*?\}', output, re.DOTALL)
        if json_matches:
            return json_matches[-1]
        logger.warning("No JSON found in OpenAI output")
        return None
    
    def _post_process_customer_data(self, data: CustomerData) -> CustomerData:
        """Validate and standardize customer data."""
        
        logger.debug(f"Post-processing customer data: {data.model_dump()}")
        
        # Convert boolean fields to integers
        data.HasCrCard = int(data.HasCrCard)
        data.IsActiveMember = int(data.IsActiveMember)
        
        logger.debug(f"Post-processed customer data: {data.model_dump()}")
        return data
    
    def extract_features(self, text: str) -> CustomerData:
        """Extract customer features from natural language text using OpenAI's model."""
        
        logger.info("Extracting features from text")
        
        # Generate prompt for OpenAI
        prompt = self._generate_prompt(text)
        logger.debug(f"Generated prompt: {prompt}")
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured data from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more deterministic outputs
                max_tokens=1000
            )
            
            # Extract response text
            result_text = response.choices[0].message.content
            logger.debug(f"OpenAI result: {result_text[:100]}...")  # Log just the first 100 chars
            
        except Exception as e:
            logger.error(f"Error during OpenAI processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI processing failed: {str(e)}")
        
        # Extract and process JSON
        json_text = self._extract_last_json_from_output(result_text)
        if not json_text:
            logger.error("JSON format not found in the output")
            raise HTTPException(status_code=400, detail='JSON format not found in the output')
            
        try:
            result_json = json.loads(json_text)
            logger.debug(f"Parsed JSON: {result_json}")
            
            # Apply Pydantic validation
            customer_data = CustomerData(**result_json)
            
            # Post-process and validate
            customer_data = self._post_process_customer_data(customer_data)
            logger.info("Successfully extracted customer data")
            return customer_data
                
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            raise HTTPException(status_code=400, detail=f'Failed to parse the structured data: {str(e)}')
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
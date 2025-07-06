from pydantic import BaseModel, Field, field_validator
from typing import Literal, List

class CustomerData(BaseModel):
    """Pydantic model for customer data used for churn prediction."""
    
    CreditScore: float = Field(..., description='Credit score of the customer', ge=300, le=900)
    Geography: Literal['Spain', 'Germany', 'France'] = Field(..., description='Customer country of residence')
    Gender: Literal['Male', 'Female'] = Field(..., description='Customer gender')
    Age: int = Field(..., description='Age of the customer', ge=18, le=100)
    Tenure: int = Field(..., description='Number of years the customer has been with the bank', ge=0, le=10)
    Balance: float = Field(..., description='Account balance')
    NumOfProducts: int = Field(..., description='Number of products the customer has', ge=1, le=4)
    HasCrCard: int = Field(..., description='Does the customer have a credit card (1 for yes, 0 for no)', ge=0, le=1)
    IsActiveMember: int = Field(..., description='Is the customer an active member (1 for yes, 0 for no)', ge=0, le=1)
    EstimatedSalary: float = Field(..., description='Estimated salary of the customer', ge=0)
    
    @field_validator('Geography', 'Gender', mode='before')
    @classmethod
    def normalize_case(cls, v, info):
        """Normalize string fields to proper case."""
        if isinstance(v, str):
            if info.field_name == 'Geography':
                return v.title()
            elif info.field_name == 'Gender':
                return v.capitalize()
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Female",
                "Age": 35,
                "Tenure": 3,
                "Balance": 2000.0,
                "NumOfProducts": 1,
                "HasCrCard": True,
                "IsActiveMember": True,
                "EstimatedSalary": 75000.0
            }
        }
    }
        
# Request models
class TextRequest(BaseModel):
    text: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Jane Smith is a 35-year-old female from France with a credit score of 650. She has been with the bank for 3 years, has a balance of 2000.0 USD, holds 1 product, owns a credit card, is an active member, and earns an estimated salary of 75000.0 USD."
            }
        }
    }

class BatchTextRequest(BaseModel):
    texts: List[str]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "Jane Smith is a 35-year-old female from France with a credit score of 650. She has been with the bank for 3 years, has a balance of 2000.0 USD, holds 1 product, owns a credit card, is an active member, and earns an estimated salary of 75000.0 USD.",
                    "John Doe is a 42-year-old male from Germany with a credit score of 720. He has been with the bank for 5 years, has a balance of 15000.0 USD, holds 2 products, owns a credit card, is not an active member, and earns an estimated salary of 95000.0 USD."
                ]
            }
        }
    }
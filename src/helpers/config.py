import os
import joblib
import logging
from pathlib import Path
from dotenv import load_dotenv
import openai

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings:
    """Application settings and configurations."""
    
    def __init__(self):
        """Initialize settings from environment variables."""
        # Load environment variables
        load_dotenv(override=True)
        
        # API settings
        self.api_name = os.getenv('API_NAME', 'Churn-Detection-Model')
        self.api_port = int(os.getenv('API_PORT', 8000))
        self.api_description = os.getenv('API_DESCRIPTION', 'Churn Detection Model API')
        self.api_secret_key = os.getenv('API_SECRET_KEY', '')
        
        # Paths
        self.base_dir = Path(__file__).resolve().parent.parent
        self.assets_folder = self.base_dir / 'assets'
        
        # Model paths
        self.preprocessor_path = self.assets_folder / 'preprocessor.pkl'
        self.classifier_path = self.assets_folder / 'Tuned-RF-with-SMOTE.pkl'
        
        # OpenAI settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = 'gpt-4o-mini'
        
        # Initialize OpenAI client
        self._initialize_openai()
        
        # Load ML models
        self._load_ml_models()
        
        # Define constants
        self.columns = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        self.dtypes = {
            'CreditScore': float,
            'Geography': str,
            'Gender': str,
            'Age': int,
            'Tenure': int,
            'Balance': float,
            'NumOfProducts': int,
            'HasCrCard': int,  # Changed from bool to int to match post-processing
            'IsActiveMember': int,  # Changed from bool to int to match post-processing
            'EstimatedSalary': float
        }
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        # Check if API key is provided
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        try:
            openai.api_key = self.openai_api_key
            self.client = openai.OpenAI(api_key=self.openai_api_key)
            logger.info(f"Successfully initialized OpenAI client with model {self.openai_model}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise
    
    def _load_ml_models(self):
        """Load ML pipeline and classifier models."""
        try:
            self.pipe = joblib.load(self.preprocessor_path)
            self.classifier = joblib.load(self.classifier_path)
            logger.info("Successfully loaded preprocessor and classifier models")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

# Create settings instance
settings = Settings()
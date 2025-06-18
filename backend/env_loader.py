"""
Environment Variables Loader
Handles loading of API keys and configuration from .env files or environment variables
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def load_environment_variables() -> Dict[str, Optional[str]]:
    """
    Load environment variables for the DSR-RL system

    Returns:
        Dictionary with configuration values
    """
    try:
        # Try to load from .env file first
        env_file_path = ".env"
        if os.path.exists(env_file_path):
            logger.info("Loading environment variables from .env file")
            with open(env_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()

        # Load configuration
        config = {
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "google_cse_id": os.getenv("GOOGLE_CSE_ID"),
            "flask_env": os.getenv("FLASK_ENV", "development"),
            "flask_debug": os.getenv("FLASK_DEBUG", "True").lower() == "true",
        }

        # Validate critical configuration
        missing_config = []
        if not config["google_api_key"]:
            missing_config.append("GOOGLE_API_KEY")
        if not config["google_cse_id"]:
            missing_config.append("GOOGLE_CSE_ID")

        if missing_config:
            logger.warning(
                f"Missing environment variables: {', '.join(missing_config)}"
            )
            logger.warning(
                "The system will work with limited functionality (Wikipedia and datasets only)"
            )
            logger.warning(
                "To enable Google Custom Search, set these environment variables:"
            )
            for var in missing_config:
                logger.warning(f"  export {var}='your_value_here'")
        else:
            logger.info("✅ Google Custom Search API configuration loaded successfully")

        return config

    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return {
            "google_api_key": None,
            "google_cse_id": None,
            "flask_env": "development",
            "flask_debug": True,
        }


def check_api_configuration() -> Dict[str, bool]:
    """
    Check if APIs are properly configured

    Returns:
        Dictionary with API status
    """
    config = load_environment_variables()

    status = {
        "google_custom_search": bool(
            config["google_api_key"] and config["google_cse_id"]
        ),
        "datasets_available": True,  # Always available
        "wikipedia_available": True,  # Always available
    }

    return status


def get_api_instructions() -> str:
    """
    Get instructions for setting up missing APIs

    Returns:
        Formatted instructions string
    """
    instructions = """
🔧 Google Custom Search API Setup Instructions:

1. 📝 Get Google API Key:
   - Go to: https://console.developers.google.com/
   - Create a new project or select existing one
   - Enable 'Custom Search API'
   - Create credentials (API Key)

2. 🔍 Get Custom Search Engine ID:
   - Go to: https://cse.google.com/cse/
   - Create a new Custom Search Engine
   - Configure to search the entire web
   - Copy your Search Engine ID

3. ⚙️ Set Environment Variables:
   Option A - Export in terminal:
   export GOOGLE_API_KEY='your_api_key_here'
   export GOOGLE_CSE_ID='your_search_engine_id_here'
   
   Option B - Create .env file:
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   echo "GOOGLE_CSE_ID=your_search_engine_id_here" >> .env

4. 🔄 Restart the application

💡 Note: Google Custom Search API includes 100 free searches per day.
   For more searches, billing must be enabled in Google Cloud Console.
"""

    return instructions

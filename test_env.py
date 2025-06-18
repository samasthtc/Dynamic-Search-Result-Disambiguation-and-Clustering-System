# from dotenv import load_dotenv
import os

# load_dotenv()

print("Testing environment variable loading...")


def load_environment_variables():
    """
    Load environment variables and return a configuration dictionary.
    """
    try:
        config = {
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "google_cse_id": os.getenv("GOOGLE_CSE_ID"),
            "flask_env": os.getenv("FLASK_ENV", "development"),
            "flask_debug": os.getenv("FLASK_DEBUG", "True").lower() == "true",
        }

        print("config:")

        # Validate critical configuration
        missing_config = []
        if not config["google_api_key"]:
            missing_config.append("GOOGLE_API_KEY")
        if not config["google_cse_id"]:
            missing_config.append("GOOGLE_CSE_ID")
        if not config["flask_env"]:
            missing_config.append("FLASK_ENV")
        if not config["flask_debug"]:
            missing_config.append("FLASK_DEBUG")

        if missing_config:
            print(f"Missing environment variables: {', '.join(missing_config)}")
            print(
                "The system will work with limited functionality (Wikipedia and datasets only)"
            )
            print("To enable Google Custom Search, set these environment variables:")
            for var in missing_config:
                print(f"  export {var}='your_value_here'")
        else:
            print("✅ Google Custom Search API configuration loaded successfully")

        return config

    except Exception as e:
        print(f"Error loading environment variables: {e}")
        return {
            "google_api_key": None,
            "google_cse_id": None,
            "flask_env": "development",
            "flask_debug": True,
        }


def check_api_configuration():
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


if __name__ == "__main__":
    config = load_environment_variables()
    print("Configuration:", config)

    api_status = check_api_configuration()
    print("API Status:", api_status)

# This code is a simplified version of the original backend/env_loader.py
# and is designed to run in a standalone Python environment for testing purposes.
# It loads environment variables, checks for critical configurations,

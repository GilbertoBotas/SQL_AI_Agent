import os
from dotenv import load_dotenv

load_dotenv()

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        'SQL_USER_NAME',
        'SQL_PASSWORD',
        'SQL_SERVER',
        'SQL_DATABASE',
        'OPENAI_API_KEY'
    ]
    missing = []
    for var in required_vars:
        if not os.environ.get(var):
            missing.append(var)

    if missing:
        error_msg = "Missing required environment variables: " + ", ".join(missing)
        raise EnvironmentError(error_msg)
    
    print("All required environment variables are set.") 
        
validate_environment()

def get_sql_credentials():
    """Get SQL credentials from environment variables."""
    return {
        'username': os.getenv('SQL_USER_NAME'),
        'password': os.getenv('SQL_PASSWORD'),
        'server': os.getenv('SQL_SERVER'),
        'database': os.getenv('SQL_DATABASE')
    }

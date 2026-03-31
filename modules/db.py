# db.py
import modules.file_manager as file_manager
from sqlalchemy import create_engine

def get_engine():
    database_credentials = file_manager.get_sql_credentials()
    
    username = database_credentials['username']
    password = database_credentials['password']
    server = database_credentials['server']
    database = database_credentials['database']
    driver = 'ODBC Driver 17 for SQL Server'
    
    connection_url = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
    engine = create_engine(
        url=connection_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
    
    return engine
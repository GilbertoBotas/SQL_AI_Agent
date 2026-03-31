# schema_cache.py
from sqlalchemy import inspect
from functools import lru_cache

@lru_cache(maxsize=1)
def load_schema(engine):
   inspector = inspect(engine)
   schema = {}

   for table_name in inspector.get_table_names(schema="public"):
       cols = inspector.get_columns(table_name, schema="public")
       schema[table_name] = [
           {"name": c["name"], "type": str(c["type"])}
           for c in cols
       ]

   return schema
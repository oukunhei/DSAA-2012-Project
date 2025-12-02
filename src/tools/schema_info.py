import os
import sqlite3

def get_db_schema(db_path): 
    if not os.path.exists(db_path):
        return ""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_prompt = ""
        for table_name in tables:
            table_name = table_name[0]
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            schema_prompt += cursor.fetchone()[0] + ";\n\n"
        conn.close()
        return schema_prompt.strip()
    except:
        return ""
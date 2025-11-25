#get_table_names增加空值检查
import json
import sqlite3
import threading
import queue
from tabulate import tabulate
from typing import List, Tuple, Dict, Any
from src.tools.sql_execution import execute_sql

def get_table_names(db_path: str) -> List[str]:
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    result = execute_sql(sql, db_path, timeout=60)
    if not result or "result" not in result or not result["result"]:
        return []  # 或其他默认值
    return [row[0] for row in result["result"]]

def get_column_names(db_path: str, table_name: str) -> List[str]:
    sql = f"PRAGMA table_info(`{table_name}`);"
    result = execute_sql(sql, db_path, timeout=60)
    return [row[1] for row in result["result"]]

def get_column_types(db_path: str, table_name: str) -> List[str]:
    sql = f"PRAGMA table_info(`{table_name}`);"
    result = execute_sql(sql, db_path, timeout=60)
    return [row[2] for row in result["result"]]

def get_column_values(db_path: str, table_name: str, column_name: str, column_type: str, values_num: int = 3) -> List[str]:
    if column_type.upper() == "BLOB":
        return []
    sql = f"SELECT DISTINCT `{column_name}` FROM `{table_name}` LIMIT {values_num};"
    result = execute_sql(sql, db_path, timeout=60)
    return [row[0] for row in result["result"]]

def get_primary_keys(db_path: str, table_name: str) -> List[str]:
    sql = f"PRAGMA table_info(`{table_name}`);"
    result = execute_sql(sql, db_path, timeout=60)
    return [row[1] for row in result["result"] if row[5] != 0]

def get_foreign_keys(db_path: str, table_name: str) -> List[Dict[str, Dict[str, str]]]:
    sql = f"PRAGMA foreign_key_list(`{table_name}`);"
    result = execute_sql(sql, db_path, timeout=60)
    foreign_keys = []
    for row in result["result"]:
        target_table, source_column, target_column = row[2:5]
        if target_table and source_column and target_column:
            foreign_keys.append({
                "source": {"table_name": table_name, "column_name": source_column},
                "target": {"table_name": target_table, "column_name": target_column}
            })
    return foreign_keys

def get_column_meaning_dict(column_meaning_json_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    with open(column_meaning_json_path, "r", encoding='utf-8') as f:
        column_meaning = json.load(f)
    column_meaning_dict = {}
    for key, value in column_meaning.items():
        db_id, table_name, column_name = key.split("|")
        db_id, table_name, column_name = db_id.lower(), table_name.lower(), column_name.lower()
        if db_id not in column_meaning_dict:
            column_meaning_dict[db_id] = {}
        if table_name not in column_meaning_dict[db_id]:
            column_meaning_dict[db_id][table_name] = {}
        column_meaning_dict[db_id][table_name][column_name] = value.strip()
    return column_meaning_dict

def get_column_meanings(column_meaning_dict: Dict[str, Dict[str, Dict[str, str]]], db_id: str, table_name: str, column_names: List[str]) -> List[str]:
    column_meanings = []
    for column_name in column_names:
        try:
            column_meanings.append(column_meaning_dict[db_id.lower()][table_name.lower()][column_name.lower()])
        except KeyError:
            column_meanings.append("")  # 或者用其他默认值
            print(f"Warning: No meaning found for {db_id}.{table_name}.{column_name}")
    return column_meanings

def construct_col_ddl(column_name: str, column_type: str) -> str:
    return f"`{column_name}` {column_type}"

def construct_pk_ddl(primary_keys: List[str]) -> str:
    pks_str = ", ".join([f"`{pk}`" for pk in primary_keys])
    return f"PRIMARY KEY ({pks_str})"

def construct_fk_ddl(source_column_name: str, target_table_name: str, target_column_name: str) -> str:
    return f"FOREIGN KEY (`{source_column_name}`) REFERENCES `{target_table_name}`(`{target_column_name}`)"

def construct_table_ddl(table_schema: dict, column_meanings: List[str], column_values: List[List[str]]) -> str:
    table_name = table_schema["table_name"]
    columns_ddl_list = []
    for col_name, col_type, col_meaning, col_values in zip(table_schema["column_names"], table_schema["column_types"], column_meanings, column_values):
        col_ddl = construct_col_ddl(col_name, col_type)
        
        comment_part = ""
        if col_meaning.strip():
            comment_part += col_meaning
        if len(col_values) > 0:
            comment_part += f" #Values Examples: [{', '.join([str(_) for _ in col_values])}]"
            
        if comment_part:
            col_ddl += f", -- {comment_part.strip()}"
        else:
            col_ddl += ","
        columns_ddl_list.append(col_ddl)
    
    if table_schema["primary_keys"]:
        columns_ddl_list.append(construct_pk_ddl(table_schema["primary_keys"]) + ",")
        
    for fk in table_schema["foreign_keys"]:
        source_column_name = fk["source"]["column_name"]
        target_table_name = fk["target"]["table_name"]
        target_column_name = fk["target"]["column_name"]
        columns_ddl_list.append(construct_fk_ddl(source_column_name, target_table_name, target_column_name) + ",")
    
    # fix the last line defination syntax
    if ", -- " in columns_ddl_list[-1]:
        columns_ddl_list[-1] = columns_ddl_list[-1].replace(", -- ", " -- ", 1)
    elif columns_ddl_list[-1].endswith(","):
        columns_ddl_list[-1] = columns_ddl_list[-1].rstrip(",")
        
    columns_ddl = "\n\t".join(columns_ddl_list)
    table_ddl = f"CREATE TABLE `{table_name}` (\n\t{columns_ddl}\n);"
    return table_ddl

def get_db_schema(db_path: str,
                  values_num: int = 3,
                  add_column_meaning: bool = True,
                  column_meaning_json_path: str = None) -> List[Dict]:
    if add_column_meaning and column_meaning_json_path is None:
        raise ValueError("column_meaning_json_path is required when add_column_meaning is True")
    column_meaning_dict = get_column_meaning_dict(column_meaning_json_path) if add_column_meaning else None
    
    db_id = db_path.split("/")[-1].split(".")[0]
    schema = []
    table_names = get_table_names(db_path)
    for table_name in table_names:
        column_names = get_column_names(db_path, table_name)
        column_types = get_column_types(db_path, table_name)
        column_values = [get_column_values(db_path, table_name, column_name, column_type, values_num) for column_name, column_type in zip(column_names, column_types)]
        primary_keys = get_primary_keys(db_path, table_name)
        foreign_keys = get_foreign_keys(db_path, table_name)
        column_meanings = get_column_meanings(column_meaning_dict, db_id, table_name, column_names) if add_column_meaning else ["" for _ in column_names]
        table_schema = {
            "table_name": table_name,
            "column_types": column_types,
            "column_names": column_names,
            "column_values": column_values,
            "column_meanings": column_meanings,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }
        table_ddl = construct_table_ddl(table_schema, column_meanings, column_values)
        schema.append(table_ddl)
    return schema

if __name__ == "__main__":
    db_path = "data/bird/dev/dev_databases/dev_databases/debit_card_specializing/debit_card_specializing.sqlite"
    schema = get_db_schema(db_path, add_column_meaning=True, column_meaning_json_path="data/bird/dev/dev_databases/dev_databases/debit_card_specializing/column_meanings.json")
    for table in schema:
        print(table)
        print("\n")
# 对于每个item，先获取执行结果不同的represent_sqls，然后送到selector里面选，输出final_sql文件
import os
import json
import re
import sqlite3
from typing import List, Dict 
import gc
import torch
import sqlglot
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join (os.path.dirname(__file__), '..')))
from schema_info import get_db_schema
from Selector import Selector
from sql_execution import execute_sql

# 常量定义
DEV_DATA_PATH = '/ssd/yunxiou/data/yunxiou/bird/dev/dev.json'
COLUMN_MEANING_PATH = "/ssd/yunxiou/data/yunxiou/bird/dev/column_meaning.json"
DEV_DB_PATH_TEMPLATE = "/ssd/yunxiou/data/yunxiou/bird/dev/dev_databases/dev_databases/{db_id}/{db_id}.sqlite"

def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL query.
    
    Args:
        sql: The SQL query to normalize.
    Returns:
        The normalized SQL query.
    """
    sql = sql.strip()
    if sql.startswith("```sql") and sql.endswith("```"):
        sql = sql[6:-3]
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
        return parsed.sql(dialect="sqlite", normalize=True, pretty=False, comments=False)
    except Exception as e:
        return sql
    
def extract_sql_query_answer(sql_generation_response: str) -> str:
    try:
        sql_query = re.search(r"<sql>(.*?)</sql>", sql_generation_response, flags=re.DOTALL)
        if sql_query != None:
            sql_query = sql_query.group(1).strip()
            return normalize_sql(sql_query)
        else:
            print(f"Cannot extract SQL query from response")
            return None
    except Exception as e:
        print(f"Error parsing sql generation response: {e}")
        return None

def validate_sql_query(query: str, db_path: str) -> dict:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(f"EXPLAIN {query}")
        return {"is_valid": True, "error_msg": ""}
    except sqlite3.Error as e:
        print(f"SQL验证失败: {e}\n查询: {query}")
        return {"is_valid": False, "error_msg": str(e)}
    finally:
        if conn:
            conn.close()

def group_by_execution_result(sql_list: List[str], db_path: str) -> Dict[str, List[str]]:
    result_groups = defaultdict(list)
    for sql in sql_list:
        result = execute_sql(sql, db_path, timeout=30)['result']
        if result is None:
            continue
        key = frozenset(result)
        result_groups[key].append(sql)

    represent_sqls =[queries[0] for key, queries in result_groups.items()]

    return represent_sqls

def select_candidates(dataset, output_file) -> List[str]:
    selector = Selector(temperature=0.8, top_p=0.8)
    results = {}
    responses = []
    
    for item in dataset: 
        # 显存清理 (每条都清，最安全)
        gc.collect()
        torch.cuda.empty_cache()

        q_id = item.get('question_id')
        print(f"\n处理问题 ID: {q_id}")
        try:
            db_id = item['db_id']
            nl_question = item['question']
            db_path = DEV_DB_PATH_TEMPLATE.format(db_id=db_id)            
            schema_info = get_db_schema(db_path, add_column_meaning=True, column_meaning_json_path=COLUMN_MEANING_PATH)

            response = selector.invoke(nl_question, schema_info, item['sql_candidates'])

            responses.append(response)
            with open('/ssd/yunxiou/DSAA-2012/selector_responses.json', 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent=4, ensure_ascii=False)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[OOM 警告] ID {q_id} 显存溢出，已跳过并清理。")
                torch.cuda.empty_cache()
            else:
                print(f"\n[错误] ID {q_id}: {e}")
            continue
        except Exception as e:
            print(f"\n[错误] ID {q_id}: {e}")
            continue

        sql = extract_sql_query_answer(response)
        if sql is None:
            print(f"[警告] ID {q_id} 未能提取SQL，跳过。")
            continue

        results[f"{q_id}"] = sql

        # 4. 纯粹的进度保存 (每10条覆盖写入一次文件，不是断点续存)
        if len(results) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # 5. 最终保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # 加载测试数据
    with open('/ssd/yunxiou/DSAA-2012/qwen-7B-bird-dev.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    output_file = "/ssd/yunxiou/DSAA-2012/predict_result.json"
    select_candidates(data, output_file=output_file)  


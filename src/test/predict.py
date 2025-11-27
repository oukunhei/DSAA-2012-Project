# 多语言测试文件(未加入fixer模块）针对xiyan模型
import sys
import os
import time
import json
import re
import sqlite3
from typing import List, Dict, Tuple

import sqlglot

# 设置项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
os.chdir(project_root)
from src.tools.schema_info import get_db_schema
from src.test.Selector import Selector

# 常量定义
DEV_DATA_PATH = 'data/bird/dev/dev.json'
COLUMN_MEANING_PATH = "data/bird/dev/column_meaning.json"
DEV_DB_PATH_TEMPLATE = "data/bird/dev/dev_databases/dev_databases/{db_id}/{db_id}.sqlite"

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


def get_query(responses: List[str]) -> List[str]:
    """将responses列表转成查询列表，并返回无效查询及其错误信息"""
    queries = []
    
    for resp in responses:
        query = extract_sql_query_answer(resp)
        if query != None:
            queries.append(query)
                
    return queries

def test_single_item(selector: Selector, item: Dict) -> bool:
    nl_question = item['question']
    
    # 准备数据库信息
    db_path = DEV_DB_PATH_TEMPLATE.format(db_id=db_id)
    schema = get_db_schema(db_path, add_column_meaning=True, column_meaning_json_path=COLUMN_MEANING_PATH)
    schema_info = "\n".join(f"{k}: {v}" for d in schema for k, v in d.items())
    
    # 生成SQL响应
    responses = sql_agent(question, schema_info, evidence)
    usage = sql_agent.get_usage_stats() 
    print(f"目前sql agent累计使用的token数: {usage['prompt_tokens']}, {usage['completion_tokens']}")
    
    # 提取sql查询
    queries = get_query(responses)
    valid_queries = []
    invalid_queries = []

    for q in queries:
        validation_result = validate_sql_query(q, db_path)
        if validation_result['is_valid']:
            valid_queries.append(q)
        else:
            invalid_queries.append({
                "query": q,
                "error": validation_result['error_msg']
            })

    # 返回JSON格式的数据
    valid_data = {question_id: valid_queries}
    invalid_data = {question_id: invalid_queries}
    
    return valid_data, invalid_data

def run_batch_test(test_data: List[Dict]) -> Tuple[int, float]:

    total_items = len(test_data) 
    print(f"\n测试 {total_items} 项数据")

    # 初始化Selector
    selector = Selector()
    
    start_time = time.time()
    
    for i, item in enumerate(test_data, 1):
        print(f"\n[测试 {i}/{total_items}] 数据库: {item['db_id']}")
        print(f"问题: {item['question']}")

        valid_data, invalid_data = test_single_item(selector, item)
        
        # 合并结果
        all_valid_queries.update(valid_data)
        all_invalid_queries.update(invalid_data)

        print(f"有效查询：{len(valid_data[item['question_id']])}条，无效查询{len(invalid_data[item['question_id']])}条")

    # 保存结果到JSON文件
    with open(f'xiyan_bird_dev_valid.json', 'w', encoding='utf-8') as f:
        json.dump(all_valid_queries, f, indent=2, ensure_ascii=False)
    
    with open(f'xiyan_bird_dev_invalid.json', 'w', encoding='utf-8') as f:
        json.dump(all_invalid_queries, f, indent=2, ensure_ascii=False)

            
    elapsed = time.time() - start_time
    
    print(f"\n测试完成 耗时: {elapsed:.2f}秒")
    return None

if __name__ == "__main__":
    # 加载测试数据
    with open('data/test_set/omni-bird-dev.json', 'r') as f:
        data = json.load(f)



# 第一步：SC，保存sql，执行结果和分数
# 第二步：执行selector选出的，查看结果是否在执行结果里，在的话给相应结果加分，不在就得到默认分数
# 第三步：提取出最高分并和标准结果比对，计算正确率

import os
import sys
import json
from collections import defaultdict
from typing import List, Dict, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.tools.sql_execution import execute_sql

gt_json_path = "/ssd/yunxiou/data/yunxiou/bird/dev/dev.json"
db_path_template = "/ssd/yunxiou/data/yunxiou/bird/dev/dev_databases/dev_databases/{db_id}/{db_id}.sqlite"
final_sql_path = "/ssd/yunxiou/DSAA-2012/src/test/omni_final_sqls.json"

def group_by_execution_result(sql_list: List[str], db_path: str) -> Dict[str, List[str]]:
    """return:
        Dict[result_frozenset: List of sqls producing this result]
    """
    result_groups = defaultdict(list)
    for sql in sql_list:
        result = execute_sql(sql, db_path, timeout=30)['result']
        if result is None:
            continue
        key = frozenset(result)
        result_groups[key].append(sql)
    
    represent_sqls ={}
    for group in result_groups.values():
        represent_sqls[group[0]] = len(group)

    return result_groups, represent_sqls


def score(query_file: str, selector_result_file: str, default_score: int = 4):

    with open(query_file, 'r', encoding='utf-8') as f:
        sqls = json.load(f)

    with open(gt_json_path, 'r') as f:
        gt_map = json.load(f)
    
    with open(selector_result_file, 'r', encoding='utf-8') as f:
        selector_results = json.load(f)

    total = len(gt_map)
    print(f"Total questions: {total}")

    final_sqls = {}

    for item in sqls:
        qid = item['question_id']
        print(f"Processing QID: {qid}")

        sql_list = item['sql_candidates']
        q = gt_map[qid]  
        db_id = q['db_id']
        db_path = db_path_template.format(db_id=db_id)
        if not os.path.exists(db_path):
            continue

        result_groups, represent_sqls = group_by_execution_result(sql_list, db_path)

        selector_sql = selector_results.get(f"{qid}", sql_list[0] if sql_list else None)
        selector_execution = execute_sql(selector_sql, db_path, timeout=30)["result"]
        if selector_execution is None:
            continue
        selector_execution = frozenset(selector_execution)

        if selector_execution is not None:
            if selector_execution in result_groups.keys():
                key = result_groups[selector_execution][0]
                represent_sqls[key] += default_score  # 给等价sql加分
            else: 
                result_groups[selector_execution] = [selector_sql]
                represent_sqls[selector_sql] = default_score 
        
        # 选择分数最高的sql
        best_sql = max(represent_sqls.items(), key=lambda x: x[1])[0]
        final_sqls[f"{qid}"] = best_sql

        with open(final_sql_path, 'w', encoding='utf-8') as f:
            json.dump(final_sqls, f, indent=2, ensure_ascii=False)
    return final_sqls
                
def validate(final_sqls):  
    
    with open(gt_json_path, 'r') as f:
        gt_map = json.load(f) 
    
    selfcons_matched = 0
    sql_exec_failed = 0
    results = {}  # 用于存储每个问题的详细结果

    for qid, sql in final_sqls.items():
        qid = int(qid)
        if qid not in gt_map:
            print(f"[QID {qid}] ❌ 题号不在数据库内")
            continue

        q = gt_map[qid]
        db_id = q['db_id']
        db_path = db_path_template.format(db_id=db_id)

        if not os.path.exists(db_path):
            print(f"[QID {qid}] ❌ 数据库文件不存在: {db_path}")
            continue

        results[qid] = False  # 默认结果为不匹配

        try:
            expected_exec = execute_sql(q['SQL'], db_path, timeout=30)

            if not expected_exec or 'result' not in expected_exec or expected_exec['result'] is None:
                sql_exec_failed += 1
                print(f"[QID {qid}] ❌ 标准SQL执行无结果")
                continue

            expected_result = expected_exec['result']
        except Exception as e:
            sql_exec_failed += 1
            print(f"[QID {qid}] ❌ 标准SQL执行异常: {str(e)}")
            continue


        try:
            execution = execute_sql(sql, db_path, timeout=30)

            if not execution or 'result' not in execution or execution['result'] is None:
                print(f"[QID {qid}] ❌ 选择SQL执行无结果")
                continue

            result = execution['result']

            if set(result) == set(expected_result):
                selfcons_matched += 1
                results[qid] = True  # 标记为匹配
                print(f"[QID {qid}] ✅ SQL匹配")
            else:
                print(f"[QID {qid}] ❌ SQL不匹配")
            
        except Exception as e:
            print(f"[QID {qid}] ❌ 选择SQL执行异常: {str(e)}")            

    print("\n" + "=" * 50)
    print(f"总问题数: {len(gt_map)}")
    print(f"标准SQL执行失败数量: {sql_exec_failed}")
    print(f"SQL执行正确率: {selfcons_matched}/{len(final_sqls)} = {selfcons_matched/len(final_sqls):.2%}")
    return results

if __name__ == "__main__":
    query_file = "/ssd/yunxiou/DSAA-2012/data/test_set/omni-bird-dev.json"
    selector_result_file = "/ssd/yunxiou/DSAA-2012/omni_selector.json"
    final_sqls = score(query_file, selector_result_file, default_score=4)
    results = validate(final_sqls)
    with open ("/ssd/yunxiou/DSAA-2012/src/test/omni_final_execution_result.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

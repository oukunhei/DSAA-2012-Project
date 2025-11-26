# 判断单个txt或json文件的Self Consistency
import os
import sys
import json
from collections import defaultdict
from typing import List, Dict, Tuple

# 设置项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tools.schema_info import execute_sql


def group_by_execution_result(sql_list: List[str], db_path: str) -> Dict[str, List[str]]:
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

def self_consistency_select(sql_list: List[str], db_path: str) -> Tuple[str, int]:
    groups, represent_sqls = group_by_execution_result(sql_list, db_path)
    if not groups:
        return None, None
    largest_group = max(groups.values(), key=len)
    return largest_group[0], represent_sqls

def SC(query_file: str):
    gt_json_path = "data/bird/dev/dev.json"
    db_path_template = "data/bird/dev/dev_databases/dev_databases/{db_id}/{db_id}.sqlite"
    output_json_path = f'src/data_processing/result_SC.json'

    with open(query_file, 'r', encoding='utf-8') as f:
        sqls = json.load(f)

    with open(gt_json_path, 'r') as f:
        gt_map = json.load(f)

    total = len(gt_map)
    print(f"Total questions: {total}")

    selfcons_matched = 0
    sql_exec_failed = 0
    results = {}
    details = []

    # with open(output_json_path, 'r', encoding='utf-8') as f:
        # results = json.load(f)
    for item in sqls:
        qid = item['question_id']
        sql_list = item['sql_candidates']
        # if qid < 517:
            # continue
        q = gt_map[qid]  
        db_id = q['db_id']
        db_path = db_path_template.format(db_id=db_id)
        if not os.path.exists(db_path):
            continue

        results[qid] = False
        standard_sql = q['SQL']
        expected_result = execute_sql(standard_sql, db_path, timeout=30)['result']
        if expected_result is None:
            sql_exec_failed += 1
            print(f"[QID {qid}] 的标准SQL执行失败")
            continue

        # Self-Consistency Accuracy
        print(len(sql_list))
        candidate_size = len(sql_list)
        selected_sql, represent_sqls = self_consistency_select(sql_list, db_path)

        if selected_sql is not None:
            result = execute_sql(selected_sql, db_path, timeout=30)['result']

            if result is None:
                sql_exec_failed += 1
                print(f"[QID {qid}] ❌ SQL执行失败")
                continue
            if set(result) == set(expected_result):
                selfcons_matched += 1
                results[qid] = True
                print(f"[QID {qid}] ✅ Self-Consistent match")
            else:
                print(f"[QID {qid}] ❌ Self-Consistent mismatch")

            details.append({
                "question_id": qid,
                "candidate_size": candidate_size,
                "represent_sqls": represent_sqls,
                "result": results[qid]
            })

        else:
            print(f"[QID {qid}] ❌ 无法做Self-Consistency选择")

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"总问题数: {total}")
    print(f"执行失败（GT SQL）数量: {sql_exec_failed}")
    print(f"Self-Consistency Accuracy: {selfcons_matched}/{total} = {selfcons_matched / total:.2%}")

if __name__ == "__main__":
    query_file = "data/raw_data/xiyan_7B_bird_dev.json"
    SC(query_file)
# 训练集需要的数据：nl_question, schema, candidate_sqls, SC_sql, standard_sql
from pathlib import Path
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tools.schema_info import get_db_schema

DEV_DB_PATH_TEMPLATE = "data/bird/dev/dev_databases/dev_databases/{db_id}/{db_id}.sqlite"
COLUMN_MEANING_PATH = "data/bird/dev/column_meaning.json"


def preparation(SC_file, SC_success_output_file, SC_fail_output_file):
    dev_path = "data/bird/dev/dev.json"

    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open(SC_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    SC_success_group = []
    SC_fail_group = []

    for item in data:
        qid = item['question_id']
        represent_sqls = item['represent_sqls']
        candidate_sqls = list(represent_sqls.keys())
        nl_question = dev_data[qid]['question']
        db_id = dev_data[qid]['db_id']
        db_path = DEV_DB_PATH_TEMPLATE.format(db_id=db_id)
        schema_info = get_db_schema(db_path)

        max_group_size = max(v for k,v in represent_sqls.items())
        SC_sql = next(k for k,v in represent_sqls.items() if v == max_group_size)

        if item["result"] == True:
            SC_success_group.append({
                "nl_question": nl_question,
                "schema_info": schema_info,
                "candidate_sqls": candidate_sqls,
                "SC_sql": SC_sql,
            })
        else:
            SC_fail_group.append({
                "nl_question": nl_question,
                "schema_info": schema_info,
                "candidate_sqls": candidate_sqls,
                "standard_sql": dev_data[qid]['SQL'],
                "SC_sql": SC_sql,
            })

    with open(SC_success_output_file, 'w', encoding='utf-8') as f:
        json.dump(SC_success_group, f, indent=4, ensure_ascii=False)
    
    with open(SC_fail_output_file, 'w', encoding='utf-8') as f:
        json.dump(SC_fail_group, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    SC_file = "src/data_processing/result_SC.json"
    SC_success_output_file = "data/raw_data/SC_success_group.json"
    SC_fail_output_file = "data/raw_data/SC_fail_group.json"


    preparation(SC_file, SC_success_output_file, SC_fail_output_file)
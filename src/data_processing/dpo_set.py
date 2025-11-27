from pathlib import Path
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

DEV_DB_PATH_TEMPLATE = "data/bird/dev/dev_databases/dev_databases/{db_id}/{db_id}.sqlite"
COLUMN_MEANING_PATH = "data/bird/dev/column_meaning.json"

def build_prompt(nl_question, schema_info, candidate_sqls):
    template_path = Path("src/templete/DPO_prompt.txt")
    template = template_path.read_text(encoding='utf-8')
    return template.format(nl_question=nl_question, schema_info=schema_info, candidate_sqls=candidate_sqls)

def build_DPO_set(SC_success_output_file, SC_fail_output_file, output_file): 

    with open(SC_success_output_file, 'r', encoding='utf-8') as f:
        success_data = json.load(f)

    with open(SC_fail_output_file, 'r', encoding='utf-8') as f:
        fail_data = json.load(f)
    
    
    DPO_set = []

    num_for_success = 4 * len(success_data) // 5
    num_for_fail = 4 * len(fail_data) // 5


    for item in success_data[num_for_success:]:
        nl_question = item['nl_question']
        candidate_sqls = item['candidate_sqls']
        schema_info = item['schema_info']

        prompt = build_prompt(nl_question, schema_info, candidate_sqls)

        chosen = item["SC_sql"]
        rejected_sql = None
        for sql in candidate_sqls:
            if sql != chosen:
                rejected_sql = sql

        DPO_set.append({
            "prompt": prompt,
            "chosen": f"<sql> {chosen} </sql>",
            "rejected": f"<sql> {rejected_sql} </sql>"
        })

    for item in fail_data[num_for_fail:]:
        nl_question = item['nl_question']
        candidate_sqls = item['candidate_sqls']
        schema_info = item['schema_info']

        prompt = build_prompt(nl_question, schema_info, candidate_sqls)
        chosen = item["standard_sql"]
        rejected_sql = item['SC_sql']
        DPO_set.append({
            "prompt": prompt,
            "chosen": f"<sql> {chosen} </sql>",
            "rejected": f"<sql> {rejected_sql} </sql>"
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(DPO_set, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    SC_success_output_file = "data/raw_data/SC_success_group.json"
    SC_fail_output_file = "data/raw_data/SC_fail_group.json"
    output_file = "data/train_set/DPO_set.json"
    build_DPO_set(SC_success_output_file, SC_fail_output_file, output_file)
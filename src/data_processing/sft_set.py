from pathlib import Path
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

DEV_DB_PATH_TEMPLATE = "data/bird/dev/dev_databases/dev_databases/{db_id}/{db_id}.sqlite"
COLUMN_MEANING_PATH = "data/bird/dev/column_meaning.json"

def build_input(nl_question, schema_info, candidate_sqls):
    template_path = Path("src/templete/SFT_input.txt")
    template = template_path.read_text(encoding='utf-8')
    return template.format(nl_question=nl_question, schema_info=schema_info, candidate_sqls=candidate_sqls)


def build_SFT_set(SC_success_output_file, SC_fail_output_file, output_file): 

    with open(SC_success_output_file, 'r', encoding='utf-8') as f:
        success_data = json.load(f)

    with open(SC_fail_output_file, 'r', encoding='utf-8') as f:
        fail_data = json.load(f)
    
    with open("src/templete/SFT_instruction.txt", 'r', encoding='utf-8') as f:
        instruction = f.read()
    
    SFT_set = []

    num_for_success = 4 * len(success_data) // 5
    num_for_fail = 4 * len(fail_data) // 5


    for item in success_data[:num_for_success]:
        nl_question = item['nl_question']
        candidate_sqls = item['candidate_sqls']
        schema_info = item['schema_info']

        input = build_input(nl_question, schema_info, candidate_sqls)
        output = item["SC_sql"]
        SFT_set.append({
            "instruction": instruction,
            "input": input,
            "output": f"<sql> {output} </sql>"
        })

    for item in fail_data[:num_for_fail]:
        nl_question = item['nl_question']
        candidate_sqls = item['candidate_sqls']
        schema_info = item['schema_info']

        input = build_input(nl_question, schema_info, candidate_sqls)
        output = item["standard_sql"]
        SFT_set.append({
            "instruction": instruction,
            "input": input,
            "output": f"<sql> {output} </sql>"
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(SFT_set, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    SC_success_output_file = "data/raw_data/SC_success_group.json"
    SC_fail_output_file = "data/raw_data/SC_fail_group.json"
    output_file = "data/train_set/SFT_set.json"
    build_SFT_set(SC_success_output_file, SC_fail_output_file, output_file)
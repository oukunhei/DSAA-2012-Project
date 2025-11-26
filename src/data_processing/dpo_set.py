from pathlib import Path
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tools.schema_info import get_db_schema

DEV_DB_PATH_TEMPLATE = "data/bird/dev/dev_databases/dev_databases/{db_id}/{db_id}.sqlite"
COLUMN_MEANING_PATH = "data/bird/dev/column_meaning.json"

def build_prompt(nl_question, schema_info, candidate_sqls):
    template_path = Path("src/templete/DPO_prompt.txt")
    template = template_path.read_text(encoding='utf-8')
    return template.format(nl_question=nl_question, schema_info=schema_info, candidate_sqls=candidate_sqls)

def build_DPO_set(SC_file, output_file):
    dev_path = "data/bird/dev/dev.json"

    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open(SC_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open("src/templete/SFT_instruction.txt", 'r', encoding='utf-8') as f:
        instruction = f.read()
    
    SFT_set = []

    for item in data:
        if item["result"] == True:
            continue
        qid = item['question_id']
        represent_sqls = item['represent_sqls']
        candidate_sqls = list(represent_sqls.keys())
        nl_question = dev_data[qid]['question']
        db_id = dev_data[qid]['db_id']
        standard_sql = dev_data[qid]['SQL']
        db_path = DEV_DB_PATH_TEMPLATE.format(db_id=db_id)
        schema = get_db_schema(db_path, add_column_meaning=True, column_meaning_json_path=COLUMN_MEANING_PATH)
        schema_info = "\n".join(schema)

        prompt = build_prompt(nl_question, schema_info, candidate_sqls)
        max_group_size = max(v for k,v in represent_sqls.items())
        output = next(k for k,v in represent_sqls.items() if v == max_group_size)
        SFT_set.append({
            "prompt": prompt,
            "chosen": f"<sql> {standard_sql} </sql>",
            "rejected": f"<sql> {output} </sql>"
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(SFT_set, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    SC_file = "src/data_processing/result_SC.json"
    output_file = "data/train_set/DPO_train_set.json"
    build_DPO_set(SC_file, output_file)
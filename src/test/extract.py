import json
import re

with open("src/test/selector_responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

PLACEHOLDER = "The final SQL query that answers the question."

def extract_sql(response_text: str):
    # Find all <sql>...</sql> blocks
    matches = re.findall(r"<sql>(.*?)</sql>", response_text, flags=re.DOTALL)
    # Return the first non-placeholder, else None
    for m in matches:
        content = m.strip()
        if content == PLACEHOLDER:
            continue
        return content
    return None

final_sqls = {}
for i in range(len(responses)):
    resp = responses[i]
    print(f"\nResponse {i+1}:")
    try:
        # If the item isn't a string, coerce to string
        text = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
        sql = extract_sql(text)
        if sql is not None:
            final_sqls[f"{i}"] = sql
            print(sql)
        else:
            print("Cannot extract SQL query from response")
    except Exception as e:
        print(f"Error parsing sql generation response: {e}")
        
with open("src/test/final_sqls.json", "w", encoding="utf-8") as f:
    json.dump(final_sqls, f, ensure_ascii=False, indent=2)
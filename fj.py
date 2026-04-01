import json
import re

INPUT_FILE = "examples.jsonl"
OUTPUT_FILE = "dataset_clean.jsonl"

def extract_objects(text):
    """Extract JSON objects by tracking brace depth."""
    objs = []
    depth = 0
    start = None

    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start:i+1])

    return objs


def normalize_json(obj):
    """Fix common LLM JSON mistakes."""
    obj = obj.strip()

    # remove trailing commas
    obj = re.sub(r",\s*}", "}", obj)
    obj = re.sub(r",\s*]", "]", obj)

    # replace smart quotes
    obj = obj.replace("“", '"').replace("”", '"')

    return obj


def repair_jsonl():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    objects = extract_objects(raw)

    print(f"Found {len(objects)} JSON objects")

    fixed = []

    for i, obj in enumerate(objects):

        obj = normalize_json(obj)

        try:
            parsed = json.loads(obj)
            fixed.append(parsed)
        except Exception as e:
            print("\nFAILED OBJECT:")
            print(obj)
            raise RuntimeError(f"\nJSON repair failed on object {i}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for obj in fixed:
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    print(f"\nClean JSONL written to {OUTPUT_FILE}")


if __name__ == "__main__":
    repair_jsonl()

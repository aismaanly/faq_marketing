import json
from pathlib import Path
from collections import defaultdict
import os

JSON_PATH = "app/data/faq_baru.json"

# File untuk menyimpan log pertanyaan
COUNTER_FILE = Path("asked_counter.json")

def load_counter():
    if COUNTER_FILE.exists():
        with open(COUNTER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_counter(counter_dict):
    with open(COUNTER_FILE, "w", encoding="utf-8") as f:
        json.dump(counter_dict, f, indent=2, ensure_ascii=False)

def increment_question_counter(meta_question: str):
    counter = load_counter()
    if meta_question in counter:
        counter[meta_question] += 1
    else:
        counter[meta_question] = 1
    save_counter(counter)

def get_all_counts():
    return load_counter()

def load_json():
    if not os.path.exists(JSON_PATH):
        return []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data):
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

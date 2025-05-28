import os
import base64
import json, re
import numpy as np
import imghdr
import pandas as pd
from tqdm import tqdm


def read_jsonl(file_path: str) -> list[any]:
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def store_jsonl(file_path: str, data: list[any]) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_json(file_path: str) -> list[any] | dict:
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def store_json(file_path: str, data: list[any] | dict) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def pretty_print_json(data, key_color="green", value_color="reset"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "reset": "\033[0m",
        "blue": "\033[94m",
    }

    key_color_code = colors.get(key_color, colors["red"])
    value_color_code = colors.get(value_color, colors["reset"])

    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    pattern = r'(".*?")(\s*:\s*)(.*?)(,?\n?)'

    def replace_match(match):
        key = f"{key_color_code}{match.group(1)}{colors['reset']}"
        separator = match.group(2)
        value = f"{value_color_code}{match.group(3)}{colors['reset']}"
        return f"{key}{separator}{value}{match.group(4)}"

    formatted_str = re.sub(pattern, replace_match, json_str)

    print(formatted_str)


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["qid"])
        fout = open(path, "a")
        return fout, processed_results


def get_vec1_and_vec2_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) == 0 or len(vec2) == 0:
        raise ValueError("Input vectors must not be empty.")
    vec1_emb = np.array(vec1, dtype=float)
    vec2_emb = np.array(vec2, dtype=float)
    norm_vec1 = np.linalg.norm(vec1_emb)
    norm_vec2 = np.linalg.norm(vec2_emb)
    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ValueError("Input vectors must not be zero vectors.")
    vec1_emb = vec1_emb / norm_vec1
    vec2_emb = vec2_emb / norm_vec2
    similarity = np.dot(vec1_emb, vec2_emb)
    return float(similarity)


def get_base64_img(
    image_name: str,
    img_dir=str,
):
    file_path = os.path.join(img_dir, image_name)
    image_type = imghdr.what(file_path)

    with open(file_path, "rb") as img_file:
        img_data = img_file.read()
    b64_img = base64.b64encode(img_data).decode("utf-8")

    return image_type, b64_img


def get_query_category(query: str) -> str:
    categories = {
        "Atomic Queries": {"TextQ", "TableQ", "ImageQ", "ImageListQ"},
        "Composed Queries": {
            "Compose(TextQ,ImageListQ)",
            "Compose(TableQ,ImageListQ)",
            "Compose(ImageQ,TextQ)",
            "Compose(ImageQ,TableQ)",
            "Compose(TextQ,TableQ)",
            "Compose(TableQ,TextQ)",
        },
        "Intersection Queries": {
            "Intersect(ImageListQ,TextQ)",
            "Intersect(TableQ,TextQ)",
            "Intersect(ImageListQ,TableQ)",
        },
        "Comparison Queries": {
            "Compare(TableQ,Compose(TableQ,TextQ))",
            "Compare(Compose(TableQ,ImageQ),TableQ)",
            "Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))",
        },
    }

    for category, patterns in categories.items():
        if query in patterns:
            return category

    return "Unknown Category"


def extract_substring(
    long_string: str, start_string: str, end_string: str
) -> str | None:
    try:
        start_index = long_string.find(start_string)
        if start_index == -1:
            return None

        start_index += len(start_string)
        end_index = long_string.find(end_string, start_index)

        if end_index == -1:
            return None

        return long_string[start_index:end_index]
    except Exception as e:
        return None


def extract_json_from_text(text: str) -> list:
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
            return [parsed]
        except json.JSONDecodeError as e:
            print("JSON load fail:", e)
            return []
    else:
        print("not find JSON code block")
        return []


def load_json_from_parquet(parquet_path: str) -> dict[str, list[float]]:
    df = pd.read_parquet(parquet_path)
    data = {row["doc_id"]: row["embedding"] for _, row in df.iterrows()}
    return data

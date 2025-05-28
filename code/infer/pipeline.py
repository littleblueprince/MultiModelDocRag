from util.api import *
from util.prompt_template import *
import json5
import requests
import openai

filter_modal_name = ""
plan_path = ""
store_path = ""
log_path = ""
plan_data = read_jsonl(plan_path)


# replace with your actual paths
image_emb = load_json_from_parquet("")
table_emb = load_json_from_parquet("")
text_emb = load_json_from_parquet("")

# replace with your actual paths
info_dict = dict()
image_info = read_jsonl("")
for i in image_info:
    info_dict[i["id"]] = i
text_info = read_jsonl("")
for i in text_info:
    info_dict[i["id"]] = i
table_info = read_jsonl("")
for i in table_info:
    info_dict[i["id"]] = i


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)

    return cosine_sim


def llm_rewrite_query(query: str, chat_model: str, client, history_dict: dict) -> str:
    for key, value in history_dict.items():
        if key in query:
            query = query.replace(key, value)
    messages = [
        {
            "role": "system",
            "content": rewrite_query_prompt,
        },
        {
            "role": "user",
            "content": f"Original sub-query: {query}",
        },
    ]
    completion = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=messages,
    )
    return completion.choices[0].message.content


def llm_direct_answer(query: str, chat_model: str, client, history_dict: dict) -> str:
    for key, value in history_dict.items():
        if key in query:
            query = query.replace(key, value)
    messages = [
        {
            "role": "system",
            "content": direct_answer_prompt,
        },
        {
            "role": "user",
            "content": f"Question: {query}",
        },
    ]
    completion = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=messages,
    )
    return completion.choices[0].message.content


def get_img_filter_mesages(
    question: str,
    img_title: str,
    image_name: str,
) -> list:
    img_dir = ""  # Replace with your actual image directory
    image_type, b64_img = get_base64_img(image_name=image_name, img_dir=img_dir)
    system_inputs = []
    system_inputs.append({"type": "text", "text": filter_instruction})
    user_inputs = []
    user_inputs.append(
        {
            "type": "text",
            "text": f"The title of the candidate image is {img_title}. The content of the candidate image is ",
        }
    )
    user_inputs.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_type};base64,{b64_img}"},
        }
    )
    user_inputs.append({"type": "text", "text": f"**Question:**{question}"})
    messages = [
        {
            "role": "system",
            "content": system_inputs,
        },
        {
            "role": "user",
            "content": user_inputs,
        },
    ]
    return messages


def docs_filter(question: str, docs: list, filter_model: str, filter_client) -> list:
    filter_res = []
    for doc in docs:
        id = doc["id"]
        item = info_dict[id]
        title = item["title"]
        if "path" in item:
            path = item["path"].replace(item["path"].split(".")[1], "png")
            messages = get_img_filter_mesages(question, title, path)
        elif "table" in item:
            table = item["table"]
            table_name = item["table_name"]
            table_input_template = "**Title:**{title}\n**Table Name:**{table_name}\n**Table:**{table}\n**Question:**{question}"
            messages = [
                {
                    "role": "system",
                    "content": filter_instruction,
                },
                {
                    "role": "user",
                    "content": table_input_template.format(
                        question=question,
                        title=title,
                        table_name=table_name,
                        table=table,
                    ),
                },
            ]
        else:
            text = item["text"]
            text_input_template = "**Text Title:**{title}\n**Text content:**{text}\n**Question:**{question}"
            messages = [
                {
                    "role": "system",
                    "content": filter_instruction,
                },
                {
                    "role": "user",
                    "content": text_input_template.format(
                        question=question, title=title, text=text
                    ),
                },
            ]
        completion = filter_client.chat.completions.create(
            model=filter_model, messages=messages, temperature=0.1, top_p=0.001
        )
        output = completion.choices[0].message.content

        output = json5.loads(output)
        output["doc_id"] = id
        output["title"] = title
        filter_res.append(output)
    return sorted(filter_res, key=lambda x: x["score"], reverse=True)


def cal_query_docs_similarity(query: str, docs: list, topK: int) -> list[dict]:
    url = ""  # Replace with your actual embedding service URL, use embedding_server.py
    data = {"texts": [query]}
    response = requests.post(url, json=data)
    query_emb = response.json()["embeddings"][0]

    res = []
    for doc in docs:
        for dictory in [image_emb, table_emb, text_emb]:
            if doc in dictory:
                doc_emb = dictory[doc]
                similarity = cosine_similarity(query_emb, doc_emb)
                res.append({"id": doc, "similarity": similarity})
    sorted_res = sorted(res, key=lambda x: x["similarity"], reverse=True)
    return sorted_res[:topK]


def execute_node(
    name: str,
    query: str,
    action: str,
    history_dict: dict,
    text_filtered_ids: list,
    tabs_filtered_ids: list,
    imgs_filtered_ids: list,
    filter_client,
    llm_chat_client,
    chat_model,
    topK=5,
) -> tuple:
    for key, value in history_dict.items():
        if key in query:
            query = query.replace(key, value)
    if action == "text_retrieval":
        corpus = text_filtered_ids + tabs_filtered_ids
    elif action == "image_retrieval":
        corpus = imgs_filtered_ids
    elif action == "no_retrieval":
        corpus = []
    else:
        corpus = text_filtered_ids + tabs_filtered_ids + imgs_filtered_ids

    docs = cal_query_docs_similarity(query, corpus, topK)

    filter_res = docs_filter(
        question=query,
        docs=docs,
        filter_model=filter_modal_name,
        filter_client=filter_client,
    )
    filter_infos = []
    for m in filter_res:
        if m["score"] > 0:
            filter_infos.append(
                f'**Titile**{m["title"]}\n**Key_info**{m["key_information"]}'
            )

    # 将过滤后的信息和query包装发给llm获取结果，然后返回
    completion = llm_chat_client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": execute_node_prompt,
            },
            {
                "role": "user",
                "content": f'**question:**{query}**information:**{" ||| ".join(filter_infos)}**answer**',
            },
        ],
    )
    return completion.choices[0].message.content, filter_res


dataset_path = ""  # Replace with your actual dataset path

dataset = read_jsonl(dataset_path)
query_dict = dict()
for item in dataset:
    query_dict[item["question"]] = item


res = []

# Replace with your api keys, URLs and model names
llm_chat_api_key = ""
llm_chat_base_url = ""
llm_chat_chat_model = ""

filter_api_key = ""
filter_base_url = ""
filter_model = ""


f, processed = get_output_file(store_path, force=False)
filter_client = openai.Client(api_key=filter_api_key, base_url=filter_base_url)
llm_chat_client = openai.Client(api_key=llm_chat_api_key, base_url=llm_chat_base_url)


log = ""
for plan in tqdm(plan_data):
    try:
        question = (
            plan["prompt"]
            .split("user\n")[1]
            .replace("<|im_end|>\n<|im_start|>assistant\n", "")
        )
        plan_str = plan["predict"]
        query_item = query_dict[question]
        qid = query_item["qid"]
        if qid in processed:
            continue
        plan = json5.loads(plan_str)
        history_dict = dict()
        plan_log = []
        for node in plan:
            node_res, filter_res = execute_node(
                node["name"],
                node["query"],
                node["action"],
                history_dict=history_dict,
                text_filtered_ids=query_item["metadata"]["text_doc_ids"],
                tabs_filtered_ids=[query_item["metadata"]["table_id"]],
                imgs_filtered_ids=query_item["metadata"]["image_doc_ids"],
                filter_client=filter_client,
                llm_chat_client=llm_chat_client,
                chat_model=llm_chat_chat_model,
            )

            if "I don't know" in node_res:
                node_res, filter_res = execute_node(
                    node["name"],
                    node["query"],
                    "general_retrieval",
                    history_dict=history_dict,
                    text_filtered_ids=query_item["metadata"]["text_doc_ids"],
                    tabs_filtered_ids=[query_item["metadata"]["table_id"]],
                    imgs_filtered_ids=query_item["metadata"]["image_doc_ids"],
                    filter_client=filter_client,
                    llm_chat_client=llm_chat_client,
                    chat_model=llm_chat_chat_model,
                )

            if "I don't know" in node_res:
                modified_query = llm_rewrite_query(
                    node["query"], llm_chat_chat_model, llm_chat_client, history_dict
                )
                node_res, filter_res = execute_node(
                    node["name"],
                    modified_query,
                    "general_retrieval",
                    history_dict=history_dict,
                    text_filtered_ids=query_item["metadata"]["text_doc_ids"],
                    tabs_filtered_ids=[query_item["metadata"]["table_id"]],
                    imgs_filtered_ids=query_item["metadata"]["image_doc_ids"],
                    filter_client=filter_client,
                    llm_chat_client=llm_chat_client,
                    chat_model=llm_chat_chat_model,
                )

            if "I don't know" in node_res:
                node_res = llm_direct_answer(
                    node["query"], llm_chat_chat_model, llm_chat_client, history_dict
                )
                filter_res = "llm answer"

            if "I don't know" in node_res:
                node_res = ""
                filter_res = "node skipped"

            node["answer"] = node_res
            node["filter_res"] = filter_res
            plan_log.append(node)
            history_dict[node["name"]] = node_res

        f.write(
            json.dumps(
                {
                    "qid": qid,
                    "question": query_item["question"],
                    "ground_truth": [a["answer"] for a in query_item["answers"]],
                    "supporting_context": query_item["supporting_context"],
                    "plan_log": plan_log,
                }
            )
            + "\n"
        )
        f.flush()
    except Exception as e:
        log += f"{qid}:{e}\n\n"
        print(f"{qid}:{e}")
        continue
f.close()
with open(log_path, "w") as f:
    f.write(log)

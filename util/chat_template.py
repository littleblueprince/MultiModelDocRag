# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 14:13
# @Author  : blue
# @Description :
import base64
import os


def change_extension_to_png(filename):
    base = os.path.splitext(filename)[0]
    new_filename = base + ".png"
    return new_filename


def get_base64_img(image_name: str, img_dir='/data1/zch/datasets/multimodalqa/final_dataset_images/'):
    with open(img_dir + image_name, "rb") as img_file:
        img_data = img_file.read()
    b64_img = base64.b64encode(img_data).decode("utf-8")
    return b64_img


def get_temp_messages_v1(question: str, corpus: list) -> list:
    instruction = "**Instruction:**  Answer the question accurately using the information from the reference documents."
    QA_prompt_template1 = "**Reference Documents:**\n"
    QA_prompt_template2 = """
    **Question:**  
    {question}
    """
    contexts = []
    contexts.append(
        {"type": "text", "text": QA_prompt_template1}
    )
    for doc in corpus:
        doc_type = doc.get('type')
        title = doc.get('title', 'No Title')
        content = doc.get('content', 'No Content')
        if doc_type == "img":
            contexts.extend([
                {"type": "text", "text": title},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}}
            ])
        else:
            contexts.append({"type": "text", "text": f'**title**:{title} **text**:{content}'})
    contexts.append(
        {"type": "text", "text": QA_prompt_template2.format(question=question)}
    )
    messages = [
        {
            "role": "system",
            # "content":  [{"type": "text", "text": instruction}],
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": contexts,
        }
    ]

    return messages


def get_train_QA_prompt(question: str, corpus: list, ground_truth=None) -> dict:
    instruction = "**Instruction:**  Answer the question accurately using the information from the reference documents."
    QA_prompt_template = """
**Reference Documents:**  
{docs}  

**Question:**  
{question}
"""
    images = []
    docs = ""
    for doc in corpus:
        doc_type = doc.get('type')
        title = doc.get('title', 'No Title')
        content = doc.get('content', 'No Content')
        if doc_type == "img":
            path = doc.get('path', None)
            images.append('/mnt/sdb/zch/LLaMA-Factory/data/qwen2_vl_ft/mllm_data/' +
                          change_extension_to_png(path))
            docs += f'**title**:{title} **content**:<image>'
        elif doc_type == "tab":
            table_name = doc.get('table_name', None)
            docs += f'**title**:{title} **content**:{table_name}:{content}'
        else:
            docs += f'**title**:{title} **content**:{content}'
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {
            "role": "user",
            "content": QA_prompt_template.format(question=question, docs=docs),
        }
    ]
    if ground_truth:
        messages.append(
            {
                "role": "assistant",
                "content": str(ground_truth),
            }
        )
    return {
        "messages": messages,
        "images": images,
    }


def get_normal_QA_prompt(question: str, corpus: list) -> list:
    instruction = "**Instruction:**  Answer the question accurately using the information from the reference documents."
    user_inputs = []
    user_inputs.append({"type": "text", "text": instruction})
    user_inputs.append({"type": "text", "text": "**Reference Documents:**  \n"})
    for doc in corpus:
        text = doc['text']
        if 'file_path' in doc:
            file_path = doc['file_path']
            user_inputs.append({"type": "text", "text": text})
            user_inputs.append({
                "type": "image_url",
                "image_url":
                    {
                        "url": f"data:image/jpeg;base64,{get_base64_img(file_path)}"
                    }
            }
            )
        else:
            user_inputs.append({"type": "text", "text": text})

    user_inputs.append({"type": "text", "text": f"**Question:**  \n{question}"})
    messages = [
        {
            "role": "user",
            "content": user_inputs,
        }
    ]
    return messages


def get_normal_QA_prompt_v1(question: str, corpus: list) -> list:
    instruction = "**Instruction:**  Answer the question accurately using the information from the reference documents."
    user_inputs = []
    user_inputs.append({"type": "text", "text": instruction})
    user_inputs.append({"type": "text", "text": "**Reference Documents:**  \n"})
    for doc in corpus:
        text = doc['content']
        if 'path' in doc['metadata']:
            file_path = doc['metadata']['path']
            user_inputs.append({"type": "text", "text": text})
            user_inputs.append({
                "type": "image_url",
                "image_url":
                    {
                        "url": f"data:image/jpeg;base64,{get_base64_img(file_path)}"
                    }
            }
            )
        else:
            user_inputs.append({"type": "text", "text": text})

    user_inputs.append({"type": "text", "text": f"**Question:**  \n{question}"})
    messages = [
        {
            "role": "user",
            "content": user_inputs,
        }
    ]
    return messages


def get_normal_QA_prompt_v2(original_question: str, sub_question: str, corpus: list) -> list:
    instruction = "**Instruction:**  Extract content from the reference information that helps answer the question."
    user_inputs = []
    user_inputs.append({"type": "text", "text": instruction})
    user_inputs.append({"type": "text", "text": "**Reference Documents:**  \n"})
    for doc in corpus:
        text = doc['content']
        if 'path' in doc['metadata']:
            file_path = doc['metadata']['path']
            user_inputs.append({"type": "text", "text": text})
            user_inputs.append({
                "type": "image_url",
                "image_url":
                    {
                        "url": f"data:image/jpeg;base64,{get_base64_img(file_path)}"
                    }
            }
            )
        else:
            user_inputs.append({"type": "text", "text": text})

    user_inputs.append({"type": "text", "text": f"**Global Question:**  \n{original_question}"})
    user_inputs.append({"type": "text", "text": f"**Local Question:**  \n{sub_question}"})
    messages = [
        {
            "role": "user",
            "content": user_inputs,
        }
    ]
    return messages


def get_normal_QA_prompt_v3(original_question: str, sub_question: str, history: str, corpus: list) -> list:
    instruction = "**Instruction:**  Extract content from the reference information that helps answer the question."
    user_inputs = []
    user_inputs.append({"type": "text", "text": instruction})
    user_inputs.append({"type": "text", "text": "**Reference Documents:**  \n"})

    user_inputs.append({"type": "text", "text": f"**\n{history}\n"})
    for doc in corpus:
        text = doc['content']
        if 'path' in doc['metadata']:
            file_path = doc['metadata']['path']
            user_inputs.append({"type": "text", "text": text})
            user_inputs.append({
                "type": "image_url",
                "image_url":
                    {
                        "url": f"data:image/jpeg;base64,{get_base64_img(file_path)}"
                    }
            }
            )
        else:
            user_inputs.append({"type": "text", "text": text})

    user_inputs.append({"type": "text", "text": f"**Global Question:**  \n{original_question}"})
    user_inputs.append({"type": "text", "text": f"**Local Question:**  \n{sub_question}"})
    messages = [
        {
            "role": "user",
            "content": user_inputs,
        }
    ]
    return messages


def get_dag_query_prompt_v1(question: str, examples: list):
    instruction = """
**Instruction:**

Develop an **abstract and concise query plan** to address the given query. The plan should align with the query type (progressive, comparative, or global summary) and adhere to the following structure:

1. **Query Plan Structure**:
   - **Name**: Unique identifier for the query (e.g., Q1, Q2).
   - **Query**: A concise description of the query being addressed.
   - **Type**: Categorize as "retrieval" (fetching facts) or "reasoning" (analyzing or comparing).
   - **Dependencies**: List dependencies (other query nodes) that this query builds upon.

2. **Guidelines**:
   - For **global summary queries**, provide a single high-level query summarizing the intent.
   - For **comparative queries**, create two retrieval nodes for fact collection, followed by one reasoning node for comparison.
   - For **progressive queries**, start with a retrieval node for the initial fact and add a reasoning node to build upon it.

**Requirements**:
- Ensure simplicity and clarity in query plans.
- Avoid unnecessary details while maintaining the logical flow.

### Example 1:  
**Question:**  
{question1}

**Output:**
{dag1}

### Example 2:  
**Question:**  
{question2}

**Output:**
{dag2}

### Example 3:  
**Question:**  
{question3}

**Output:**
{dag3}

**Question:**  
{question}

**Output:**
"""
    messages = [
        {
            "role": "user",
            "content": instruction.format(
                question=question,
                question1=examples[0]['question'],
                dag1=examples[0]['dag'],
                question2=examples[1]['question'],
                dag2=examples[1]['dag'],
                question3=examples[2]['question'],
                dag3=examples[2]['dag'],
            ),
        }
    ]
    return messages


def get_train_query_plan_prompt(question: str, answer: str, type: str):
    #     instruction = f"""Your task is to generate a Directed Acyclic Graph (DAG) query plan for the given question. The plan should be as simple as possible while maintaining logical clarity. Use the provided question type to guide the decomposition. Avoid overthinking or over-decomposing the question, and ensure the queries accurately reflect the original intent.
    #
    # ---
    #
    # ### **Guidelines**
    #
    # 1. **Node Structure:**
    #    Each node in the query plan should be structured as follows:
    #    {{
    #      "name": "Qx",             // A unique identifier for the node
    #      "query": "<content>",     // The content of the query for this node
    #      "dependencies": []        // A list of names of other nodes this query depends on
    #    }}
    #
    # 2. **Dependencies:**
    #    - If a query depends on the results of another query, list the dependent nodes in the **dependencies** field.
    #    - Clearly reference the dependencies in the query content, e.g., "based on Q1."
    #
    # 3. **Keep It Simple:**
    #    - Avoid introducing unnecessary subqueries or complex reasoning.
    #    - Focus on the core logic required to answer the question and minimize the number of nodes.
    #
    # 4. **Examples:**
    #
    # #### **Example 1: Sequential Query (Bridge)**
    # - **Question:**
    #   Who painted the Mona Lisa, and what year did they die?
    # - **Answer:**
    #   Leonardo da Vinci, 1519
    # - **DAG Plan:**
    #   {{
    #     "name": "Q1",
    #     "query": "Who painted the Mona Lisa?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q2",
    #     "query": "When did Leonardo da Vinci die (based on Q1)?",
    #     "dependencies": ["Q1"]
    #   }}
    #
    # ---
    #
    # #### **Example 2: Comparative Query (Compare)**
    # - **Question:**
    #   Who won more Ballon d'Or awards, Messi or Ronaldo?
    # - **Answer:**
    #   Messi
    # - **DAG Plan:**
    #   {{
    #     "name": "Q1",
    #     "query": "How many Ballon d'Or awards has Messi won?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q2",
    #     "query": "How many Ballon d'Or awards has Ronaldo won?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q3",
    #     "query": "Compare the answers of Q1 and Q2 to determine who won more Ballon d'Or awards.",
    #     "dependencies": ["Q1", "Q2"]
    #   }}
    #
    # ---
    #
    # #### **Example 3: Global Query**
    # - **Question:**
    #   Summarize the causes of World War I.
    # - **Answer:**
    #   A combination of militarism, alliances, imperialism, and nationalism led to World War I.
    # - **DAG Plan:**
    #   {{
    #     "name": "Q1",
    #     "query": "Summarize the causes of World War I based on historical context.",
    #     "dependencies": []
    #   }}
    #
    # ---
    #
    # #### **Example 4: Mixed Queries (Bridge + Compare)**
    # - **Question:**
    #   Was the Eiffel Tower built before or after the construction of the Statue of Liberty?
    # - **Answer:**
    #   After
    # - **DAG Plan:**
    #   {{
    #     "name": "Q1",
    #     "query": "When was the Eiffel Tower built?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q2",
    #     "query": "When was the Statue of Liberty constructed?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q3",
    #     "query": "Compare the construction dates of the Eiffel Tower and the Statue of Liberty based on Q1 and Q2.",
    #     "dependencies": ["Q1", "Q2"]
    #   }}
    #
    # ---
    #
    # ### **Important Notes**
    # - **Avoid Overthinking:** Focus on solving the question directly without introducing unnecessary complexity.
    # - **Maintain Simplicity:** Only include the essential nodes required to address the query, and ensure dependencies are clear.
    #
    # **Now, based on these guidelines and examples, generate a DAG plan for the following question and answer:**
    #
    # **Question:**
    # {question}
    # **Question Type:**
    # {type}
    # **Answer:**
    # {answer}
    #
    # **Generate only the DAG plan in JSON format, without any additional explanation.**
    # """
    instruction = f"""Your task is to generate a Directed Acyclic Graph (DAG) query plan for the given question. The plan should be as simple as possible while maintaining logical clarity. Use the provided question type to guide the decomposition. Avoid overthinking or over-decomposing the question, and ensure the queries accurately reflect the original intent.

---

### **Guidelines**

1. **Node Structure:**  
   Each node in the query plan should be structured as follows:  
   {{
     "name": "Qx",             // A unique identifier for the node
     "query": "<content>",     // The content of the query for this node
     "dependencies": []        // A list of names of other nodes this query depends on
     "action": ""        // One of the defined retrieval action types (text_retrieval, image_retrieval, general_retrieval).
   }}

2. **Dependencies:**  
   - If a query depends on the results of another query, list the dependent nodes in the **dependencies** field.  
   - Clearly reference the dependencies in the query content, e.g., "based on Q1."
   - **action**: Defines the type of retrieval action. Choose from:
     - `text_retrieval`: For retrieving general knowledge from text or table-based data.
     - `image_retrieval`: For retrieving information about posters, portraits, colors, or other visual features.
     - `general_retrieval`: For cases where the retrieval type is ambiguous or requires a mix of text and image sources.

3. **Keep It Simple:**  
   - Avoid introducing unnecessary subqueries or complex reasoning.  
   - Focus on the core logic required to answer the question and minimize the number of nodes.

4. **Examples:**  
### **Examples**

#### **Example 1: Factual Query**
- **Question:**  
  Who painted the Mona Lisa, and what year did they die?  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "Who painted the Mona Lisa?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q2",
      "query": "When did Leonardo da Vinci die (based on Q1)?",
      "action": "text_retrieval",
      "dependencies": ["Q1"]
    }}
  ]

---

#### **Example 2: Comparative Query**
- **Question:**  
  Who won more Ballon d'Or awards, Messi or Ronaldo?  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "How many Ballon d'Or awards has Messi won?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q2",
      "query": "How many Ballon d'Or awards has Ronaldo won?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q3",
      "query": "Compare the answers of Q1 and Q2 to determine who won more Ballon d'Or awards.",
      "action": "general_retrieval",
      "dependencies": ["Q1", "Q2"]
    }}
  ]

---

#### **Example 3: Image-Based Query**
- **Question:**  
  Identify the artist of the painting in the provided image.  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "Analyze the provided image to identify the painting and its artist.",
      "action": "img_retrieval",
      "dependencies": []
    }}
  ]

---

#### **Example 4: Mixed Queries (Bridge + Compare)**
- **Question:**  
  Was the Eiffel Tower built before or after the construction of the Statue of Liberty?  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "When was the Eiffel Tower built?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q2",
      "query": "When was the Statue of Liberty constructed?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q3",
      "query": "Compare the construction dates of the Eiffel Tower and the Statue of Liberty based on Q1 and Q2.",
      "action": "general_retrieval",
      "dependencies": ["Q1", "Q2"]
    }}
  ]

---

#### **Example 5: Global Query**
- **Question:**  
  Summarize the causes of World War I.  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "Summarize the causes of World War I based on historical context.",
      "action": "text_retrieval",
      "dependencies": []
    }}
  ]

### **Important Notes**
- **Avoid Overthinking:** Focus on solving the question directly without introducing unnecessary complexity.  
- **Maintain Simplicity:** Only include the essential nodes required to address the query, and ensure dependencies are clear.  

**Now, based on these guidelines and examples, generate a DAG plan for the following question and answer:**  

**Question:**  
{question}  
**Question Type:**  
{type}  
**Answer:**  
{answer}  

**Generate only the DAG plan in JSON format, without any additional explanation.**  
"""
    messages = [
        {
            "role": "user",
            "content": instruction,
        },
    ]
    return messages


def get_train_query_plan_prompt_1_hop(question: str, answer: str):
    instruction = f"""Your task is to generate a Directed Acyclic Graph (DAG) query plan for the given question and answer. The plan should consist of a single node, as the question is standalone and does not require decomposition. Ensure the query accurately reflects the question and directly addresses the answer.

---

### **Guidelines**

1. **Node Structure:**  
   Each node in the query plan should be structured as follows:  
   ```json
   {{
     "name": "Q1",            // The unique identifier for the node
     "query": "<content>",    // The content of the query for this node
     "dependencies": []       // A list of names of other nodes (empty for single-node DAGs)
   }}
   ```

2. **Keep It Simple:**  
   - Focus on creating a single node that directly addresses the question and answer.  
   - Avoid introducing unnecessary dependencies or complex reasoning.  

---

### **Example**

#### **Example: Single-Node DAG**  
- **Question:**  
  Who painted the Mona Lisa?  
- **Answer:**  
  Leonardo da Vinci  
- **DAG Plan:**  
  ```json
  {{
    "name": "Q1",
    "query": "Who painted the Mona Lisa?",
    "dependencies": []
  }}
  ```

---

**Task:**  
Using the above example as a reference, generate a single-node DAG query plan for the following question and answer:  

**Question:**  
{question}  
**Answer:**  
{answer}  

**Output the DAG plan in JSON format, with no additional explanation.**  
    """
    messages = [
        {
            "role": "user",
            "content": instruction,
        },
    ]
    return messages


def get_dev_query_plan_prompt(question: str):
    #     instruction = f"""Your task is to decompose the provided question into a Directed Acyclic Graph (DAG) query plan. The plan should consist of minimal, logically coherent nodes, with each query accurately reflecting the question's intent. Use the examples provided below as references for the structure and dependencies.
    #
    # #### **Node Structure**
    # Each node should follow this format:
    # ```json
    # {{
    #   "name": "Qx",             // A unique identifier for the node
    #   "query": "<content>",     // The content of the query for this node
    #   "dependencies": []        // A list of names of other nodes this query depends on
    # }}
    # ```
    #
    # #### **Key Rules**
    # 1. **Dependencies:**
    #    - If a query depends on the result of another query, list the dependent nodes in the **dependencies** field.
    #    - Clearly reference the dependencies in the query content, e.g., "based on Q1."
    #
    # 2. **Simplification:**
    #    - Avoid overthinking or over-decomposing the question.
    #    - Keep the DAG plan simple and focused, ensuring that the queries align with the question's intent.
    #
    # 3. **Focus on Clarity:**
    #    - Each query should be logically consistent and straightforward.
    #    - Avoid introducing unnecessary complexity or steps.
    #
    # ---
    #
    # ### **Examples**
    #
    # #### **Example 1: Sequential Query (Bridge)**
    # - **Question:**
    #   Who painted the Mona Lisa, and what year did they die?
    # - **DAG Plan:**
    #   ```json
    #   {{
    #     "name": "Q1",
    #     "query": "Who painted the Mona Lisa?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q2",
    #     "query": "When did Leonardo da Vinci die (based on Q1)?",
    #     "dependencies": ["Q1"]
    #   }}
    #   ```
    #
    # ---
    #
    # #### **Example 2: Comparative Query (Compare)**
    # - **Question:**
    #   Who won more Ballon d'Or awards, Messi or Ronaldo?
    # - **DAG Plan:**
    #   ```json
    #   {{
    #     "name": "Q1",
    #     "query": "How many Ballon d'Or awards has Messi won?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q2",
    #     "query": "How many Ballon d'Or awards has Ronaldo won?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q3",
    #     "query": "Compare the answers of Q1 and Q2 to determine who won more Ballon d'Or awards.",
    #     "dependencies": ["Q1", "Q2"]
    #   }}
    #   ```
    #
    # ---
    #
    # #### **Example 3: Global Query**
    # - **Question:**
    #   Summarize the causes of World War I.
    # - **DAG Plan:**
    #   ```json
    #   {{
    #     "name": "Q1",
    #     "query": "Summarize the causes of World War I based on historical context.",
    #     "dependencies": []
    #   }}
    #   ```
    #
    # ---
    #
    # #### **Example 4: Mixed Queries (Bridge + Compare)**
    # - **Question:**
    #   Was the Eiffel Tower built before or after the construction of the Statue of Liberty?
    # - **DAG Plan:**
    #   ```json
    #   {{
    #     "name": "Q1",
    #     "query": "When was the Eiffel Tower built?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q2",
    #     "query": "When was the Statue of Liberty constructed?",
    #     "dependencies": []
    #   }}
    #   {{
    #     "name": "Q3",
    #     "query": "Compare the construction dates of the Eiffel Tower and the Statue of Liberty based on Q1 and Q2.",
    #     "dependencies": ["Q1", "Q2"]
    #   }}
    #   ```
    #
    # ---
    #
    # ### **Task**
    # Using the above examples as references, decompose the following question into a DAG query plan:
    #
    # **Question:**
    # {question}
    # """
    instruction = f"""Your task is to generate a Directed Acyclic Graph (DAG) query plan for the given question. The plan should be as simple as possible while maintaining logical clarity. Use the provided question type to guide the decomposition. Avoid overthinking or over-decomposing the question, and ensure the queries accurately reflect the original intent.

---

### **Guidelines**

1. **Node Structure:**
   - **name**: A unique identifier for the node, e.g., "Q1", "Q2".
   - **query**: A description of the query content for this node.
   - **dependencies**: A list of node names this query depends on.
   - **action**: Defines the type of retrieval action. Possible values are:
     - `text_retrieval`: Used for retrieving information from textual or tabular sources.
     - `image_retrieval`: Used for retrieving information from image sources (e.g., posters, photographs, artwork).
     - `general_retrieval`: Used when a combination of text and image sources is required.

2. **Action Type Examples:**
   - `text_retrieval`: "Extract Messi's Ballon d'Or wins from a table."
   - `image_retrieval`: "Identify the artist of a painting from an image."
   - `general_retrieval`: "Compare Eiffel Tower's construction timeline (text) with its design blueprint (image)."

3. **Dependencies:**
   - Explicitly list nodes this query depends on in the `dependencies` field.

4. **Keep It Simple:**
   - Avoid unnecessary complexity. Each query should represent the minimal step to achieve the answer.

5. **Examples:**  
### **Examples**

#### **Example 1: Factual Query**
- **Question:**  
  Who painted the Mona Lisa, and what year did they die?  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "Who painted the Mona Lisa?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q2",
      "query": "When did Leonardo da Vinci die (based on Q1)?",
      "action": "text_retrieval",
      "dependencies": ["Q1"]
    }}
  ]

---

#### **Example 2: Comparative Query**
- **Question:**  
  Who won more Ballon d'Or awards, Messi or Ronaldo?  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "How many Ballon d'Or awards has Messi won?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q2",
      "query": "How many Ballon d'Or awards has Ronaldo won?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q3",
      "query": "Compare the answers of Q1 and Q2 to determine who won more Ballon d'Or awards.",
      "action": "general_retrieval",
      "dependencies": ["Q1", "Q2"]
    }}
  ]

---

#### **Example 3: Image-Based Query**
- **Question:**  
  Identify the artist of the painting in the provided image.  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "Analyze the provided image to identify the painting and its artist.",
      "action": "img_retrieval",
      "dependencies": []
    }}
  ]

---

#### **Example 4: Mixed Queries (Bridge + Compare)**
- **Question:**  
  Was the Eiffel Tower built before or after the construction of the Statue of Liberty?  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "When was the Eiffel Tower built?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q2",
      "query": "When was the Statue of Liberty constructed?",
      "action": "text_retrieval",
      "dependencies": []
    }},
    {{
      "name": "Q3",
      "query": "Compare the construction dates of the Eiffel Tower and the Statue of Liberty based on Q1 and Q2.",
      "action": "general_retrieval",
      "dependencies": ["Q1", "Q2"]
    }}
  ]

---

#### **Example 5: Global Query**
- **Question:**  
  Summarize the causes of World War I.  
- **DAG Plan:**  
  [
    {{
      "name": "Q1",
      "query": "Summarize the causes of World War I based on historical context.",
      "action": "text_retrieval",
      "dependencies": []
    }}
  ]

### **Important Notes**
- **Avoid Overthinking:** Focus on solving the question directly without introducing unnecessary complexity.  
- **Maintain Simplicity:** Only include the essential nodes required to address the query, and ensure dependencies are clear.  

**Now, based on these guidelines and examples, generate a DAG plan for the following question and answer:**  

**Question:**  
{question}  

**Generate only the DAG plan in JSON format, without any additional explanation.**  
"""

    messages = [
        {
            "role": "user",
            "content": instruction,
        },
    ]
    return messages


def get_dev_query_plan_prompt_few_shot(question: str, examples: list):
    examples_str = ""
    for obj in examples:
        examples_str += f"- **Question:**  {obj['question']}\n - **DAG Plan:**  {obj['dag']}\n "
    instruction = f"""Decompose the provided question into an **abstract and high-level Directed Acyclic Graph (DAG) query plan**. The plan should consist of logically coherent nodes that address the key aspects of the question without delving into excessive details or unnecessary intermediate steps. The focus is on **concise abstraction** and maintaining logical clarity.

---

#### **Node Structure**  
Each node in the DAG should follow this format:


{{
  "name": "Qx",             // A unique identifier for the node
  "query": "<content>",     // The content of the query for this node
  "dependencies": []        // A list of names of other nodes this query depends on
}}


---

### **Examples**
{examples_str}

---

- **Question:**  
{question}

- **DAG Plan:**  
"""
    messages = [
        {
            "role": "user",
            "content": instruction,
        },
    ]
    return messages


def get_dev_query_plan_prompt_few_shot_v1(question: str, examples: list):
    examples_str = ""
    for obj in examples:
        examples_str += f"- **Question:**  {obj['question']}\n - **DAG Plan:**  {obj['dag']}\n "
    examples_str = examples_str.replace('json', '').replace('`', '')
    instruction = f"""Refer to the following example to construct a clear and concise query plan directed acyclic graph for the input question.

### **Examples**
{examples_str}

---

- **Question:**  
{question}

- **DAG Plan:**  
"""
    messages = [
        {
            "role": "user",
            "content": instruction,
        },
    ]
    return messages

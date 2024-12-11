# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 14:13
# @Author  : blue
# @Description :

def parse_corpus_to_messages(question: str, milvus_corpus: list) -> list:
    # 根据milvus_corpus生成messages
    inputs = []
    for doc in milvus_corpus:
        doc_type = doc.get('type')
        title = doc.get('title', 'No Title')
        content = doc.get('content', 'No Content')
        if doc_type == "img":
            inputs.extend([
                {"type": "text", "text": title},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}}
            ])
        else:
            inputs.append({"type": "text", "text": f'**title**:{title} **text**:{content}'})
    messages = [
        {
            "role": "system",
            "content": f"you are a helpful answer generator. Answer the question based on the information. **question**: {question}",
        },
        {
            "role": "user",
            "content": inputs,
        }
    ]
    return messages


def get_node1_messages_v1(question: str, contexts: list[str]) -> list:
    # 根据原始query获取子查询的messages
    prompt_template = """
### **Task**:

Analyze the original query and knowledge contexts to determine whether the query can be directly answered. If sufficient information is available, provide a direct answer. If not, generate a refined sub-query and propose a hypothetical answer based on the given information.

#### **Input**:

```plaintext
- Question: {question}
- Knowledge Contexts: {contexts}
```

#### **Instructions**:

1. **Determine Answerability**: Decide whether the query can be answered using the provided contexts.
2. **Response Based on Outcome**:
   - If **answerable**, provide a clear and concise direct answer.
   - If **not answerable**, provide:
     - A refined sub-query to address the information gap.
     - A hypothetical answer based on reasoning from the given contexts.

#### **Output Format**:

```plaintext
1. **Answerability**:
   - Can the query be directly answered: {{Yes/No}}

2. **Response**:
   - If "Yes":
     **Direct Answer**: {{Provide the answer based on the contexts}}
   - If "No":
     **Next Sub-Query**: {{Refined query to address the gap}}
     **Hypothetical Answer**: {{A reasonable guess based on the contexts}}
```
    """
    messages = [
        {
            "role": "user",
            "content": prompt_template.format(
                contexts="\n".join(contexts), question=question),
        }
    ]
    return messages


def get_node2_messages_v1(sub_question: str, corpus: list) -> list:
    # 根据sub query以及milvus检索到的值获取messages
    prompt_template_prefix = """
Your task is to answer the following questions clearly and accurately based on the context provided. Please strictly follow the following requirements:

1. Use only the information in the context to answer. Do not add any speculation or external knowledge not mentioned in the context.

2. If the context does not provide enough information to answer the question, please clearly reply: "No relevant information is provided in the context."

3. Make sure your answer is concise and covers the key points of the question.

**Input**:
```plaintext
Context:
"""
    prompt_template_suffix = """
Question:
{question}
```

**Output**:
```plaintext
Answer:
{{Specific answer based on the context, or "No relevant information is provided in the context"}}
```
    """
    contexts = []
    contexts.append({"type": "text", "text": prompt_template_prefix})
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
    contexts.append({"type": "text", "text": prompt_template_suffix.format(question=sub_question)})
    messages = [
        {
            "role": "user",
            "content": contexts,
        }
    ]
    return messages


def get_node3_messages_v1(sub_question: str, evidence1: str, evidence2: str):
    prompt_template = """
**Task**: Evaluate and compare the two pieces of evidence to provide a final answer based on the question. Follow these rules:

1. **Conflict Resolution**: If Evidence 1 and Evidence 2 contain conflicting information, Evidence 2 takes precedence.  
2. **Information Completeness**: If relevant information is missing in Evidence 2, rely on Evidence 1.  
3. **Answer Requirements**: Your answer must be concise, directly address the question, and avoid unnecessary elaboration.

---

**Input**:
```plaintext
Question: {question}

Evidence 1: {evidence1}

Evidence 2: {evidence2}
```

**Output**:
```plaintext
Final Answer: 
```
"""
    messages = [
        {
            "role": "user",
            "content": prompt_template.format(
                question=sub_question, evidence1=evidence1, evidence2=evidence2),
        }
    ]
    return messages


def get_final_messages_v1(question: str, contexts: list[str]) -> list:
    # 获取最终基于上下文和question进行问答的messages
    prompt_template = """
**Prompt Template**:

Your task is to answer the given question using the provided reference document as the primary source. If the reference does not contain enough information, you may supplement the answer with your own knowledge. If the answer cannot be determined from both sources, reply with: "I don't know."

---

**Input**:
```plaintext
Reference Document:
{contexts}

Question:
{question}
```

**Output**:
```plaintext
Answer:
{{answer based on the reference document and/or your knowledge, or "I don't know."}}
```
    """
    messages = [
        {
            "role": "user",
            "content": prompt_template.format(
                contexts="\n".join(contexts), question=question),
        }
    ]
    return messages

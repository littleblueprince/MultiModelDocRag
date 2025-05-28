
filter_instruction = f"""
Task: Given an input question and a set of candidate information, your task is to analyze whether the candidate information is helpful in answering the question and assess its relevance. Please output the following:

Relevance Score: Provide a score from 0 to 5, representing how relevant the candidate information is to the question, where 0 means completely irrelevant and 5 means highly relevant.
Key Information: If the candidate information is helpful in answering the question, extract and list the key information. If it is not helpful, leave this empty.
"""
filter_instruction_4o = f"""
Task: Given an input question and a set of candidate information, your task is to analyze whether the candidate information is helpful in answering the question and assess its relevance. Please output the following:

Relevance Score: Provide a score from 0 to 5, representing how relevant the candidate information is to the question, where 0 means completely irrelevant and 5 means highly relevant.
Key Information: If the candidate information is helpful in answering the question, extract and list the key information. If it is not helpful, leave this empty.

Example
    Question: What are the points of the Reds in the 2002 Super 12 season table?
    Title: 2002 Super 12 season
    Table:"|  | Team | Pld | W | D | L | PF | PA | PD | BP | Pts |\n|---|---|---|---|---|---|---|---|---|---|---|\n| 1 | Crusaders | 11 | 11 | 0 | 0 | 469 | 264 | +205 | 7 | 51 |\n| 2 | Waratahs | 11 | 8 | 0 | 3 | 337 | 284 | +53 | 7 | 39 |\n| 3 | Brumbies | 11 | 7 | 0 | 4 | 374 | 230 | +144 | 10 | 38 |\n| 4 | Highlanders | 11 | 8 | 0 | 3 | 329 | 207 | +122 | 6 | 38 |\n| 5 | Reds | 11 | 7 | 0 | 4 | 336 | 287 | +49 | 6 | 34 |\n| 6 | Blues | 11 | 6 | 0 | 5 | 318 | 249 | +69 | 5 | 29 |\n| 7 | Stormers | 11 | 5 | 0 | 6 | 310 | 314 | 4 | 7 | 27 |\n| 8 | Chiefs | 11 | 4 | 0 | 7 | 323 | 341 | 18 | 8 | 24 |\n| 9 | Hurricanes | 11 | 5 | 0 | 6 | 232 | 317 | 85 | 3 | 23 |\n| 10 | Sharks | 11 | 4 | 0 | 7 | 221 | 309 | 88 | 3 | 19 |\n| 11 | Cats | 11 | 1 | 0 | 10 | 228 | 407 | 179 | 2 | 6 |\n| 12 | Bulls | 11 | 0 | 0 | 11 | 232 | 500 | 268 | 1 | 1 |\n"
    
    output:```json
    {{
        "score":5,
        "key_information": "| 5 | Reds | 11 | 7 | 0 | 4 | 336 | 287 | +49 | 6 | 34 |"
    }}```

"""

direct_answer_prompt = f"""You are an intelligent assistant."
    Answer the following question using only your internal knowledge. 
    If you don't know the answer, just reply with "I don't know."
"""

rewrite_query_prompt = f"""
    "You are a query optimization assistant. The user will provide a sub-query that returned no relevant results. "
    "Your task is to revise or improve the query to increase its retrieval effectiveness. "
    "Only return the improved query. Do not include any explanation, reasoning, or formatting."
"""


execute_node_prompt = """Answer the query based on the information. 
    Requirement:
    1.Don't return any explanation.
    2.Don't use your own knowledge to user the question just answer it based the given info.
    3.just reuturn the answer briefly and accurately
    4.If you can't find the answer in the information, please just return "I don't know"

    **Example**
        **question** when was the last nintendo 64 game released
        **information**  **Titile**Bomberman 64 (1997 video game)\n**Key_info**Bomberman 64 was released for the Nintendo 64 in North America and Europe on November 30, 1997.||| **Title**List of Nintendo 64 games \n**Key_info** The last game published for the Nintendo 64 was Tony Hawk's Pro Skater 3 on August 20, 2002.|||**Title** Nintendo 64\n**Key_info**The Nintendo 64 was discontinued in early-mid 2002.
        **answer** August 20, 2002
    """

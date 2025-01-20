from openai import OpenAI

base_url = "http://36.213.0.171:9997/v1/"
model = "qwen2.5-instruct"
api_key = 'test'
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Your task is to specify a question retrieval reasoning plan for the following question.**Question:**What is the duration of the flight from Dallas to the city with a long, arched bridge that competed in the 1997 Australia National Rugby Union team test match?\\n**Known Content:**\\n**Failure Operation Records:**According to query: Find the city where the long arch bridge that the Australian National Rugby League team played in the 1997 Test match is located. Image_retrieval failed",
        }
    ],
    model=model,
    temperature=0.7,
    top_p=0.8,
)

print(chat_completion.choices[0].message.content)


from mistralai.client import MistralClient, ChatCompletionResponse
import faiss

question = "Are we going to save democracy?"
question_embeddings = np.array([get_text_embedding(question)])

D, I = index.search(question_embeddings, k=2) # distance, index
retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

def run_mistral(user_message, model="mistral-medium-latest"):
    messages = [
        ChatCompletionResponse(role="user", content=user_message)
    ]
    chat_response = client.chat(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)

run_mistral(prompt)
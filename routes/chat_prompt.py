from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def get_prompt_template():
    system_prompt = (
            """You are a highly knowledgeable and helpful AI assistant.

        You can answer questions using:
        1. **External context** (retrieved documents), which may contain helpful or specific information.
        2. **Your internal expertise** in fields such as Machine Learning, Deep Learning, Natural Language Processing, and JavaScript.

        **Guidelines for answering:**
        - First, check if the retrieved context is relevant to the user's question.
        - If relevant, use it to improve or support your answer.
        - If the context is missing or unrelated, answer based on your **OWN knowledge**.
        - If you genuinely don't know the answer, respond with: "I don't know that."

        Context (if available):
        {context}

        User Question:
        {input}

        Answer:"""
        )
    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
    ])
    return prompt
        

def llm_model() -> ChatOllama:
    return ChatOllama(
        model="gemma3:1b",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
    )
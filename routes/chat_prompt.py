from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def get_prompt_template():
    system_prompt = (
                '''You are an AI assistant skilled in answering both context-based and knowledge-based questions.

        You have access to two types of information:
        1. **Your own knowledge** in areas like Machine Learning, Deep Learning, Natural Language Processing, and JavaScript.
        2. **External context** (retrieved documents) which may contain additional information for answering specific questions.

        Use the provided context to improve your answer when relevant. 
        If the context is irrelevant or missing, rely on your internal knowledge for answering.

        If you don't know the answer, just say: "I don't know that."

        Only answer questions related to ML, DL, NLP, or JavaScript. If the question is unrelated, respond with: "I'm only trained to answer questions about ML, DL, NLP, or JavaScript."

        Context:
        {context}

        Question: {input}
        Answer:
        '''
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
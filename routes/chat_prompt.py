from langchain.prompts import ChatPromptTemplate

def get_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        "You are a helpful assistant who knows all about Machine Learning, Deep Learning, and AI. " \
        "Answer the following question With the correct Context: " \
        "{input}"
    )


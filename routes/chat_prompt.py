from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API = os.getenv("GROQ_API")
Gemini = os.getenv("GEMINI_API")

def get_prompt_template(context,user_input):
    
    final_prompt = f"""
    You are a friendly and highly knowledgeable teacher assistant who helps students understand complex topics in a simple way.

    Use the **context below** if it is helpful. If the context doesn’t help, use your own expertise in fields like Machine Learning, Deep Learning, Natural Language Processing, and JavaScript.

    When you answer:
    - Use simple and clear language, as if you're explaining to a student.
    - Start with a short and clear **definition**.
    - Then, explain the **main idea or concept** step-by-step.
    - Add a simple **example** to help understanding.
    - End with a brief **summary** or key takeaway.
    - Do NOT mention whether you used the context or not.

    Context:
    {context}

    Student's Question:
    {user_input}

    Your Answer:
    """
    return final_prompt


def groq_prompt_template():
    system_prompt = (
            """You are a highly knowledgeable and helpful AI assistant.

        You can answer questions using:
        1. **External context** (retrieved documents), which may contain helpful or specific information.
        2. **Your internal expertise** in fields such as Machine Learning, Deep Learning, Natural Language Processing, and JavaScript.

        **Guidelines for answering:**
        When you answer:
        - Use simple and clear language, as if you're explaining to a student.
        - Start with a short and clear **definition**.
        - Then, explain the **main idea or concept** step-by-step.
        - Add a simple **example** to help understanding.
        - End with a brief **summary** or key takeaway.
        - Do NOT mention whether you used the context or not.

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
        

def llm_model():
    llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", temperature=0.2,google_api_key=Gemini,max_tokens=1000 , disable_streaming = False)
    return llm


def groq_llm():
    llm = ChatGroq(api_key=GROQ_API, model="llama3-8b-8192",streaming=True,temperature=0.2,max_tokens=1000)
    return llm
    



    
import json
from langchain_core.runnables import RunnableSequence


def stream_response(chain: RunnableSequence,user_input: str):
    final_response = ""
    for chunk in chain.stream({"input": user_input}):
        if "answer" in chunk:
            text = chunk["answer"]
            final_response += text
            # Stream each chunk as a JSON line
            yield json.dumps({"response": text}) + "\n"
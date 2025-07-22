import os
from langchain_groq import ChatGroq

def call_model(prompt, model_name):
    chat_groq = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=model_name,temperature=0.2)
    response = chat_groq.invoke(messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()

    

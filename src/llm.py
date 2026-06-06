from langchain_groq import ChatGroq

from .config import GROQ_MODEL, get_groq_api_key


def get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        api_key=get_groq_api_key(),
        model_name=GROQ_MODEL,
        temperature=temperature,
    )

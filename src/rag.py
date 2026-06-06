from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory

from .llm import get_llm
from .prompts import ANSWER_PROMPT, NORMAL_CHAT_PROMPT, QUESTION_REWRITE_PROMPT
from .sessions import load_chat_history, save_chat_history

_message_store: Dict[str, ChatMessageHistory] = {}


def _get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _message_store:
        _message_store[session_id] = load_chat_history(session_id)
    return _message_store[session_id]


def _persist_history(session_id: str) -> None:
    if session_id in _message_store:
        save_chat_history(session_id, _message_store[session_id])


def reset_history(session_id: str) -> None:
    if session_id in _message_store:
        del _message_store[session_id]


def get_normal_chat_chain() -> RunnableWithMessageHistory:
    llm = get_llm()
    chain = NORMAL_CHAT_PROMPT | llm
    return RunnableWithMessageHistory(
        chain,
        lambda session_id: _get_history(session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def get_rag_chain(retriever) -> RunnableWithMessageHistory:
    llm = get_llm()
    parser = StrOutputParser()

    def _rewrite_question(inputs):
        return (
            QUESTION_REWRITE_PROMPT
            | llm
            | parser
        ).invoke({"input": inputs["input"], "chat_history": inputs.get("chat_history", [])})

    def _answer(inputs):
        standalone_q = _rewrite_question(inputs)
        docs = retriever.invoke(standalone_q) if retriever else []
        if inputs.get("strict_mode") == "ON" and not docs:
            answer_text = "I don't know based on the provided sources."
        else:
            context_text = "\n\n".join(doc.page_content for doc in docs)
            answer_text = (
                ANSWER_PROMPT
                | llm
                | parser
            ).invoke(
                {
                    "context": context_text,
                    "strict_mode": inputs.get("strict_mode", "OFF"),
                    "input": standalone_q,
                }
            )
        return {"answer": answer_text, "context": docs}

    rag_chain = RunnableLambda(_answer)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: _get_history(session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def invoke_chain(
    chain: RunnableWithMessageHistory,
    user_input: str,
    session_id: str,
    extra_inputs: Dict | None = None,
    extra_config: Dict | None = None,
):
    payload = {"input": user_input}
    if extra_inputs:
        payload.update(extra_inputs)
    config = {"configurable": {"session_id": session_id}}
    if extra_config:
        config.update(extra_config)
    result = chain.invoke(
        payload,
        config=config,
    )
    _persist_history(session_id)
    return result

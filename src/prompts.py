from langchain_core.prompts import ChatPromptTemplate

QUESTION_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that reformulates follow-up questions into standalone questions "
            "using the chat history for context. Only return the standalone question.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assistant for question-answering tasks. Use only the provided context to answer. "
            "If the context is insufficient, reply exactly: \"I don't know based on the provided sources.\". "
            "If strict mode is off and you add any best-effort reasoning beyond the context, clearly label it as "
            "\"(Not sourced)\" within the answer. Keep answers concise and helpful.",
        ),
        (
            "human",
            "Context:\n{context}\n\nStrict mode: {strict_mode}\n\nQuestion: {input}",
        ),
    ]
)

NORMAL_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise, helpful chat assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

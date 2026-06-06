import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .config import APP_DESCRIPTION, APP_TITLE, TOP_K_MAX, TOP_K_MIN
from .utils import messages_to_dicts


def render_header():
    st.title(APP_TITLE)
    st.caption(APP_DESCRIPTION)


def sidebar_sessions(sessions, current_session_id):
    st.sidebar.subheader("Sessions")
    session_options = {s["name"]: s["id"] for s in sessions}
    selected_name = None
    if current_session_id:
        selected_name = next(
            (s["name"] for s in sessions if s["id"] == current_session_id), None
        )
    selected = st.sidebar.selectbox(
        "Select session",
        options=list(session_options.keys()),
        index=list(session_options.keys()).index(selected_name) if selected_name else 0
        if session_options
        else None,
    )
    selected_id = session_options.get(selected) if selected else None

    new_name = st.sidebar.text_input("Rename session", value=selected or "")
    col1, col2, col3 = st.sidebar.columns(3)
    new_clicked = col1.button("New", use_container_width=True)
    delete_clicked = col2.button("Delete", use_container_width=True)
    clear_clicked = col3.button("Clear chat", use_container_width=True)

    return {
        "selected_id": selected_id,
        "rename_to": new_name,
        "new_clicked": new_clicked,
        "delete_clicked": delete_clicked,
        "clear_clicked": clear_clicked,
    }


def sidebar_retrieval_controls(default_k: int, strict_mode: bool, show_context: bool):
    st.sidebar.subheader("Retrieval Controls")
    top_k = st.sidebar.slider("Top-K", min_value=TOP_K_MIN, max_value=TOP_K_MAX, value=default_k)
    strict = st.sidebar.checkbox("Strict mode", value=strict_mode, help="Refuse when context is insufficient.")
    show_ctx = st.sidebar.checkbox("Show retrieved context", value=show_context)
    return top_k, strict, show_ctx


def sidebar_source_actions():
    st.sidebar.subheader("Knowledge Base")
    rebuild = st.sidebar.button("Rebuild index", use_container_width=True)
    clear_sources = st.sidebar.button("Clear sources for this session", use_container_width=True)
    return rebuild, clear_sources


def render_source_inputs():
    with st.expander("Add knowledge sources", expanded=False):
        pdf_files = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True
        )
        youtube_raw = st.text_area(
            "YouTube links (comma or newline separated)",
            placeholder="https://youtu.be/..., https://www.youtube.com/watch?v=...",
        )
        web_raw = st.text_area(
            "Webpage links (comma or newline separated)",
            placeholder="https://example.com/article",
        )
    return pdf_files, youtube_raw, web_raw


def render_chat_history(messages: list[BaseMessage], citations_map: dict, show_context: bool):
    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
                if citations_map.get(idx):
                    st.markdown("**Sources**")
                    for doc in citations_map[idx]:
                        meta = doc.metadata or {}
                        label = meta.get("type", "Source")
                        source = meta.get("source") or meta.get("file_path") or "Unknown"
                        page = meta.get("page")
                        page_info = f" — page {page}" if page is not None else ""
                        st.write(f"- {label}: {source}{page_info}")
                    if show_context:
                        with st.expander("Show retrieved context"):
                            for doc in citations_map[idx]:
                                meta = doc.metadata or {}
                                st.markdown(f"**{meta.get('type', 'Source')}** - {meta.get('source', '')}")
                                st.write(doc.page_content)
        else:
            with st.chat_message("system"):
                st.write(msg.content)


def chat_history_to_dicts(messages):
    return messages_to_dicts(messages)

import streamlit as st

from src import config, indexing, loaders, rag, sessions, ui, utils


def init_state():
    config.ensure_data_dirs()
    existing_sessions = sessions.load_sessions()
    if not existing_sessions:
        first = sessions.create_session("Session 1")
        existing_sessions = [first]
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = existing_sessions[0]["id"]
    st.session_state.setdefault("top_k", config.TOP_K_DEFAULT)
    st.session_state.setdefault("strict_mode", False)
    st.session_state.setdefault("show_context", False)
    st.session_state.setdefault("vectorstores", {})
    st.session_state.setdefault("citations", {})
    return existing_sessions


def get_current_session(sessions_list):
    current_id = st.session_state.get("current_session_id")
    session = next((s for s in sessions_list if s["id"] == current_id), None)
    if session:
        return session
    if sessions_list:
        st.session_state["current_session_id"] = sessions_list[0]["id"]
        return sessions_list[0]
    new_session = sessions.create_session("Session 1")
    st.session_state["current_session_id"] = new_session["id"]
    return new_session


def refresh_sessions():
    return sessions.load_sessions()


def clear_local_state(session_id: str):
    rag.reset_history(session_id)
    st.session_state["vectorstores"].pop(session_id, None)
    st.session_state["citations"].pop(session_id, None)


def handle_session_sidebar(sessions_list):
    current_session = get_current_session(sessions_list)
    controls = ui.sidebar_sessions(sessions_list, current_session["id"])

    if controls["selected_id"] and controls["selected_id"] != current_session["id"]:
        st.session_state["current_session_id"] = controls["selected_id"]
        current_session = get_current_session(refresh_sessions())

    # Rename if changed
    if controls["rename_to"] and controls["rename_to"] != current_session["name"]:
        sessions.rename_session(current_session["id"], controls["rename_to"])
        sessions_list = refresh_sessions()
        current_session = get_current_session(sessions_list)

    if controls["new_clicked"]:
        new_session = sessions.create_session()
        sessions_list = refresh_sessions()
        st.session_state["current_session_id"] = new_session["id"]
        current_session = new_session

    if controls["delete_clicked"] and current_session:
        sessions.delete_session(current_session["id"])
        clear_local_state(current_session["id"])
        sessions_list = refresh_sessions()
        current_session = get_current_session(sessions_list)
        sessions_list = refresh_sessions()

    if controls["clear_clicked"] and current_session:
        sessions.clear_session_history(current_session["id"])
        clear_local_state(current_session["id"])
        st.success("Chat history cleared for this session.")

    return sessions_list, current_session


def load_index_for_session(session_id: str):
    cached = st.session_state["vectorstores"].get(session_id)
    if cached is not None:
        return cached
    store = indexing.load_faiss_index(session_id)
    st.session_state["vectorstores"][session_id] = store
    return store


def build_index_for_session(session: dict, uploaded_pdfs, youtube_links, web_links, force_rebuild: bool):
    existing_sources = session.get("sources", {"pdfs": [], "youtube": [], "web": []})
    docs = []
    pdf_paths = []

    # Existing PDFs on disk
    existing_pdf_docs, existing_pdf_paths = loaders.load_existing_pdfs(
        session["id"], existing_sources.get("pdfs", [])
    )
    docs.extend(existing_pdf_docs)
    pdf_paths.extend(existing_pdf_paths)

    # Newly uploaded PDFs
    if uploaded_pdfs:
        new_docs, new_paths = loaders.load_pdfs(session["id"], uploaded_pdfs)
        docs.extend(new_docs)
        pdf_paths.extend(new_paths)
        updated_pdf_files = existing_sources.get("pdfs", []) + [p.name for p in new_paths]
    else:
        updated_pdf_files = existing_sources.get("pdfs", [])

    youtube_merged = list(dict.fromkeys(existing_sources.get("youtube", []) + youtube_links))
    web_merged = list(dict.fromkeys(existing_sources.get("web", []) + web_links))

    sources_hash = utils.compute_sources_hash(pdf_paths, youtube_merged, web_merged)
    if not force_rebuild and sources_hash and sources_hash == session.get("sources_hash"):
        st.info("Sources unchanged; using existing index.")
        store = load_index_for_session(session["id"])
        return store

    try:
        yt_docs = loaders.load_youtube_transcripts(youtube_merged)
        web_docs = loaders.load_webpages(web_merged)
    except ValueError as exc:
        st.error(str(exc))
        return None
    docs.extend(yt_docs)
    docs.extend(web_docs)

    if not docs:
        st.warning("No sources available to build the index.")
        return None

    try:
        store = indexing.build_faiss_index(session["id"], docs)
    except indexing.IndexingError as exc:
        st.error(str(exc))
        return None

    if store:
        sessions.update_session_sources(
            session["id"],
            pdf_filenames=list(dict.fromkeys(updated_pdf_files)),
            youtube_links=youtube_merged,
            web_links=web_merged,
            sources_hash=sources_hash,
        )
        st.session_state["vectorstores"][session["id"]] = store
        st.session_state["citations"].pop(session["id"], None)
        st.success("Index rebuilt successfully.")
    return store


def render_sources_section(session: dict):
    pdf_files, youtube_raw, web_raw = ui.render_source_inputs()
    youtube_links = loaders.parse_links(youtube_raw)
    web_links = loaders.parse_links(web_raw)
    if st.button("Add / Update sources"):
        build_index_for_session(session, pdf_files, youtube_links, web_links, force_rebuild=False)
    return pdf_files, youtube_links, web_links


def render_sidebar_controls(session: dict):
    top_k, strict_mode, show_ctx = ui.sidebar_retrieval_controls(
        st.session_state["top_k"], st.session_state["strict_mode"], st.session_state["show_context"]
    )
    st.session_state["top_k"] = top_k
    st.session_state["strict_mode"] = strict_mode
    st.session_state["show_context"] = show_ctx
    rebuild, clear_sources_clicked = ui.sidebar_source_actions()
    return rebuild, clear_sources_clicked


def display_existing_sources(session: dict):
    sources = session.get("sources", {})
    has_sources = any([sources.get("pdfs"), sources.get("youtube"), sources.get("web")])
    if has_sources:
        st.markdown("**Current sources**")
        if sources.get("pdfs"):
            st.write("PDFs: " + ", ".join(sources.get("pdfs", [])))
        if sources.get("youtube"):
            st.write("YouTube: " + ", ".join(sources.get("youtube", [])))
        if sources.get("web"):
            st.write("Web: " + ", ".join(sources.get("web", [])))


def main():
    config.get_groq_api_key()
    sessions_list = init_state()
    sessions_list, current_session = handle_session_sidebar(sessions_list)
    rebuild_index_clicked, clear_sources_clicked = render_sidebar_controls(current_session)

    ui.render_header()
    display_existing_sources(current_session)
    pdf_files, youtube_links, web_links = render_sources_section(current_session)

    if clear_sources_clicked:
        sessions.clear_session_sources(current_session["id"])
        clear_local_state(current_session["id"])
        st.success("Sources cleared for this session.")
        sessions_list = refresh_sessions()
        current_session = get_current_session(sessions_list)

    if rebuild_index_clicked:
        build_index_for_session(current_session, pdf_files, youtube_links, web_links, force_rebuild=True)

    store = load_index_for_session(current_session["id"])
    retriever = indexing.get_retriever(store, st.session_state["top_k"])

    if retriever:
        st.info("RAG mode enabled for this session.")
    else:
        st.info("Normal chat mode (no sources indexed yet).")

    chat_prompt = "Ask a question..."
    chat_input = st.chat_input(chat_prompt)

    history = sessions.load_chat_history(current_session["id"]).messages
    citations_map = st.session_state["citations"].get(current_session["id"], {})
    ui.render_chat_history(history, citations_map, st.session_state["show_context"])

    if chat_input:
        if retriever:
            chain = rag.get_rag_chain(retriever)
            result = rag.invoke_chain(
                chain,
                chat_input,
                current_session["id"],
                extra_inputs={"strict_mode": "ON" if st.session_state["strict_mode"] else "OFF"},
            )
            store = st.session_state["vectorstores"].get(current_session["id"])
            if store is None:
                store = load_index_for_session(current_session["id"])
            history = sessions.load_chat_history(current_session["id"]).messages
            if current_session["id"] not in st.session_state["citations"]:
                st.session_state["citations"][current_session["id"]] = {}
            last_idx = len(history) - 1
            st.session_state["citations"][current_session["id"]][last_idx] = result.get("context", [])
        else:
            chain = rag.get_normal_chat_chain()
            rag.invoke_chain(chain, chat_input, current_session["id"])

        history = sessions.load_chat_history(current_session["id"]).messages
        citations_map = st.session_state["citations"].get(current_session["id"], {})
        ui.render_chat_history(history, citations_map, st.session_state["show_context"])


if __name__ == "__main__":
    main()

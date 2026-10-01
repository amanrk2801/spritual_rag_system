"""Dharma Sahayak – Streamlit chat UI for the spiritual RAG backend.

Run:  streamlit run frontend/streamlit_app.py
Env:  BACKEND_URL (default http://localhost:8000), PUBLIC_API_KEY (optional)
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("PUBLIC_API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

SUGGESTIONS = {
    ":orange[:material/water:] रामसेतु का भेद": "रामसेतु का भेद क्या है? पत्थर कैसे तैरे?",
    ":orange[:material/auto_stories:] 33 करोड़ सुरों का रहस्य": "33 करोड़ सुरों का रहस्य क्या है?",
    ":orange[:material/self_improvement:] Krishna & Sudama": "What is the secret of the Krishna–Sudama leela?",
    ":orange[:material/forest:] Baba Matang": "Baba Matang kaun hain aur unhone kya seekha?",
}

st.set_page_config(page_title="Dharma Sahayak", page_icon=":material/temple_hindu:", layout="centered")


# ------------------------------------------------------------------ backend helpers
@st.cache_resource
def http_client() -> httpx.Client:
    return httpx.Client(
        base_url=BACKEND_URL,
        headers=HEADERS,
        timeout=httpx.Timeout(connect=5, read=120, write=10, pool=5),
    )


@st.cache_data(ttl=60, show_spinner=False)
def fetch_books() -> list[dict] | None:
    try:
        r = http_client().get("/v1/books")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None


def stream_events(payload: dict) -> Iterator[dict]:
    with http_client().stream("POST", "/v1/chat/stream", json=payload) as r:
        if r.status_code != 200:
            r.read()
            detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith(
                "application/json"
            ) else r.text
            yield {"type": "error", "message": f"{r.status_code}: {detail}"}
            return
        for line in r.iter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])


def page_label(s: dict) -> str:
    a, b = s.get("page_start"), s.get("page_end")
    return f"p. {a}" if a == b else f"pp. {a}–{b}"


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/menu_book:"):
        for s in sources:
            heading = f"**[{s['n']}] {s['title']}** · {page_label(s)}"
            if s.get("section"):
                heading += f" · _{s['section']}_"
            st.markdown(heading)
            st.caption(s["text"][:700] + ("…" if len(s["text"]) > 700 else ""))


# ------------------------------------------------------------------ state
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "sources"?}

# ------------------------------------------------------------------ sidebar
books = fetch_books()
with st.sidebar:
    st.header(":material/temple_hindu: Dharma Sahayak")
    if books is None:
        st.error(f"Backend unreachable at {BACKEND_URL}", icon=":material/cloud_off:")
        selected_ids: list[str] | None = None
    else:
        st.badge(f"{len(books)} book(s) indexed", icon=":material/check_circle:", color="green")
        options = {b["doc_id"]: b["title"] for b in books}
        chosen = st.multiselect(
            "Search within",
            list(options),
            format_func=options.get,
            placeholder="All books",
        )
        selected_ids = chosen or None

    top_k = st.slider("Passages to consult", 3, 12, 6)
    show_sources = st.toggle("Show sources", value=True)
    if st.button("New conversation", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Answers are generated only from the indexed scriptures, with page citations.")

# ------------------------------------------------------------------ header
st.title("Dharma Sahayak")
st.caption("अपने आध्यात्मिक प्रश्न पूछें · Ask in Hindi, English or Hinglish")

# ------------------------------------------------------------------ history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            render_sources(msg["sources"])

prompt = st.chat_input("अपना प्रश्न लिखें… / Type your question…", submit_mode="disable")

if not st.session_state.messages and not prompt:
    picked = st.pills("Try asking", list(SUGGESTIONS), label_visibility="collapsed")
    if picked:
        prompt = SUGGESTIONS[picked]

# ------------------------------------------------------------------ new turn
if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        meta: dict = {}
        error: str | None = None
        payload = {"question": prompt, "history": history, "doc_ids": selected_ids, "top_k": top_k}

        try:
            events = stream_events(payload)
            with st.status(":shimmer[Searching the scriptures]", type="compact") as status:
                for ev in events:
                    if ev["type"] == "meta":
                        meta = ev
                        break
                    if ev["type"] == "error":
                        error = ev["message"]
                        break
                if error:
                    status.update(label="Search failed", state="error")
                else:
                    queries = meta.get("search_queries", [])
                    if queries:
                        st.markdown("**Search queries**\n" + "\n".join(f"- {q}" for q in queries))
                    status.update(
                        label=f"Consulted {len(meta.get('sources', []))} passages"
                        + (" (cached)" if meta.get("cached") else ""),
                        state="complete",
                    )

            def tokens() -> Iterator[str]:
                for ev in events:
                    if ev["type"] == "token":
                        yield ev["text"]
                    elif ev["type"] == "error":
                        yield f"\n\n:red[{ev['message']}]"
                    elif ev["type"] == "done":
                        meta["latency_ms"] = ev.get("latency_ms")

            answer = "" if error else st.write_stream(tokens())
        except httpx.HTTPError as exc:
            error, answer = f"Could not reach the backend: {exc}", ""

        if error:
            st.error(error, icon=":material/error:")
            answer = answer or f":red[{error}]"
        else:
            if meta.get("latency_ms") is not None:
                st.caption(f":material/timer: {meta['latency_ms'] / 1000:.1f}s")
            if show_sources:
                render_sources(meta.get("sources", []))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer if isinstance(answer, str) else str(answer),
         "sources": meta.get("sources", [])}
    )

"""Interface Streamlit — FOURNIE, FONCTIONNELLE dès la Brique 0.

Lancement :  streamlit run src/orange_bank_agent/ui/app.py
Utilise exactement le même `build_graph()` et le même runner que la CLI.
Après avoir modifié le code d'une brique, relancez Streamlit.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # permet de lancer sans `pip install`

import streamlit as st  # noqa: E402

from orange_bank_agent.config import settings  # noqa: E402
from orange_bank_agent.graph.agent_graph import build_graph  # noqa: E402
from orange_bank_agent.observability.tracing import flush_traces  # noqa: E402
from orange_bank_agent.runner import ask, resume  # noqa: E402
from orange_bank_agent.tools import tool_status  # noqa: E402

st.set_page_config(page_title="Orange Bank — Assistant", page_icon="🟧")


@st.cache_resource
def get_graph():
    return build_graph()


def new_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending = None


if "thread_id" not in st.session_state:
    new_conversation()


def to_message(result) -> dict:
    return {
        "role": "assistant",
        "content": result.answer or "⏳ En attente de votre confirmation.",
        "tools": [(c.label, c.status) for c in result.tool_calls],
        "sources": result.sources,
        "guardrails": result.guardrail_events,
    }


def render(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tools"):
            st.caption("Outils appelés")
            st.markdown("\n".join(
                f"- `{label}`" + (" ⚠️ erreur" if status == "error" else "")
                for label, status in message["tools"]
            ))
        for event in message.get("guardrails", []):
            st.caption(f"🛡 Garde-fou : {event}")
        if message.get("sources"):
            with st.expander(f"Sources citées ({len(message['sources'])})"):
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}** — score {src.get('score', 0):.2f}")
                    st.caption(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))


def run_turn(fn, *args) -> None:
    """Exécute un tour (ask/resume) en capturant les erreurs pour ne jamais planter l'UI."""
    try:
        with st.spinner("L'assistant réfléchit…"):
            result = fn(get_graph(), *args, st.session_state.thread_id)
        st.session_state.messages.append(to_message(result))
        st.session_state.pending = result.pending_confirmation
        flush_traces()
    except Exception as exc:
        st.session_state.pending = None
        st.session_state.messages.append(
            {"role": "assistant", "content": f"⚠️ Erreur : `{type(exc).__name__}: {exc}`"}
        )
    st.rerun()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Orange Bank")
    if st.button("Nouvelle conversation"):
        new_conversation()
        st.rerun()
    st.subheader("État du système")
    st.write(f"Modèle : `{settings.chat_model}`")
    st.write(f"Langfuse : {'✅ actif' if settings.langfuse_enabled else '⚪ désactivé'}")
    st.write("Outils :")
    for name, implemented in tool_status().items():
        st.write(f"{'✅' if implemented else '⚪'} `{name}`")
    if not settings.llm_configured:
        st.warning("LLM_API_KEY absente : copiez .env.example en .env.")

# ---------------------------------------------------------------- chat
st.title("🟧 Assistant Orange Bank")

for message in st.session_state.messages:
    render(message)

pending = st.session_state.pending
if pending:
    with st.chat_message("assistant"):
        st.warning(pending.get("message", "Confirmation requise avant d'exécuter cette action."))
        if pending.get("details"):
            st.json(pending["details"])
        col_yes, col_no = st.columns(2)
        if col_yes.button("✅ Confirmer", key="confirm"):
            run_turn(resume, True)
        if col_no.button("❌ Refuser", key="refuse"):
            run_turn(resume, False)

if prompt := st.chat_input("Posez votre question…", disabled=bool(pending)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    render(st.session_state.messages[-1])
    run_turn(ask, prompt)

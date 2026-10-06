"""The demo skin. It talks HTTP and nothing else.

THE ONE RULE THIS FILE OBEYS
    No regrag import. Not one. If this page could reach the pipeline directly,
    the API would stop being the product and become decoration - and the claim
    "a bank calls the API from their own systems" would be untested by the only
    client that exists. Everything here goes through localhost:8000, exactly as
    a customer's system would.

    A side effect worth knowing: this process loads no models, so Streamlit's
    re-run-the-whole-script-on-every-click behaviour costs nothing here. The
    expensive state lives in the API, loaded once at ITS startup.

WHY POLLING LOOKS LIKE THIS
    POST /ask returns at once with an id; this page then asks GET /ask/{id}
    every two seconds. The `stage` field is what fills the waiting time with
    something true instead of a spinner that says "Loading...".
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

# Read from the environment so Docker can point the UI at the API container
# (REGRAG_API_URL=http://api:8000); local runs keep the localhost default.
API = os.getenv("REGRAG_API_URL", "http://localhost:8000")
POLL_SECONDS = 2
GIVE_UP_AFTER = 600          # 10 min. CPU generation is slow, not infinite.

st.set_page_config(page_title="RegRAG", layout="wide")
st.title("Regulatory assistant")
st.caption(
    "Capital adequacy (Basel) · Stress testing (Basel, CCAR / DFAST) · Data quality "
    "(BCBS 239) · Model risk (SR 11-7, SS1/23) · IFRS 9 — every claim "
    "carries its source."
)


# ---------------------------------------------------------------------------
# HEALTH
#
# MEASURED 2026-09-14: /health takes ~16s, because the API asks Ollama for its
# model list (~5.8s) and Qdrant for readyz (~2.8s), and Streamlit re-runs this
# whole script on EVERY interaction. So the check was being paid again on every
# question, before a single pixel was drawn - that was the 7-8 second freeze and
# the grey fade after clicking Ask.
#
# (Those two numbers are themselves abnormal for localhost and are NOT explained
#  - suspected Docker/WSL2 connection cost, the same one open in the progress
#  file. Cached here, not fixed here.)
#
# TWO CHANGES, AND WHAT EACH COSTS
#   1. ttl=30 - at most one real check every 30 seconds.
#   2. On a SUBMIT rerun, skip it entirely and reuse the last known answer, so
#      asking a question never waits on it even if the cache has just expired.
# THE COST, STATED: for up to 30 seconds after Ollama or Qdrant dies, the banner
# still shows green. The failure then surfaces as a failed answer instead of a
# warning. Acceptable for a demo, NOT for anything a person relies on.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict | None:
    try:
        return requests.get(f"{API}/health", timeout=20).json()
    except Exception:                                          # noqa: BLE001
        return None


# Drawn before the form so the banner stays ABOVE the question box, but FILLED
# after it, once this run knows whether it is a submit.
health_box = st.container()


def show_health(h: dict | None, *, stale: bool) -> None:
    with health_box:
        if h is None:
            st.error(f"No API at {API}. "
                     f"Start it: `uv run uvicorn api.main:app --port 8000`")
            return
        for problem in h["problems"]:
            st.error(problem)
        if not h["problems"] and not h["warm"]:
            st.info("The API is still warming up (loading the embedder and the "
                    "corpus). The first answer will be slow if you ask now.")
    st.sidebar.write(f"**model** {h['model']}")
    st.sidebar.write(f"**gate** {'on' if h['gate'] else 'off'}")
    st.sidebar.write(f"**tracing** {'on' if h['tracing'] else 'off'}")
    st.sidebar.write(f"**warm** {h['warm']} ({h['warm_seconds']}s)")
    if stale:
        st.sidebar.caption("status as of the last check, not re-checked for "
                           "this question")


st.sidebar.subheader("Known limitations")
st.sidebar.caption(
    "**Cross-section questions.** If the answer spans several sections of one "
    "document, only the sections retrieved are used.\n\n"
    "**Cross-jurisdiction questions.** The model can blend rules from two "
    "regulators into one statement. A model limit, not a retrieval one.\n\n"
    "**30-60 seconds per answer.** The model runs on this machine, so nothing "
    "leaves it. Speed is the price of residency."
)


def _clear() -> None:
    """Drop the answer AND the question box.

    A widget's value can only be reset from a callback, before the widget is
    built again - assigning to st.session_state["q"] after the text_input has
    been drawn raises. Hence on_click, not a plain if-block.
    """
    st.session_state["result"] = None
    st.session_state["q"] = ""


# A FORM, NOT A LOOSE INPUT AND BUTTON.
# Outside a form, typing does not reach the server until the box loses focus,
# so the first click only COMMITS the text and spends itself on a rerun - the
# question is sent on the second click, and every ask pays that round trip.
# Inside a form, the text and the submit arrive together, and Enter submits.
with st.form("ask", clear_on_submit=False):
    question = st.text_input(
        "Question", key="q",
        placeholder="what does SR 11-7 require of model validation?")
    go = st.form_submit_button("Ask", type="primary")

# ONLY NOW is it known whether this run is a submit.
submitting = bool(go and question.strip())
if submitting:
    _h = st.session_state.get("health_last")          # no network call at all
else:
    _h = health()
    if _h is not None:
        st.session_state["health_last"] = _h
show_health(_h, stale=submitting and _h is not None)
if _h is None and not submitting:
    st.stop()


def run(q: str) -> dict | None:
    # THE BOX IS DRAWN FIRST, BEFORE ANY NETWORK CALL.
    # Streamlit pushes each element as the script produces it, so an
    # element created after a slow call cannot appear until that call
    # returns. Anything drawn here is on screen immediately, and the wait
    # that follows becomes visible instead of blank.
    box = st.empty()
    box.info("sending the question …")
    t_click = time.perf_counter()
    r = requests.post(f"{API}/ask", json={"question": q}, timeout=30)
    r.raise_for_status()
    post_s = time.perf_counter() - t_click
    job_id = r.json()["id"]
    waited = 0
    while waited < GIVE_UP_AFTER:
        job = requests.get(f"{API}/ask/{job_id}", timeout=30).json()
        if job["status"] == "done":
            box.empty()
            return job["result"]
        if job["status"] == "error":
            box.empty()
            st.error(job["error"])
            return None
        box.info(f"{job['stage']} … {waited}s   (accepted in {post_s:.1f}s)")
        time.sleep(POLL_SECONDS)
        waited += POLL_SECONDS
    st.warning(f"No answer after {GIVE_UP_AFTER}s. The job id is {job_id}; "
               f"it may still finish.")
    return None


if submitting:
    st.session_state["result"] = run(question.strip())

res = st.session_state.get("result")
if res:
    if True:
        if res["stopped_at_gate"]:
            st.warning(res["text"])
            st.caption(f"gate: {res['gate_decision']} — {res['gate_reason']}")
            # THE GATE ASKED SOMETHING AND NEEDS SOMEWHERE TO HEAR THE ANSWER.
            # The reply is GLUED to the original question, never sent on its
            # own - typing "UK" alone would start a fresh classification of the
            # word "UK". Same rule as ask.py's PENDING; the state lives in the
            # caller, so respond() stays a pure function of one question.
            if res["gate_decision"] == "INCOMPLETE":
                with st.form("followup", clear_on_submit=True):
                    reply = st.text_input(
                        "Your answer — just the missing piece",
                        placeholder="e.g. UK, or SS1/23")
                    sent = st.form_submit_button("Send", type="primary")
                if sent and reply.strip():
                    combined = f"{res['question']} ({reply.strip()})"
                    st.session_state["result"] = run(combined)
                    st.rerun()
        elif res["refused"]:
            st.warning(f"Refused: {res['refusal_reason']}")
        else:
            st.markdown(res["text"])

        # THE EVIDENCE PANEL IS THE FEATURE, not a nicety. For a validator the
        # answer is a claim; the passage underneath it is why the claim counts.
        if res["citations"]:
            st.subheader("Citations")
            st.dataframe(res["citations"], use_container_width=True)
        if res["evidence"]:
            st.subheader("Evidence — what the model was actually given")
            for i, e in enumerate(res["evidence"], 1):
                label = (f"S{i} · {e['short_name']} · {e['heading'][:70]} "
                         f"({e['chars']:,} chars"
                         + (", windowed)" if e["windowed"] else ")"))
                with st.expander(label):
                    st.write("matched on: " + ", ".join(e["cites"]))

        st.divider()
        st.caption(f"{res['seconds']}s · {res['model']} · {res['config_hash']}"
                   + (f" · trace {res['trace_id'][:12]}" if res["trace_id"] else ""))

        st.button("Ask another question", type="primary",
                  use_container_width=True, on_click=_clear)

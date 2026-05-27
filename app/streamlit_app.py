"""Streamlit demo for the TriSafe-RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.build import build_pipeline, load_config


@st.cache_resource
def get_pipeline():
    cfg = load_config(os.getenv("TRISAFE_CONFIG", "configs/experiment_v1.yaml"))
    data_dir = os.getenv("TRISAFE_DATA", "data/raw")
    src = data_dir if Path(data_dir).exists() else None
    return build_pipeline(cfg, source_dir=src)


def main():
    st.set_page_config(page_title="TriSafe-RAG", page_icon="🛡️", layout="wide")
    st.title("TriSafe-RAG — Trustworthy Multilingual RAG")
    st.caption("Evidence-grounded · cross-lingual fairness · selective abstention")

    pipe = get_pipeline()
    lang = st.selectbox("Force language", ["auto", "en", "si", "ta", "mixed"], index=0)
    q = st.text_area("Ask a question", height=100, placeholder="e.g. How do I renew my passport?")

    if st.button("Run", type="primary") and q.strip():
        with st.spinner("Retrieving and grounding..."):
            resp = pipe.query(q.strip(), language=None if lang == "auto" else lang)

        c1, c2, c3 = st.columns(3)
        c1.metric("Decision", resp.decision.decision)
        c2.metric("Confidence", f"{resp.confidence:.2f}")
        c3.metric("Grounding", f"{resp.grounding.overall_support:.2f}")

        st.subheader("Answer")
        st.write(resp.answer.text)
        if resp.decision.decision != "answer":
            st.warning(f"Abstention reason: {resp.decision.reason}")

        st.subheader("Evidence")
        for h in resp.hits:
            with st.expander(f"[{h.rank + 1}] {h.chunk.chunk_id} — score {h.score:.3f}"):
                st.write(h.chunk.text)

        st.subheader("Audit record")
        st.json(resp.audit.__dict__)


if __name__ == "__main__":
    main()

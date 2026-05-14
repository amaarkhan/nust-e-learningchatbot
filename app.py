import os
import re
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional fallback path
    SentenceTransformer = None


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "KB.txt"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


load_dotenv()


def load_kb_text():
    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as handle:
        return handle.read()


def normalize_text(text):
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def split_course_sections(text):
    pattern = re.compile(r"(?m)^COURSE\s+(\d+):\s*(.+)$")
    matches = list(pattern.finditer(text))
    documents = []

    if not matches:
        return documents

    overview_text = normalize_text(text[: matches[0].start()])
    if overview_text:
        documents.append(
            {
                "title": "Platform overview",
                "content": overview_text,
                "source_url": "https://connected.nust.edu.pk/course/",
            }
        )

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        course_text = normalize_text(text[start:end])
        documents.append(
            {
                "title": f"Course {match.group(1)}: {match.group(2).strip()}",
                "content": course_text,
                "source_url": f"https://connected.nust.edu.pk/course/view.php?id={re.search(r'id=(\d+)', course_text).group(1)}" if re.search(r'id=(\d+)', course_text) else "https://connected.nust.edu.pk/course/",
            }
        )

    return documents


def extract_special_sections(text):
    documents = []

    quick_ref_match = re.search(r"(?ms)^QUICK REFERENCE: ALL COURSES AT A GLANCE\n(.*?)\n\nFREQUENTLY ASKED QUESTIONS", text)
    if quick_ref_match:
        documents.append(
            {
                "title": "Quick reference: all courses at a glance",
                "content": normalize_text(quick_ref_match.group(1)),
                "source_url": "https://connected.nust.edu.pk/course/",
            }
        )
        documents.append(
            {
                "title": "Course count summary",
                "content": "The quick reference lists 16 courses at a glance. A separate Course 17 entry is marked as details pending.",
                "source_url": "https://connected.nust.edu.pk/course/",
            }
        )

    faq_match = re.search(r"(?ms)^FREQUENTLY ASKED QUESTIONS \(FAQ\)\n(.*)$", text)
    if faq_match:
        documents.append(
            {
                "title": "Frequently asked questions",
                "content": normalize_text(faq_match.group(1)),
                "source_url": "https://connected.nust.edu.pk/course/",
            }
        )

    return documents


def build_documents_from_kb(text):
    documents = []
    overview_section = re.search(r"(?ms)^NUST ConnectEd — Complete Course Knowledge Base\n(.*?)\n\nCOURSE 1:", text)
    if overview_section:
        documents.append(
            {
                "title": "Knowledge base header and platform overview",
                "content": normalize_text(overview_section.group(1)),
                "source_url": "https://connected.nust.edu.pk/course/",
            }
        )

    documents.extend(split_course_sections(text))
    documents.extend(extract_special_sections(text))
    return documents


def build_corpus(documents):
    docs = []
    for item in documents:
        title = item.get("title") or item.get("question") or "Untitled document"
        parts = [f"Title: {title}"]
        if item.get("question"):
            parts.append(f"Question: {item['question']}")
        if item.get("answer"):
            parts.append(f"Answer: {item['answer']}")
        if item.get("content"):
            parts.append(f"Content: {item['content']}")
        if item.get("note"):
            parts.append(f"Note: {item['note']}")
        if item.get("description"):
            parts.append(f"Description: {item['description']}")
        if item.get("video_url"):
            parts.append(f"Video: {item['video_url']}")
        docs.append("\n".join(parts))
    return docs


def build_query_rich_text(query):
    query_lower = query.lower()
    extra_terms = []

    if any(term in query_lower for term in ["how many", "total", "number of courses", "courses are being offered"]):
        extra_terms.append("quick reference all courses at a glance total count list")

    if any(term in query_lower for term in ["price", "fee", "cost"]):
        extra_terms.append("price PKR bank transfer debit credit card")

    if any(term in query_lower for term in ["enroll", "enrollment", "apply"]):
        extra_terms.append("enrollment bank transfer course page enroll now")

    if any(term in query_lower for term in ["certificate", "certificates"]):
        extra_terms.append("custom certificate completion certificate")

    if any(term in query_lower for term in ["deadline", "date", "when should i apply"]):
        extra_terms.append("last updated deadline 25 July 2026")

    if extra_terms:
        return f"{query}\n\n{' '.join(extra_terms)}"
    return query


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


@st.cache_resource
def get_embedding_model():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def build_index():
    kb_text = load_kb_text()
    items = build_documents_from_kb(kb_text)
    corpus = build_corpus(items)
    embedding_model = get_embedding_model()

    if embedding_model is not None:
        embeddings = embedding_model.encode(
            corpus,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return items, corpus, embedding_model, embeddings, "embeddings"

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return items, corpus, vectorizer, matrix, "tfidf"


def retrieve_context(query, items, corpus, retriever, matrix, mode, top_k=3):
    query_text = build_query_rich_text(query)

    if mode == "embeddings":
        query_vector = retriever.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        scores = np.dot(matrix, query_vector)
    else:
        query_vector = retriever.transform([query_text])
        scores = cosine_similarity(query_vector, matrix).flatten()

    ranked = scores.argsort()[::-1][:top_k]

    retrieved = []
    for index in ranked:
        item = items[index].copy()
        item["score"] = float(scores[index])
        item["context"] = corpus[index]
        retrieved.append(item)
    return retrieved


def format_context(retrieved):
    blocks = []
    for item in retrieved:
        block = [f"Title: {item.get('title') or item.get('question') or 'Untitled document'}"]
        if item.get("question"):
            block.append(f"Question: {item['question']}")
        if item.get("answer"):
            block.append(f"Answer: {item['answer']}")
        if item.get("content"):
            block.append(f"Content: {item['content']}")
        if item.get("note"):
            block.append(f"Note: {item['note']}")
        if item.get("description"):
            block.append(f"Description: {item['description']}")
        if item.get("video_url"):
            block.append(f"Video: {item['video_url']}")
        if item.get("source_url"):
            block.append(f"Source: {item['source_url']}")
        blocks.append("\n".join(block))
    return "\n\n---\n\n".join(blocks)


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def generate_answer(question, retrieved):
    client = get_groq_client()
    if client is None:
        return (
            "GROQ_API_KEY is missing. Add it to your .env file and restart the app."
        )

    context = format_context(retrieved)
    system_prompt = (
        "You are the ConnectEd knowledge base assistant for NUST. "
        "Answer using only the provided context. "
        "If the context includes an explicit list, table, total, date, price, or course count, state it directly and do not say it is unclear. "
        "If the context does not contain the answer, say you could not find it in the knowledge base and suggest contacting the support email info@connected.nust.edu.pk. "
        "Keep answers short, helpful, and direct. "
        "When a course question appears, use the course details in the context. "
        "Do not mention internal retrieval limits or that the answer is 'not explicitly stated' if the context already provides the fact."
    )
    user_prompt = f"""
Question: {question}

Knowledge base context:
{context}

Write a direct answer from the context. If the context contains a number, count, price, deadline, or course total, give that exact fact.
""".strip()

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask anything about ConnectEd courses, pricing, requirements, enrollment, or FAQ details.",
            }
        ]


def render_sources(retrieved):
    with st.expander("Retrieved sources", expanded=False):
        for item in retrieved:
            st.markdown(f"**{item.get('title') or item.get('question') or 'Untitled document'}**")
            if item.get("description"):
                st.write(item["description"])
            if item.get("content"):
                st.write(item["content"])
            if item.get("answer"):
                st.write(item["answer"])
            if item.get("note"):
                st.caption(item["note"])
            if item.get("video_url"):
                st.link_button("Open video", item["video_url"])
            if item.get("source_url"):
                st.link_button("Open source", item["source_url"])
            st.divider()


def main():
    st.set_page_config(page_title="ConnectEd Bot", page_icon="💬", layout="wide")
    st.title("ConnectEd Bot")
    st.caption("RAG-powered chatbot for the NUST ConnectEd knowledge base in KB.txt.")

    items, corpus, retriever, matrix, mode = build_index()
    init_state()

    with st.sidebar:
        st.subheader("Status")
        st.write(f"Knowledge base documents: {len(items)}")
        st.write(f"Retrieval mode: {mode}")
        st.write(f"Groq model: {DEFAULT_MODEL}")
        st.write("Data source: data/KB.txt")
        st.info("If you update KB.txt, restart the app so the index rebuilds. Set EMBEDDING_MODEL to change the local embedding model.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a ConnectEd FAQ question")
    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    retrieved = retrieve_context(user_question, items, corpus, retriever, matrix, mode, top_k=3)
    answer = generate_answer(user_question, retrieved)

    with st.chat_message("assistant"):
        st.markdown(answer)
        render_sources(retrieved)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
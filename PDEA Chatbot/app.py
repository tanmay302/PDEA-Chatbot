# app.py  —  Streamlit chat UI for PDEA College Chatbot

import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.rag_pipeline import RAGPipeline
from config.settings import TOP_K, SIMILARITY_THRESHOLD

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDEA College Chatbot",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f1b2d 0%, #1a2d4a 100%);
}
[data-testid="stVerticalBlock"] { gap: 0.4rem; }

.user-bubble {
    background: #1e4fa3;
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px 15%;
    font-size: 15px;
    line-height: 1.6;
    word-wrap: break-word;
}
.bot-bubble {
    background: #1e2d45;
    color: #e8edf5;
    padding: 14px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 15% 6px 0;
    font-size: 15px;
    line-height: 1.6;
    border: 1px solid #2a3f5f;
    word-wrap: break-word;
}
.bot-bubble.fallback {
    border-color: #7a3f2a;
    background: #2a1f1a;
}
.source-badge {
    font-size: 11px;
    color: #6b8cba;
    margin-bottom: 6px;
    letter-spacing: 0.3px;
}
.bubble-content strong { color: #a8c8f0; }
.bubble-content em     { color: #c0d8f0; font-style: italic; }

.score-pill {
    display: inline-block;
    background: #0d2137;
    color: #4a9eda;
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 11px;
    margin: 3px 3px 0 0;
    border: 1px solid #1e3d5c;
}
.related-label {
    color: #5a7a9a;
    font-size: 12px;
    margin-top: 10px;
    margin-bottom: 2px;
}
.header-bar {
    background: #0d1f35;
    border: 1px solid #1e3d5c;
    padding: 14px 20px;
    border-radius: 12px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.header-title {
    color: #e8edf5;
    font-size: 20px;
    font-weight: 600;
    margin: 0 0 2px 0;
}
.header-sub {
    color: #6b8cba;
    font-size: 12px;
    margin: 0;
}
.status-dot {
    width: 8px; height: 8px;
    background: #2ecc71;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}
.stats-row {
    display: flex;
    gap: 10px;
    margin-bottom: 14px;
}
.stat-box {
    flex: 1;
    background: #0d1f35;
    border: 1px solid #1e3d5c;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
}
.stat-num   { color: #4a9eda; font-size: 22px; font-weight: 700; }
.stat-label { color: #5a7a9a; font-size: 11px; margin-top: 2px; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px; }
.empty-hint {
    color: #5a7a9a;
    font-size: 14px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def md_to_html(text: str) -> str:
    """Convert basic markdown to HTML for safe injection into bubbles."""
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',     r'<em>\1</em>',         text)
    text = re.sub(r'`(.*?)`',       r'<code>\1</code>',     text)
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return text

# ── Intent detection ──────────────────────────────────────────────────────────
GREETINGS = {"hi","hello","hey","helloo","helo","hii","hiii",
             "namaste","good morning","good afternoon","good evening","sup","yo"}
THANKS    = {"thanks","thank you","thankyou","thank u","ty","thx",
             "great","awesome","perfect","helpful","got it","ok thanks"}
FAREWELLS = {"bye","goodbye","see you","take care","cya","tata"}
ABOUT     = {"who are you","what are you","what can you do",
             "help","how can you help","what do you know","about"}

def detect_intent(query: str) -> str | None:
    q = query.lower().strip().rstrip("!?.")
    if q in GREETINGS or any(q.startswith(g) for g in GREETINGS):
        return "greeting"
    if q in THANKS:    return "thanks"
    if q in FAREWELLS: return "farewell"
    if q in ABOUT:     return "about"
    return None

INTENT_RESPONSES = {
    "greeting": (
        "👋 Hello! Welcome to the **PDEA College Chatbot**.\n\n"
        "I can help you with:\n"
        "• Admissions & eligibility\n"
        "• Courses & programs offered\n"
        "• Fees & scholarships\n"
        "• Hostel, placements & facilities\n\n"
        "What would you like to know?"
    ),
    "thanks":   "😊 You're welcome! Feel free to ask anything else about PDEA colleges.",
    "farewell": "👋 Goodbye! Best of luck with your admissions and studies.",
    "about": (
        "🤖 I'm the **PDEA College Chatbot**, built using a RAG pipeline.\n\n"
        "**How I work:**\n"
        "• Your question is converted to a vector embedding\n"
        "• FAISS searches the knowledge base for similar Q&A pairs\n"
        "• The best matching answer is returned instantly\n\n"
        "Ask me anything about PDEA Pune colleges!"
    ),
}

# ── Load pipeline (cached — model loads only once per session) ────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return RAGPipeline()

# ── Session state init ────────────────────────────────────────────────────────
for key, default in {
    "messages":       [],
    "total_queries":  0,
    "matched_queries": 0,
    "debug":          False,
    "prefill":        "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div style="font-size:36px;line-height:1">🎓</div>
  <div>
    <p class="header-title">PDEA College Chatbot</p>
    <p class="header-sub">
      <span class="status-dot"></span>Online &nbsp;·&nbsp;
      RAG + FAISS + sentence-transformers
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stats bar ─────────────────────────────────────────────────────────────────
total   = st.session_state.total_queries
matched = st.session_state.matched_queries
accuracy = f"{(matched / total * 100):.0f}%" if total > 0 else "—"

st.markdown(f"""
<div class="stats-row">
  <div class="stat-box">
    <div class="stat-num">{total}</div>
    <div class="stat-label">Queries asked</div>
  </div>
  <div class="stat-box">
    <div class="stat-num">{matched}</div>
    <div class="stat-label">Matched</div>
  </div>
  <div class="stat-box">
    <div class="stat-num">{accuracy}</div>
    <div class="stat-label">Match rate</div>
  </div>
  <div class="stat-box">
    <div class="stat-num">{TOP_K}</div>
    <div class="stat-label">Top-K</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state.debug = st.toggle(
        "Debug mode (show retrieval scores)",
        value=st.session_state.debug
    )

    st.markdown("---")
    st.markdown("### 📊 Pipeline info")
    st.markdown(f"""
- **Model:** all-MiniLM-L6-v2
- **Dimensions:** 384
- **Top-K:** {TOP_K}
- **Threshold:** {SIMILARITY_THRESHOLD}
- **Vector DB:** FAISS IndexFlatIP
- **Similarity:** Cosine
    """)

    st.markdown("---")
    st.markdown("### 💡 Try these questions")
    sample_questions = [
        "What courses are offered?",
        "What are the fees?",
        "How to apply for admission?",
        "Is there a hostel?",
        "Tell me about placements",
        "Are scholarships available?",
        "Is PDEA AICTE approved?",
        "What is the contact number?",
        "Where is PDEA located?",
        "What is MHT-CET cutoff?",
    ]
    for q in sample_questions:
        if st.button(q, key=f"sb_{q}", use_container_width=True):
            st.session_state.prefill = q

    st.markdown("---")
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages       = []
        st.session_state.total_queries  = 0
        st.session_state.matched_queries = 0
        st.rerun()

# ── Suggestion chips (only on empty chat) ─────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<p class="empty-hint">👋 Hello! What would you like to know about PDEA colleges?</p>',
                unsafe_allow_html=True)
    chips = [
        "Courses offered", "Fee structure", "Admission process",
        "Hostel facility", "Placement cell", "Scholarships",
    ]
    cols = st.columns(3)
    for i, chip in enumerate(chips):
        if cols[i % 3].button(chip, key=f"chip_{chip}"):
            st.session_state.prefill = chip

# ── Render one message ────────────────────────────────────────────────────────
def render_message(role: str, content: str, meta: dict = None):
    if role == "user":
        st.markdown(
            f'<div class="user-bubble">{md_to_html(content)}</div>',
            unsafe_allow_html=True,
        )
        return

    # Bot bubble
    source  = meta.get("source",  "local") if meta else "local"
    results = meta.get("results", [])       if meta else []

    badge_map = {
        "local":    "📚 Knowledge base",
        "gemini":   "🤖 Gemini AI",
        "fallback": "❓ No match found",
        "intent":   "👋 Greeting",
    }
    badge       = badge_map.get(source, "📚 Knowledge base")
    bubble_cls  = "bot-bubble fallback" if source == "fallback" else "bot-bubble"

    # Related questions pills (skip for fallback / intent)
    related_html = ""
    if results and len(results) > 1 and source not in ("fallback", "intent"):
        pills = "".join(
            f'<span class="score-pill">'
            f'{r["score"]:.2f} · {r["question"][:45]}…'
            f'</span>'
            for r in results[1:]
        )
        related_html = (
            f'<div class="related-label">Related questions:</div>{pills}'
        )

    st.markdown(f"""
    <div class="{bubble_cls}">
      <div class="source-badge">{badge}</div>
      <div class="bubble-content">{md_to_html(content)}</div>
      {related_html}
    </div>
    """, unsafe_allow_html=True)

    # Debug expander
    if st.session_state.debug and results:
        with st.expander("🔍 Debug — retrieval scores"):
            for r in results:
                st.markdown(
                    f"`{r['score']:.4f}` &nbsp; **{r['question']}**\n\n"
                    f"> {r['answer'][:150]}…"
                )

# ── Render full chat history ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("meta"))

# ── Input handling ────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
query   = st.chat_input("Ask about admissions, courses, fees, hostel…")

# Chip / sidebar button takes priority if no direct input typed
if prefill and not query:
    query = prefill

# ── Process & respond ─────────────────────────────────────────────────────────
if query:
    query = query.strip()

    # Save + render user message
    st.session_state.messages.append({"role": "user", "content": query})
    render_message("user", query)

    intent = detect_intent(query)

    if intent:
        reply = INTENT_RESPONSES.get(intent, "How can I help you?")
        st.session_state.messages.append({
            "role": "bot",
            "content": reply,
            "meta": {"source": "intent", "results": []},
        })
        render_message("bot", reply, {"source": "intent", "results": []})

    else:
        with st.spinner("🔍 Searching knowledge base…"):
            pipeline = load_pipeline()
            result   = pipeline.answer(query)

        st.session_state.total_queries += 1
        if result["source"] != "fallback":
            st.session_state.matched_queries += 1

        st.session_state.messages.append({
            "role":    "bot",
            "content": result["answer"],
            "meta":    result,
        })
        render_message("bot", result["answer"], result)

    st.rerun()
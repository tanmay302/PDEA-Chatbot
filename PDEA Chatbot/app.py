# app.py — PDEA College Assistant (Upgraded UI)

import sys, os, re
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st
from src.rag_pipeline import RAGPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDEA College Assistant",
    page_icon="🎓",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & base ── */
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stAppViewContainer"] {
    background: #0b1929;
}
[data-testid="stVerticalBlock"] { gap: 0.3rem; }
[data-testid="stSidebar"] {
    background: #0d1f35 !important;
    border-right: 1px solid #1a3050;
}

/* ── Header ── */
.pdea-header {
    background: linear-gradient(90deg, #0d1f35 0%, #132d4a 100%);
    border: 1px solid #1e3d5c;
    border-radius: 16px;
    padding: 16px 22px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 6px;
}
.pdea-logo {
    width: 54px;
    height: 54px;
    background: linear-gradient(135deg, #1a56db, #0ea5e9);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    flex-shrink: 0;
}
.pdea-title {
    color: #f0f4ff;
    font-size: 20px;
    font-weight: 700;
    margin: 0 0 3px 0;
    letter-spacing: 0.2px;
}
.pdea-subtitle {
    color: #7a9cc0;
    font-size: 12px;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 6px;
}
.online-dot {
    width: 7px;
    height: 7px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 6px #22c55e88;
}
.pdea-tagline {
    color: #94b4d4;
    font-size: 11px;
    margin: 0;
    margin-left: auto;
    text-align: right;
    flex-shrink: 0;
    line-height: 1.5;
}

/* ── Chip section label ── */
.chip-section-label {
    color: #4a7a9a;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin: 14px 0 8px 2px;
}

/* ── Chip / quick-topic buttons ── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #0f2540;
    color: #7eb8e8;
    border: 1px solid #1e3d5c;
    border-radius: 20px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.18s ease;
    width: 100%;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #1a3d6e;
    border-color: #3a7cbd;
    color: #c8e4ff;
    transform: translateY(-1px);
}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: #0f2540;
    color: #7eb8e8;
    border: 1px solid #1e3d5c;
    border-radius: 10px;
    font-size: 13px;
    text-align: left;
    padding: 8px 12px;
    width: 100%;
    transition: background 0.15s;
    margin-bottom: 2px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1a3d6e;
    border-color: #3a7cbd;
    color: #c8e4ff;
}
[data-testid="stSidebar"] h3 {
    color: #94b4d4 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    margin-bottom: 8px !important;
}

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #1a4fad, #1e3f8a);
    color: #ffffff;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 18%;
    font-size: 15px;
    line-height: 1.65;
    word-wrap: break-word;
    box-shadow: 0 2px 10px #00000044;
}
.bot-bubble {
    background: #0e2035;
    color: #dde8f5;
    padding: 16px 18px;
    border-radius: 4px 18px 18px 18px;
    margin: 8px 10% 8px 0;
    font-size: 14px;
    line-height: 1.75;
    border: 1px solid #1e3a58;
    word-wrap: break-word;
    box-shadow: 0 2px 10px #00000033;
}
.bot-bubble.fallback {
    border-color: #5c3020;
    background: #1c1008;
}

/* ── Formatted response inside bubble ── */
.bot-response h3 {
    color: #5ab0ee;
    font-size: 13px;
    font-weight: 700;
    margin: 14px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e3a58;
    letter-spacing: 0.2px;
    text-transform: uppercase;
}
.bot-response h3:first-child { margin-top: 2px; }

.bot-response ul {
    margin: 4px 0 10px 0;
    padding-left: 16px;
}
.bot-response ul li {
    color: #c0d8f0;
    margin-bottom: 5px;
    font-size: 14px;
    line-height: 1.6;
}
.bot-response p {
    color: #c0d8f0;
    margin: 0 0 8px 0;
    font-size: 14px;
    line-height: 1.75;
}
.bot-response table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 12px 0;
    font-size: 13px;
    border-radius: 8px;
    overflow: hidden;
}
.bot-response table th {
    background: #1a3a5c;
    color: #80bcf0;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #254e78;
}
.bot-response table td {
    background: #0a1e32;
    color: #b8d4ee;
    padding: 7px 12px;
    border-bottom: 1px solid #182e46;
    vertical-align: top;
}
.bot-response table tr:last-child td { border-bottom: none; }
.bot-response table tr:hover td { background: #102840; }

.bot-response strong { color: #80c0f8; font-weight: 600; }
.bot-response em     { color: #a0ccf0; font-style: italic; }
.bot-response code   {
    background: #081828;
    color: #50a8e0;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 12px;
}

/* ── Related pills ── */
.related-wrap {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid #162a40;
}
.related-label {
    color: #3a6a8a;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.related-pill {
    display: inline-block;
    background: #081828;
    color: #4090cc;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 11px;
    margin: 2px 3px 2px 0;
    border: 1px solid #1a3050;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 30px 16px 18px;
    color: #4a7a9a;
}
.empty-state .big-icon { font-size: 44px; margin-bottom: 12px; }
.empty-state h2 { color: #6a9ec8; font-size: 18px; font-weight: 600; margin: 0 0 6px; }
.empty-state p  { color: #3a6a8a; font-size: 14px; margin: 0; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .user-bubble { margin-left: 4%; font-size: 14px; padding: 10px 14px; }
    .bot-bubble  { margin-right: 2%; font-size: 13px; padding: 12px 14px; }
    .pdea-title  { font-size: 16px; }
    .pdea-tagline { display: none; }
    .bot-response table { font-size: 12px; }
    .bot-response table th,
    .bot-response table td { padding: 5px 8px; }
}
@media (max-width: 480px) {
    .pdea-header { padding: 12px 14px; gap: 10px; }
    .pdea-logo   { width: 42px; height: 42px; font-size: 20px; border-radius: 10px; }
    .user-bubble, .bot-bubble { margin-left: 1%; margin-right: 1%; }
    .bot-response h3 { font-size: 12px; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_str(val) -> str:
    """Convert any value to a plain string safely."""
    if val is None:             return ""
    if isinstance(val, str):   return val
    if isinstance(val, bool):  return "Yes" if val else "No"
    if isinstance(val, (int, float)): return str(val)
    if isinstance(val, list):  return ", ".join(safe_str(x) for x in val)
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            label = k.replace("_", " ").capitalize()
            c = safe_str(v)
            if c: parts.append(f"{label}: {c}")
        return ". ".join(parts)
    return str(val)


def _inline_md(text: str) -> str:
    """Convert **bold**, *em*, `code` markdown to HTML."""
    text = safe_str(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',     r'<em>\1</em>',         text)
    text = re.sub(r'`(.*?)`',       r'<code>\1</code>',     text)
    return text


def format_response(raw: str) -> str:
    """
    Transform a flat dot-separated answer into clean HTML:
    headings + bullet lists + tables where appropriate.
    """
    raw = safe_str(raw).strip()
    if not raw:
        return '<div class="bot-response"><p>No information available.</p></div>'

    lower = raw.lower()

    # ── Detect if table-worthy (paired key-value data) ────────────────────────
    table_triggers = [
        ("course", "affiliation"), ("course", "fee"), ("college", "location"),
        ("program", "duration"),   ("branch", "fee"), ("type", "details"),
        ("college", "address"),    ("subject", "marks"),
    ]
    is_table_candidate = any(a in lower and b in lower for a, b in table_triggers)

    # ── Split into segments at ". " followed by a capital letter ──────────────
    segments = [s.strip() for s in re.split(r'\.\s+(?=[A-Z])', raw) if s.strip()]

    # Very short answer — render as single paragraph
    if len(segments) <= 2:
        return (
            f'<div class="bot-response">'
            f'<p>{_inline_md(raw)}</p>'
            f'</div>'
        )

    # ── Parse into (heading, [items]) sections ────────────────────────────────
    sections: list[tuple[str, list[str]]] = []
    current_head  = ""
    current_items: list[str] = []

    for seg in segments:
        colon_pos = seg.find(": ")
        if colon_pos != -1 and colon_pos < 50:
            # Flush previous section
            if current_head or current_items:
                sections.append((current_head, current_items))
            current_head  = seg[:colon_pos].strip()
            rest          = seg[colon_pos + 2:].strip()
            current_items = [rest] if rest else []
        else:
            current_items.append(seg)

    if current_head or current_items:
        sections.append((current_head, current_items))

    # ── Try table: all sections have exactly one value item ───────────────────
    if is_table_candidate and len(sections) >= 3:
        all_single = all(len(items) == 1 for _, items in sections if _[0])
        headed     = [(h, items) for h, items in sections if h and items]
        if all_single and len(headed) >= 3:
            rows = "".join(
                f"<tr>"
                f"<td><strong>{_inline_md(h)}</strong></td>"
                f"<td>{_inline_md(items[0])}</td>"
                f"</tr>"
                for h, items in headed
            )
            return (
                f'<div class="bot-response">'
                f"<table><thead><tr><th>Category</th><th>Details</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
                f"</div>"
            )

    # ── Default: headings + bullets ───────────────────────────────────────────
    html_parts: list[str] = []
    for head, items in sections:
        clean_items = [i for i in items if i.strip()]
        if head:
            html_parts.append(f"<h3>{_inline_md(head)}</h3>")
        if not clean_items:
            continue
        if len(clean_items) == 1:
            html_parts.append(f"<p>{_inline_md(clean_items[0])}</p>")
        else:
            lis = "".join(f"<li>{_inline_md(i)}</li>" for i in clean_items)
            html_parts.append(f"<ul>{lis}</ul>")

    return f'<div class="bot-response">{"".join(html_parts)}</div>'


def format_plain(text: str) -> str:
    """For greeting / intent messages — simple newline-to-br conversion."""
    text = safe_str(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',     r'<em>\1</em>',         text)
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
GREETINGS = {"hi","hello","hey","helloo","helo","hii","hiii",
             "namaste","good morning","good afternoon","good evening","sup","yo"}
THANKS    = {"thanks","thank you","thankyou","thank u","ty","thx",
             "great","awesome","perfect","helpful","got it","ok thanks"}
FAREWELLS = {"bye","goodbye","see you","take care","cya","tata"}
ABOUT     = {"who are you","what are you","what can you do",
             "help","how can you help","what do you know","about"}

def detect_intent(query: str) -> str | None:
    q = query.lower().strip().rstrip("!?.")
    if q in GREETINGS or any(q.startswith(g) for g in GREETINGS): return "greeting"
    if q in THANKS:    return "thanks"
    if q in FAREWELLS: return "farewell"
    if q in ABOUT:     return "about"
    return None

INTENT_RESPONSES = {
    "greeting": (
        "👋 **Hello! Welcome to PDEA College Assistant.**\n\n"
        "I can help you with:\n"
        "• Admissions and eligibility criteria\n"
        "• Courses and programs offered\n"
        "• Fee structure and scholarships\n"
        "• Hostel, placements and campus facilities\n\n"
        "What would you like to know today?"
    ),
    "thanks":   "😊 You're welcome! Feel free to ask anything else about PDEA colleges.",
    "farewell": "👋 Goodbye! Best of luck with your admissions. We are here if you need us.",
    "about": (
        "🎓 **I am the PDEA College Assistant.**\n\n"
        "I can answer questions about:\n"
        "• All PDEA colleges in Pune\n"
        "• Courses, fees, and admissions\n"
        "• Hostel, library, and campus facilities\n"
        "• Placements and extracurricular activities\n\n"
        "Just type your question below!"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return RAGPipeline()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for key, default in {
    "messages":        [],
    "total_queries":   0,
    "matched_queries": 0,
    "debug":           False,
    "prefill":         "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state.debug = st.toggle(
        "Show debug info",
        value=st.session_state.debug,
    )
    st.markdown("---")

    st.markdown("### 💬 Quick Questions")
    quick_qs = [
        "What courses are offered?",
        "What are the fees?",
        "How to apply for admission?",
        "Is there a hostel?",
        "Tell me about placements",
        "Are scholarships available?",
        "Does PDEA have a law college?",
        "What is the contact number?",
        "Where is PDEA located?",
        "Does PDEA offer PhD programs?",
    ]
    for q in quick_qs:
        if st.button(q, key=f"sb_{q}", use_container_width=True):
            st.session_state.prefill = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.total_queries   = 0
        st.session_state.matched_queries = 0
        st.rerun()

    # Footer
    st.markdown(
        "<div style='margin-top:24px;text-align:center;"
        "color:#2a4a6a;font-size:11px;line-height:1.6'>"
        "PDEA College Assistant<br>Pune District Education Association"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="pdea-header">
  <div class="pdea-logo">🎓</div>
  <div style="flex:1;min-width:0;">
    <p class="pdea-title">PDEA College Assistant</p>
    <p class="pdea-subtitle">
      <span class="online-dot"></span>
      Online &nbsp;·&nbsp; Pune District Education Association
    </p>
  </div>
  <p class="pdea-tagline">Shaping Futures<br>Since 1957</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE + CHIPS
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
      <div class="big-icon">💬</div>
      <h2>How can I help you today?</h2>
      <p>Ask me anything about PDEA colleges — admissions, courses, fees, and more.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="chip-section-label">Popular topics</p>',
        unsafe_allow_html=True,
    )

    chip_data = [
        ("🎓 Courses",       "What courses are offered?"),
        ("💰 Fees",          "What are the fees?"),
        ("📋 Admissions",    "How to apply for admission?"),
        ("🏠 Hostel",        "Is there a hostel?"),
        ("💼 Placements",    "Tell me about placements"),
        ("🏆 Scholarships",  "Are scholarships available?"),
    ]
    row1 = st.columns(3)
    row2 = st.columns(3)
    for i, (label, query_text) in enumerate(chip_data):
        col = row1[i] if i < 3 else row2[i - 3]
        if col.button(label, key=f"chip_{label}"):
            st.session_state.prefill = query_text


# ══════════════════════════════════════════════════════════════════════════════
# RENDER MESSAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_message(role: str, content, meta: dict = None):
    content = safe_str(content)

    # ── User bubble ───────────────────────────────────────────────────────────
    if role == "user":
        st.markdown(
            f'<div class="user-bubble">{_inline_md(content)}</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Bot bubble ────────────────────────────────────────────────────────────
    source  = meta.get("source",  "local") if meta else "local"
    results = meta.get("results", [])       if meta else []
    bubble_cls = "bot-bubble fallback" if source == "fallback" else "bot-bubble"

    # Choose formatter
    if source == "intent":
        body = f'<div class="bot-response"><p>{format_plain(content)}</p></div>'
    elif source == "fallback":
        body = f'<div class="bot-response"><p>{_inline_md(content)}</p></div>'
    else:
        body = format_response(content)

    # Related questions pills
    related_html = ""
    if results and len(results) > 1 and source not in ("fallback", "intent"):
        pills = "".join(
            f'<span class="related-pill">'
            f'{safe_str(r.get("question",""))[:50]}…'
            f'</span>'
            for r in results[1:]
        )
        related_html = (
            '<div class="related-wrap">'
            '<div class="related-label">You might also ask</div>'
            f'{pills}</div>'
        )

    st.markdown(
        f'<div class="{bubble_cls}">{body}{related_html}</div>',
        unsafe_allow_html=True,
    )

    # Debug expander (only when debug toggle is on)
    if st.session_state.debug and results:
        with st.expander("🔍 Debug info"):
            for r in results:
                st.markdown(
                    f"`{r['score']:.4f}` — **{safe_str(r.get('question',''))}**\n\n"
                    f"> {safe_str(r.get('answer',''))[:200]}…"
                )


# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("meta"))


# ══════════════════════════════════════════════════════════════════════════════
# INPUT + PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
prefill = st.session_state.pop("prefill", "")
query   = st.chat_input("Type your question about PDEA colleges…")
if prefill and not query:
    query = prefill

if query:
    query = safe_str(query).strip()

    # Save + render user message
    st.session_state.messages.append({"role": "user", "content": query})
    render_message("user", query)

    intent = detect_intent(query)

    if intent:
        reply = INTENT_RESPONSES.get(intent, "How can I help you?")
        st.session_state.messages.append({
            "role":    "bot",
            "content": reply,
            "meta":    {"source": "intent", "results": []},
        })
        render_message("bot", reply, {"source": "intent", "results": []})

    else:
        with st.spinner("Looking that up for you…"):
            pipeline = load_pipeline()
            result   = pipeline.answer(query)

        result["answer"] = safe_str(result.get("answer", ""))

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
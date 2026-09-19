"""
Streamlit frontend — v2 multi-turn Support Ticket Decision Assistant.

Architecture: ALL data access via FastAPI HTTP — Streamlit never touches SQLite.

Pages (via st.session_state["page"]):
  auth          — login / register
  new_ticket    — submit a ticket, see AI response, provide follow-up
  history       — list of past tickets
  ticket_detail — full conversation view for one ticket
"""
import html
import os
from io import BytesIO

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "https://ticket-assistant-3.onrender.com/")
TIMEOUT = 60

st.set_page_config(
    page_title="Support Assistant",
    page_icon="🎫",
    layout="centered",
)

# ── CSS: conversation bubbles and badges ──────────────────────────────────────
st.markdown("""
<style>
.bubble-user {
    background-color: #e8f4fd;
    color: #111827 !important;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 10px 0;
    border: 1px solid #bfdbfe;
    border-left: 5px solid #2563eb;
    font-size: 15px;
    line-height: 1.5;
}
.bubble-asst {
    background-color: #f0fdf4;
    color: #111827 !important;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 10px 0;
    border: 1px solid #bbf7d0;
    border-left: 5px solid #16a34a;
    font-size: 15px;
    line-height: 1.5;
}
.bubble-user *, .bubble-asst * {
    color: inherit !important;
}
.bubble-content {
    margin-top: 6px;
    white-space: pre-wrap;
    word-break: break-word;
}

@media (prefers-color-scheme: dark) {
    .bubble-user {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-left: 5px solid #3b82f6 !important;
    }
    .bubble-asst {
        background-color: #0f291e !important;
        color: #f8fafc !important;
        border: 1px solid #1e3a2f !important;
        border-left: 5px solid #22c55e !important;
    }
    .bubble-user *, .bubble-asst * {
        color: inherit !important;
    }
}

.final-badge   {background:#1a7f37;color:white;border-radius:8px;padding:4px 10px;font-weight:bold;}
.progress-badge{background:#d97706;color:white;border-radius:8px;padding:4px 10px;font-weight:bold;}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "token": None,
        "user_email": None,
        "page": "auth",
        "active_ticket_id": None,
        "active_ticket_data": None,   # full TicketDetail dict
        "last_decision": None,        # latest DecisionResponse dict
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _auth() -> dict:
    return {"Authorization": f"Bearer {st.session_state['token']}"}


def _logged_in() -> bool:
    return bool(st.session_state.get("token"))


def _logout():
    for k in ["token", "user_email", "page", "active_ticket_id",
              "active_ticket_data", "last_decision"]:
        st.session_state[k] = None
    st.session_state["page"] = "auth"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post(endpoint, payload=None, auth=False, files=None):
    headers = _auth() if auth else {}
    try:
        if files:
            r = requests.post(f"{API_BASE}{endpoint}", headers=headers,
                              files=files, timeout=TIMEOUT)
        else:
            r = requests.post(f"{API_BASE}{endpoint}", json=payload,
                              headers=headers, timeout=TIMEOUT)
        return r.status_code, _json(r)
    except requests.ConnectionError:
        return 0, {"detail": "Cannot reach the backend. Is FastAPI running?"}
    except requests.Timeout:
        return 0, {"detail": "Request timed out — Gemini may be slow."}


def _get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", headers=_auth(), timeout=TIMEOUT)
        return r.status_code, _json(r)
    except requests.ConnectionError:
        return 0, {"detail": "Cannot reach the backend."}
    except requests.Timeout:
        return 0, {"detail": "Request timed out."}


def _json(r):
    try:
        return r.json()
    except Exception:
        return {"detail": r.text or "Unknown error"}


def _handle_401():
    st.error("Session expired. Please log in again.")
    _logout()
    st.rerun()


# ── Decision display helpers ──────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    if status == "FINAL":
        return '<span class="final-badge">FINAL DECISION</span>'
    return '<span class="progress-badge">IN PROGRESS</span>'


def _show_decision(d: dict, expanded: bool = True):
    """Render a decision card."""
    status = d.get("status", "")
    action = d.get("action", "")
    confidence = d.get("confidence", 0)
    reason = d.get("reason", "")
    question = d.get("question")
    sources = d.get("sources", [])
    required = d.get("required_information", [])

    st.markdown(_status_badge(status), unsafe_allow_html=True)
    st.markdown(f"### {action}")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Reasoning:** {reason}")
    with col2:
        st.metric("Confidence", f"{confidence:.0%}")

    if question:
        st.info(f"**AI Question:** {question}")
    if required:
        st.markdown("**Information still needed:**")
        for item in required:
            st.markdown(f"- `{item}`")
    if sources:
        st.caption("Sources: " + ", ".join(f"`{s}`" for s in sources))


# ── Page: Auth ────────────────────────────────────────────────────────────────

def page_auth():
    st.title("🎫 Support Ticket Assistant")
    st.markdown("*AI-powered, policy-grounded multi-turn support decisions.*")
    st.divider()

    tab_login, tab_reg = st.tabs(["Log in", "Register"])

    with tab_login:
        email = st.text_input("Email", key="li_email")
        pwd = st.text_input("Password", type="password", key="li_pwd")
        if st.button("Log in", type="primary", key="btn_li"):
            if not email or not pwd:
                st.warning("Enter email and password.")
                return
            sc, data = _post("/login", {"email": email, "password": pwd})
            if sc == 200:
                st.session_state["token"] = data["access_token"]
                st.session_state["user_email"] = email
                st.session_state["page"] = "new_ticket"
                st.rerun()
            elif sc == 401:
                st.error("Invalid email or password.")
            elif sc == 0:
                st.error(data["detail"])
            else:
                st.error(data.get("detail", "Login failed."))

    with tab_reg:
        r_email = st.text_input("Email", key="reg_email")
        r_pwd = st.text_input("Password (min 8 chars)", type="password", key="reg_pwd")
        if st.button("Register", key="btn_reg"):
            if not r_email or not r_pwd:
                st.warning("Fill in all fields.")
                return
            sc, data = _post("/register", {"email": r_email, "password": r_pwd})
            if sc == 201:
                st.success("Account created! Please log in.")
            elif sc == 409:
                st.error("Email already registered.")
            elif sc == 422:
                st.error("Invalid email or password too short (min 8 chars).")
            elif sc == 0:
                st.error(data["detail"])
            else:
                st.error(data.get("detail", "Registration failed."))


# ── Page: New Ticket ──────────────────────────────────────────────────────────

def page_new_ticket():
    st.header("📝 New Support Ticket")

    # If there's an active in-progress ticket, show the follow-up UI
    if st.session_state.get("active_ticket_id") and st.session_state.get("last_decision"):
        dec = st.session_state["last_decision"]
        if dec.get("status") == "IN_PROGRESS":
            _show_followup_ui()
            return
        else:
            # Final decision already shown — offer to start new ticket
            _show_decision(dec)
            st.divider()
            if st.button("Start a new ticket", type="primary"):
                st.session_state["active_ticket_id"] = None
                st.session_state["last_decision"] = None
                st.rerun()
            return

    # Fresh ticket form
    msg = st.text_area(
        "Describe your issue",
        height=140,
        placeholder="e.g. My ₹3,500 order arrived damaged yesterday.",
        key="new_msg",
    )

    if st.button("Submit Ticket", type="primary", key="btn_new"):
        if not msg.strip() or len(msg.strip()) < 5:
            st.warning("Please describe your issue in at least 5 characters.")
            return

        with st.spinner("Analysing with AI…"):
            sc, data = _post("/tickets", {"message": msg}, auth=True)

        if sc == 201:
            ticket = data["ticket"]
            decision = data["decision"]
            st.session_state["active_ticket_id"] = ticket["id"]
            st.session_state["last_decision"] = decision
            st.rerun()
        elif sc == 401:
            _handle_401()
        elif sc == 503:
            st.error(f"AI service unavailable: {data.get('detail','')}")
        elif sc == 0:
            st.error(data["detail"])
        else:
            st.error(data.get("detail", f"Error {sc}"))


def _show_followup_ui():
    """Show the in-progress decision and a follow-up form."""
    ticket_id = st.session_state["active_ticket_id"]
    dec = st.session_state["last_decision"]

    # Load conversation
    sc, msgs = _get(f"/tickets/{ticket_id}/messages")
    if sc == 200 and isinstance(msgs, list):
        _render_conversation(msgs)
    st.divider()

    _show_decision(dec)
    st.divider()

    # Follow-up text input
    followup = st.text_area(
        "Your reply",
        height=100,
        placeholder="Provide the requested information here…",
        key="followup_msg",
    )

    # Evidence upload
    uploaded_file = st.file_uploader(
        "Upload evidence (photo/PDF, max 10 MB)",
        type=["jpg", "jpeg", "png", "webp", "gif", "pdf"],
        key="ev_upload",
    )

    col1, col2 = st.columns(2)
    with col1:
        send_reply = st.button("Send Reply", type="primary", key="btn_reply")
    with col2:
        upload_only = st.button("Upload Evidence Only", key="btn_ev_only")

    # Handle evidence upload
    if (send_reply or upload_only) and uploaded_file:
        with st.spinner("Uploading evidence…"):
            file_bytes = uploaded_file.read()
            sc_ev, ev_data = _post(
                f"/tickets/{ticket_id}/evidence",
                auth=True,
                files={
                    "file": (
                        uploaded_file.name,
                        BytesIO(file_bytes),
                        uploaded_file.type,
                    )
                },
            )
        if sc_ev == 201:
            st.success(f"Evidence uploaded: {uploaded_file.name}")
        elif sc_ev == 415:
            st.error("Unsupported file type.")
        elif sc_ev == 413:
            st.error("File too large (max 10 MB).")
        else:
            st.error(ev_data.get("detail", f"Upload failed ({sc_ev})"))

    # Handle follow-up message
    if send_reply and followup.strip():
        with st.spinner("Analysing your reply…"):
            sc_f, f_data = _post(
                f"/tickets/{ticket_id}/messages",
                {"content": followup.strip()},
                auth=True,
            )
        if sc_f == 200:
            st.session_state["last_decision"] = f_data["decision"]
            st.rerun()
        elif sc_f == 400:
            st.warning(f_data.get("detail", "Ticket already closed."))
        elif sc_f == 401:
            _handle_401()
        elif sc_f == 503:
            st.error(f"AI service unavailable: {f_data.get('detail','')}")
        else:
            st.error(f_data.get("detail", f"Error {sc_f}"))


# ── Page: History ─────────────────────────────────────────────────────────────

def page_history():
    st.header("📋 Ticket History")
    sc, data = _get("/tickets")
    if sc == 401:
        _handle_401()
        return
    if sc == 0:
        st.error(data["detail"])
        return
    if sc != 200:
        st.error(f"Failed to load history ({sc})")
        return

    tickets = data
    if not tickets:
        st.info("No tickets yet. Create your first ticket!")
        return

    st.markdown(f"**{len(tickets)} ticket(s)**")
    for t in tickets:
        tid = t["id"]
        act = t.get("latest_action") or "Pending"
        ds = t.get("latest_decision_status") or ""
        badge = "🟢 FINAL" if ds == "FINAL" else ("🟡 IN PROGRESS" if ds == "IN_PROGRESS" else "⚪")
        label = f"#{tid} — {act}  {badge}  ·  {t['created_at'][:10]}"
        with st.expander(label):
            st.markdown(f"**{t['initial_message'][:120]}**")
            if st.button("View full conversation", key=f"view_{tid}"):
                st.session_state["active_ticket_id"] = tid
                st.session_state["page"] = "ticket_detail"
                st.rerun()


# ── Page: Ticket Detail ───────────────────────────────────────────────────────

def page_ticket_detail():
    ticket_id = st.session_state.get("active_ticket_id")
    if not ticket_id:
        st.session_state["page"] = "history"
        st.rerun()
        return

    if st.button("← Back to History"):
        st.session_state["page"] = "history"
        st.rerun()

    sc, data = _get(f"/tickets/{ticket_id}")
    if sc == 401:
        _handle_401()
        return
    if sc == 404:
        st.error("Ticket not found.")
        return
    if sc != 200:
        st.error(f"Error loading ticket ({sc})")
        return

    ticket = data
    status_icon = "🟢" if ticket["status"] == "closed" else "🟡"
    st.header(f"Ticket #{ticket_id}  {status_icon} {ticket['status'].upper()}")
    st.caption(f"Created: {ticket['created_at'][:19]}")
    st.markdown(f"**Original message:** {ticket['initial_message']}")
    st.divider()

    # Conversation
    if ticket.get("messages"):
        st.subheader("Conversation")
        _render_conversation(ticket["messages"])

    # Evidence
    if ticket.get("evidence"):
        st.subheader("Evidence")
        for ev in ticket["evidence"]:
            st.markdown(f"- 📎 `{ev['filename']}` ({ev['content_type']}) — {ev['created_at'][:19]}")

    # All decisions timeline
    if ticket.get("decisions"):
        st.subheader("Decision History")
        for i, d in enumerate(ticket["decisions"], 1):
            with st.expander(f"Decision #{i} — {d['action']}  ({d['status']})"):
                _show_decision(d)

    # If still open, allow follow-up from here too
    if ticket["status"] == "open":
        st.divider()
        latest_dec = ticket["decisions"][-1] if ticket["decisions"] else None
        if latest_dec:
            st.session_state["last_decision"] = latest_dec
        st.session_state["page"] = "new_ticket"
        if st.button("Continue this conversation", type="primary"):
            st.rerun()


def _render_conversation(messages: list):
    for m in messages:
        role = m.get("role", "user")
        content = html.escape(m.get("content", ""))
        if role == "user":
            st.markdown(
                f'<div class="bubble-user">👤 <b>You</b><div class="bubble-content">{content}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bubble-asst">🤖 <b>Assistant</b><div class="bubble-content">{content}</div></div>',
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.title("Navigation")
        if _logged_in():
            st.markdown(f"👤 **{st.session_state.get('user_email','')}**")
            st.divider()
            if st.button("📝 New Ticket", use_container_width=True):
                st.session_state["active_ticket_id"] = None
                st.session_state["last_decision"] = None
                st.session_state["page"] = "new_ticket"
                st.rerun()
            if st.button("📋 History", use_container_width=True):
                st.session_state["page"] = "history"
                st.rerun()
            st.divider()
            if st.button("🚪 Logout", use_container_width=True):
                _logout()
                st.rerun()
        else:
            st.info("Please log in.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init()
    _sidebar()

    if not _logged_in():
        page_auth()
        return

    page = st.session_state.get("page", "new_ticket")
    if page == "new_ticket":
        page_new_ticket()
    elif page == "history":
        page_history()
    elif page == "ticket_detail":
        page_ticket_detail()
    else:
        page_new_ticket()


if __name__ == "__main__":
    main()

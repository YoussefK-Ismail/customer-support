"""
Enterprise Customer Support System
Built with LangChain + GROQ + Streamlit
No PyTorch / No HuggingFace — uses lightweight TF-IDF search
"""

import streamlit as st
import json
import re
from datetime import datetime
from typing import List, Dict

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Support System",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2196F3 100%);
        padding: 20px 30px; border-radius: 12px;
        color: white; margin-bottom: 20px;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Knowledge Bases (3 Products)
# ─────────────────────────────────────────────
KNOWLEDGE_BASES = {
    "CloudStore Pro": """
CloudStore Pro - Enterprise Cloud Storage Solution
Version: 5.2.1 | Released: 2024

FEATURES:
- Unlimited storage with intelligent tiering
- End-to-end AES-256 encryption
- Real-time sync across unlimited devices
- Advanced version history (365 days)
- Team collaboration with granular permissions
- API access with 99.99% uptime SLA

PRICING PLANS:
- Starter: $9/month - 100GB, 3 users
- Business: $49/month - 2TB, 25 users
- Enterprise: $199/month - Unlimited, unlimited users

COMMON ISSUES & SOLUTIONS:
Q: Sync not working?
A: Check firewall settings allow ports 443 and 8443. Restart the sync daemon via Settings > Advanced > Restart Sync. If persists, clear local cache at %AppData%/CloudStore/cache.

Q: Upload speed slow?
A: Enable parallel uploads in Settings > Performance > Multi-thread uploads. Ensure minimum 10Mbps upload bandwidth. Disable bandwidth throttling if enabled.

Q: How to restore deleted files?
A: Go to Web Console > Trash > select files > Restore. Files available for 30 days after deletion. Enterprise users: 365-day retention in Settings > Data Management.

Q: 2FA setup?
A: Account > Security > Enable Two-Factor Auth > Choose authenticator app or SMS. Backup codes available in the same menu.

Q: API rate limits?
A: Starter: 1,000 req/hour. Business: 10,000 req/hour. Enterprise: unlimited.

SUPPORT CONTACTS:
- Priority support: support@cloudstore.com
- Emergency hotline (Enterprise): +1-800-CLOUD-911
- Status page: status.cloudstore.com
""",

    "SecureVault AI": """
SecureVault AI - Intelligent Password & Secrets Manager
Version: 3.8.0 | Released: 2024

FEATURES:
- Zero-knowledge architecture
- AI-powered password strength analysis
- Automatic breach detection (monitors 15B+ leaked credentials)
- Secure sharing with expiring links
- Browser extensions for Chrome, Firefox, Safari, Edge
- SSO integration (SAML 2.0, OIDC)
- Emergency access for trusted contacts

PRICING:
- Personal: Free (50 passwords)
- Premium: $3/month (unlimited)
- Teams: $5/user/month
- Business: $8/user/month (SSO, advanced policies)

COMMON ISSUES & SOLUTIONS:
Q: Can't login to vault?
A: Try master password reset via email verification. Note: Zero-knowledge means we CANNOT recover your master password. Enable emergency access beforehand.

Q: Browser extension not filling passwords?
A: Ensure extension is updated. Check site-specific settings in vault. Try disabling other password managers. Re-authenticate extension from Vault > Connected Apps.

Q: Breach alert received?
A: Go to Security Dashboard > Breached Items > Change password immediately. Use the built-in password generator. Enable real-time monitoring in Settings > Alerts.

Q: SSO configuration?
A: Business plan required. Go to Admin > SSO Configuration > Select provider (Okta, Azure AD, Google Workspace) > Follow SAML/OIDC setup wizard.

Q: Sharing passwords securely?
A: Click item > Share > Set permissions (view/edit) > Set expiry > Generate link. Links auto-expire.

SECURITY CERTIFICATIONS:
SOC 2 Type II, ISO 27001, GDPR compliant, CCPA compliant
""",

    "DataFlow Analytics": """
DataFlow Analytics - Business Intelligence Platform
Version: 2.4.0 | Released: 2024

FEATURES:
- Connect 200+ data sources (SQL, NoSQL, APIs, CSV, Excel)
- Drag-and-drop dashboard builder
- AI-powered insights and anomaly detection
- Automated report scheduling (PDF, Excel, Slack)
- Real-time streaming analytics
- Predictive models with AutoML
- Collaboration with version control for dashboards

PRICING:
- Analyst: $29/month - 5 dashboards, 3 sources
- Professional: $99/month - unlimited dashboards, 20 sources
- Enterprise: $399/month - unlimited everything + dedicated support

COMMON ISSUES & SOLUTIONS:
Q: Database connection failing?
A: Verify credentials and network access. For cloud DBs, whitelist DataFlow IPs: 34.102.x.x range. Check SSL certificate if required.

Q: Dashboard loading slowly?
A: Enable query caching in Dashboard > Settings > Cache. Use data extracts for large datasets. Optimize queries using SQL Analyzer tool.

Q: Scheduled reports not sending?
A: Check email settings in Admin > Notifications. Verify SMTP configuration. Review schedule timezone settings.

Q: How to set up alerts?
A: Dashboard > Alert > Create Alert > Set metric + threshold + condition > Choose notification channel (email/Slack/PagerDuty).

Q: Data not refreshing?
A: Check refresh schedule in Source > Edit > Refresh Settings. Manual refresh available via dashboard refresh button.

INTEGRATIONS:
Salesforce, HubSpot, Google Analytics, Stripe, Snowflake, BigQuery, Redshift, PostgreSQL, MySQL, MongoDB
"""
}

# ─────────────────────────────────────────────
# Lightweight TF-IDF Retrieval (No PyTorch!)
# ─────────────────────────────────────────────
class TFIDFRetriever:
    def __init__(self, knowledge_bases: Dict[str, str], chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunks = []
        self.metadata = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for product, content in knowledge_bases.items():
            for chunk in splitter.split_text(content):
                self.chunks.append(chunk)
                self.metadata.append({"product": product})
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, k: int = 4, product_filter: str = None) -> List[Dict]:
        qvec = self.vectorizer.transform([query])
        scores = cosine_similarity(qvec, self.matrix).flatten()
        # Apply product filter
        if product_filter and product_filter != "All Products":
            for i, meta in enumerate(self.metadata):
                if meta["product"] != product_filter:
                    scores[i] = 0.0
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": self.chunks[idx],
                    "product": self.metadata[idx]["product"],
                    "score": float(scores[idx]),
                })
        return results


@st.cache_resource
def build_retriever():
    return TFIDFRetriever(KNOWLEDGE_BASES)


# ─────────────────────────────────────────────
# Support Tools
# ─────────────────────────────────────────────
@tool
def check_ticket_status(ticket_id: str) -> str:
    """Check the status of a support ticket by ticket ID (e.g. TKT-001)."""
    mock_tickets = {
        "TKT-001": {"status": "Resolved",     "product": "CloudStore Pro",      "issue": "Sync issue",           "resolved_at": "2024-01-15"},
        "TKT-002": {"status": "In Progress",  "product": "SecureVault AI",      "issue": "Browser extension",    "assigned_to": "Tech Team Alpha"},
        "TKT-003": {"status": "Pending",      "product": "DataFlow Analytics",  "issue": "Dashboard performance","eta": "24 hours"},
        "TKT-004": {"status": "Resolved",     "product": "CloudStore Pro",      "issue": "API rate limiting",    "resolved_at": "2024-01-14"},
    }
    key = ticket_id.strip().upper()
    if key in mock_tickets:
        return json.dumps(mock_tickets[key])
    return json.dumps({"error": f"Ticket {ticket_id} not found. Valid IDs: TKT-001 to TKT-004"})


@tool
def calculate_plan_upgrade(current_plan: str, product: str) -> str:
    """Calculate cost increase and benefits when upgrading a subscription plan."""
    upgrades = {
        "CloudStore Pro": {
            "Starter->Business":    {"cost_increase": "$40/month",  "benefits": ["2TB storage", "22 more users", "Priority support", "Admin controls"]},
            "Business->Enterprise": {"cost_increase": "$150/month", "benefits": ["Unlimited storage", "Unlimited users", "365-day retention", "Dedicated CSM"]},
        },
        "SecureVault AI": {
            "Free->Premium":    {"cost_increase": "$3/month",         "benefits": ["Unlimited passwords", "Priority support", "Advanced MFA", "Emergency access"]},
            "Premium->Business":{"cost_increase": "$5/user/month",    "benefits": ["SSO integration", "Admin policies", "Audit logs", "Team management"]},
        },
        "DataFlow Analytics": {
            "Analyst->Professional":    {"cost_increase": "$70/month",  "benefits": ["Unlimited dashboards", "20 data sources", "AutoML", "API access"]},
            "Professional->Enterprise": {"cost_increase": "$300/month", "benefits": ["Unlimited sources", "Dedicated support", "Custom SLA", "On-premise option"]},
        },
    }
    if product in upgrades and current_plan in upgrades[product]:
        return json.dumps(upgrades[product][current_plan])
    available = list(upgrades.get(product, {}).keys())
    return json.dumps({"info": f"Contact sales@enterprise.com for custom pricing.", "available_upgrades": available})


@tool
def generate_support_ticket(issue_summary: str, product: str, severity: str) -> str:
    """Create a new support ticket for an unresolved customer issue."""
    ticket_id = f"TKT-{datetime.now().strftime('%H%M%S')}"
    sla_map = {"Critical": "1 hour", "High": "4 hours", "Medium": "24 hours", "Low": "72 hours"}
    return json.dumps({
        "ticket_id": ticket_id,
        "status": "Created",
        "product": product,
        "summary": issue_summary,
        "severity": severity,
        "sla": sla_map.get(severity, "24 hours"),
        "created_at": datetime.now().isoformat(),
        "message": f"Ticket {ticket_id} created successfully. You will receive email confirmation shortly.",
    })


TOOLS = [check_ticket_status, calculate_plan_upgrade, generate_support_ticket]

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "tickets_created": 0,
        "selected_product": "All Products",
        "memory": ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────
# Build Agent
# ─────────────────────────────────────────────
def build_agent(api_key: str, product_filter: str):
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.2)

    system_prompt = f"""You are an expert Enterprise Customer Support Agent for a software company.
You support three products: CloudStore Pro, SecureVault AI, and DataFlow Analytics.
{'Currently focused on: ' + product_filter if product_filter != 'All Products' else 'You support all three products.'}

Your responsibilities:
1. Answer technical questions using the provided knowledge base context
2. Troubleshoot issues with clear step-by-step guidance
3. Use tools to check ticket status, calculate plan upgrades, or create new support tickets
4. Be professional, empathetic, and solution-focused
5. Format responses with numbered steps when giving instructions
6. Escalate unresolved critical issues by creating a support ticket

Always acknowledge frustration when expressed. End with an offer for further help.
Current date/time: {datetime.now().strftime('%B %d, %Y %H:%M')}
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(
            "Knowledge base context:\n{context}\n\nCustomer question: {input}"
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=False, max_iterations=3)

# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    init_session()

    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🎧 Enterprise Customer Support System</h1>
        <p style="margin:0;opacity:0.9">
            AI-powered support · CloudStore Pro · SecureVault AI · DataFlow Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("GROQ API Key", type="password", placeholder="gsk_...")

        st.divider()
        st.subheader("🎯 Product Filter")
        product_filter = st.selectbox(
            "Select Product",
            ["All Products", "CloudStore Pro", "SecureVault AI", "DataFlow Analytics"],
        )
        st.session_state.selected_product = product_filter

        st.divider()
        st.subheader("📊 Session Stats")
        c1, c2 = st.columns(2)
        c1.metric("Messages", len(st.session_state.messages))
        c2.metric("Tickets",  st.session_state.tickets_created)

        st.divider()
        st.subheader("⚡ Quick Questions")
        quick = [
            "My CloudStore sync stopped working",
            "How do I set up SSO for SecureVault?",
            "DataFlow dashboard is loading slowly",
            "I got a breach alert, what should I do?",
            "Check status of ticket TKT-002",
            "What are CloudStore upgrade options?",
        ]
        for q in quick:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferWindowMemory(
                k=10, memory_key="chat_history", return_messages=True
            )
            st.rerun()

    # ── Product cards ──
    c1, c2, c3 = st.columns(3)
    for col, (name, color, desc) in zip(
        [c1, c2, c3],
        [
            ("☁️ CloudStore Pro",      "#1565c0", "Cloud Storage · Sync · API"),
            ("🔐 SecureVault AI",      "#6a1b9a", "Password Manager · SSO · Breach Detection"),
            ("📊 DataFlow Analytics",  "#1b5e20", "BI Platform · AutoML · Real-time Analytics"),
        ],
    ):
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color}22,{color}11);
                border:1px solid {color}44;border-radius:10px;padding:12px;text-align:center;">
                <b style="color:{color}">{name}</b><br>
                <small style="color:#555">{desc}</small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Chat history ──
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:40px;color:#888;">
            <h3>👋 Welcome to Enterprise Support</h3>
            <p>Ask about technical issues, billing, account settings, or anything else.<br>
            Use the quick questions in the sidebar or type below.</p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🎧"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Input ──
    user_input = st.chat_input("Describe your issue or ask a question…")

    if user_input:
        if not api_key:
            st.error("⚠️ Please enter your GROQ API Key in the sidebar.")
            return

        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Retrieve context via TF-IDF
        retriever = build_retriever()
        results   = retriever.search(user_input, k=4, product_filter=product_filter)
        context   = "\n\n".join([r["content"] for r in results])

        # Generate response
        with st.chat_message("assistant", avatar="🎧"):
            with st.spinner("🤔 Analyzing your issue…"):
                try:
                    agent    = build_agent(api_key, product_filter)
                    output   = agent.invoke({
                        "input":        user_input,
                        "context":      context,
                        "chat_history": st.session_state.memory.chat_memory.messages,
                    })
                    response = output["output"]

                    # Count tickets created
                    if "TKT-" in response and "Created" in response:
                        st.session_state.tickets_created += 1

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.memory.save_context(
                        {"input": user_input}, {"output": response}
                    )

                    # Show sources
                    if results:
                        with st.expander("📚 Knowledge Sources Used"):
                            for i, r in enumerate(results, 1):
                                st.markdown(
                                    f"**Source {i}** — {r['product']} "
                                    f"*(relevance: {r['score']:.2f})*"
                                )
                                st.caption(r["content"][:200] + "…")

                except Exception as e:
                    err = f"⚠️ Error: {str(e)}\n\nPlease check your GROQ API key and try again."
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()

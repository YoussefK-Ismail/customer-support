"""
Enterprise Customer Support System
Built with LangChain + GROQ + Streamlit
"""

import streamlit as st
import json
from datetime import datetime
from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.agents import AgentFinish, AgentAction
from langgraph.prebuilt import create_react_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Enterprise Support System", page_icon="🎧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #1e3a5f 0%, #2196F3 100%);
        padding: 20px 30px; border-radius: 12px; color: white; margin-bottom: 20px; }
</style>""", unsafe_allow_html=True)

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
A: Check firewall settings allow ports 443 and 8443. Restart the sync daemon via Settings > Advanced > Restart Sync. Clear local cache at AppData/CloudStore/cache.
Q: Upload speed slow?
A: Enable parallel uploads in Settings > Performance > Multi-thread uploads. Ensure minimum 10Mbps upload bandwidth.
Q: How to restore deleted files?
A: Go to Web Console > Trash > select files > Restore. Files available for 30 days after deletion.
Q: 2FA setup?
A: Account > Security > Enable Two-Factor Auth > Choose authenticator app or SMS.
Q: API rate limits?
A: Starter: 1,000 req/hour. Business: 10,000 req/hour. Enterprise: unlimited.
SUPPORT CONTACTS:
- Priority support: support@cloudstore.com
- Emergency hotline (Enterprise): +1-800-CLOUD-911
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
PRICING:
- Personal: Free (50 passwords)
- Premium: $3/month (unlimited)
- Teams: $5/user/month
- Business: $8/user/month (SSO, advanced policies)
COMMON ISSUES & SOLUTIONS:
Q: Can't login to vault?
A: Try master password reset via email verification. Zero-knowledge means we CANNOT recover your master password.
Q: Browser extension not filling passwords?
A: Ensure extension is updated. Re-authenticate extension from Vault > Connected Apps.
Q: Breach alert received?
A: Go to Security Dashboard > Breached Items > Change password immediately. Enable real-time monitoring in Settings > Alerts.
Q: SSO configuration?
A: Business plan required. Go to Admin > SSO Configuration > Select provider > Follow setup wizard.
SECURITY: SOC 2 Type II, ISO 27001, GDPR compliant
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
PRICING:
- Analyst: $29/month - 5 dashboards, 3 sources
- Professional: $99/month - unlimited dashboards, 20 sources
- Enterprise: $399/month - unlimited everything + dedicated support
COMMON ISSUES & SOLUTIONS:
Q: Database connection failing?
A: Verify credentials. Whitelist DataFlow IPs: 34.102.x.x range. Check SSL certificate.
Q: Dashboard loading slowly?
A: Enable query caching in Dashboard > Settings > Cache. Use data extracts for large datasets.
Q: Scheduled reports not sending?
A: Check email settings in Admin > Notifications. Verify SMTP configuration.
Q: How to set up alerts?
A: Dashboard > Alert > Create Alert > Set metric + threshold > Choose notification channel.
INTEGRATIONS: Salesforce, HubSpot, Google Analytics, Stripe, Snowflake, BigQuery, PostgreSQL, MySQL, MongoDB
"""
}

class TFIDFRetriever:
    def __init__(self, knowledge_bases):
        self.chunks, self.metadata = [], []
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        for product, content in knowledge_bases.items():
            for chunk in splitter.split_text(content):
                self.chunks.append(chunk)
                self.metadata.append({"product": product})
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query, k=4, product_filter=None):
        qvec = self.vectorizer.transform([query])
        scores = cosine_similarity(qvec, self.matrix).flatten()
        if product_filter and product_filter != "All Products":
            for i, meta in enumerate(self.metadata):
                if meta["product"] != product_filter:
                    scores[i] = 0.0
        top_idx = np.argsort(scores)[::-1][:k]
        return [{"content": self.chunks[i], "product": self.metadata[i]["product"], "score": float(scores[i])}
                for i in top_idx if scores[i] > 0]

@st.cache_resource
def build_retriever():
    return TFIDFRetriever(KNOWLEDGE_BASES)

@tool
def check_ticket_status(ticket_id: str) -> str:
    """Check the status of a support ticket by ticket ID (e.g. TKT-001)."""
    tickets = {
        "TKT-001": {"status": "Resolved",    "product": "CloudStore Pro",     "issue": "Sync issue",           "resolved_at": "2024-01-15"},
        "TKT-002": {"status": "In Progress", "product": "SecureVault AI",     "issue": "Browser extension",    "assigned_to": "Tech Team Alpha"},
        "TKT-003": {"status": "Pending",     "product": "DataFlow Analytics", "issue": "Dashboard performance","eta": "24 hours"},
        "TKT-004": {"status": "Resolved",    "product": "CloudStore Pro",     "issue": "API rate limiting",    "resolved_at": "2024-01-14"},
    }
    return json.dumps(tickets.get(ticket_id.strip().upper(), {"error": "Ticket not found. Valid: TKT-001 to TKT-004"}))

@tool
def calculate_plan_upgrade(current_plan: str, product: str) -> str:
    """Calculate cost and benefits for upgrading a subscription plan."""
    upgrades = {
        "CloudStore Pro": {
            "Starter->Business":    {"cost_increase": "$40/month",     "benefits": ["2TB storage", "22 more users", "Priority support"]},
            "Business->Enterprise": {"cost_increase": "$150/month",    "benefits": ["Unlimited storage", "Unlimited users", "365-day retention"]},
        },
        "SecureVault AI": {
            "Free->Premium":        {"cost_increase": "$3/month",      "benefits": ["Unlimited passwords", "Priority support", "Advanced MFA"]},
            "Premium->Business":    {"cost_increase": "$5/user/month", "benefits": ["SSO integration", "Admin policies", "Audit logs"]},
        },
        "DataFlow Analytics": {
            "Analyst->Professional":    {"cost_increase": "$70/month",  "benefits": ["Unlimited dashboards", "20 data sources", "AutoML"]},
            "Professional->Enterprise": {"cost_increase": "$300/month", "benefits": ["Unlimited sources", "Dedicated support", "Custom SLA"]},
        },
    }
    if product in upgrades and current_plan in upgrades[product]:
        return json.dumps(upgrades[product][current_plan])
    return json.dumps({"info": "Contact sales@enterprise.com", "available": list(upgrades.get(product, {}).keys())})

@tool
def generate_support_ticket(issue_summary: str, product: str, severity: str) -> str:
    """Create a new support ticket for a customer issue."""
    sla = {"Critical": "1 hour", "High": "4 hours", "Medium": "24 hours", "Low": "72 hours"}
    return json.dumps({
        "ticket_id": f"TKT-{datetime.now().strftime('%H%M%S')}",
        "status": "Created", "product": product, "summary": issue_summary,
        "severity": severity, "sla": sla.get(severity, "24 hours"),
    })

TOOLS = [check_ticket_status, calculate_plan_upgrade, generate_support_ticket]

def init_session():
    if "messages"        not in st.session_state: st.session_state.messages        = []
    if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
    if "tickets_created" not in st.session_state: st.session_state.tickets_created = 0

def get_ai_response(api_key: str, user_input: str, context: str, chat_history: list, product_filter: str) -> str:
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.2)

    system = f"""You are an expert Enterprise Customer Support Agent.
You support: CloudStore Pro, SecureVault AI, and DataFlow Analytics.
{'Focused on: ' + product_filter if product_filter != 'All Products' else 'Supporting all three products.'}

You have access to these tools:
- check_ticket_status: to look up ticket status by ID
- calculate_plan_upgrade: to show upgrade pricing and benefits
- generate_support_ticket: to create a new support ticket

Instructions:
1. Use the knowledge base context to answer technical questions
2. Give clear numbered troubleshooting steps
3. Call tools when the user asks about tickets or upgrades
4. Be professional and empathetic
5. End every response with an offer for further help

Current time: {datetime.now().strftime('%B %d, %Y %H:%M')}"""

    agent = create_react_agent(llm, TOOLS, prompt=system)

    messages = []
    for msg in chat_history[-10:]:
        messages.append(msg)
    messages.append(HumanMessage(content=f"Knowledge base context:\n{context}\n\nCustomer question: {user_input}"))

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content

def main():
    init_session()

    st.markdown("""<div class="main-header">
        <h1>🎧 Enterprise Customer Support System</h1>
        <p style="margin:0;opacity:0.9">AI-powered support · CloudStore Pro · SecureVault AI · DataFlow Analytics</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("GROQ API Key", type="password", placeholder="gsk_...")
        st.divider()
        st.subheader("🎯 Product Filter")
        product_filter = st.selectbox("Select Product",
            ["All Products", "CloudStore Pro", "SecureVault AI", "DataFlow Analytics"])
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Messages", len(st.session_state.messages))
        c2.metric("Tickets",  st.session_state.tickets_created)
        st.divider()
        st.subheader("⚡ Quick Questions")
        for q in [
            "My CloudStore sync stopped working",
            "How do I set up SSO for SecureVault?",
            "DataFlow dashboard is loading slowly",
            "I got a breach alert, what should I do?",
            "Check status of ticket TKT-002",
            "Upgrade options for CloudStore Pro?",
        ]:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    for col, (name, color, desc) in zip(st.columns(3), [
        ("☁️ CloudStore Pro",     "#1565c0", "Cloud Storage · Sync · API"),
        ("🔐 SecureVault AI",     "#6a1b9a", "Password Manager · SSO · Breach Detection"),
        ("📊 DataFlow Analytics", "#1b5e20", "BI Platform · AutoML · Real-time Analytics")]):
        with col:
            st.markdown(f"""<div style="background:linear-gradient(135deg,{color}22,{color}11);
                border:1px solid {color}44;border-radius:10px;padding:12px;text-align:center;">
                <b style="color:{color}">{name}</b><br><small style="color:#555">{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.divider()

    if not st.session_state.messages:
        st.markdown("""<div style="text-align:center;padding:40px;color:#888;">
            <h3>👋 Welcome to Enterprise Support</h3>
            <p>Ask about technical issues, billing, or account settings.</p>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🎧"):
            st.markdown(msg["content"])

    user_input = st.chat_input("Describe your issue or ask a question…")

    if user_input:
        if not api_key:
            st.error("⚠️ Please enter your GROQ API Key in the sidebar.")
            return

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        retriever = build_retriever()
        results   = retriever.search(user_input, k=4, product_filter=product_filter)
        context   = "\n\n".join([r["content"] for r in results])

        with st.chat_message("assistant", avatar="🎧"):
            with st.spinner("🤔 Analyzing your issue…"):
                try:
                    response = get_ai_response(api_key, user_input, context,
                                               st.session_state.chat_history, product_filter)
                    if "TKT-" in response and "Created" in response:
                        st.session_state.tickets_created += 1
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(AIMessage(content=response))
                    if len(st.session_state.chat_history) > 20:
                        st.session_state.chat_history = st.session_state.chat_history[-20:]
                    if results:
                        with st.expander("📚 Knowledge Sources Used"):
                            for i, r in enumerate(results, 1):
                                st.markdown(f"**Source {i}** — {r['product']} *(relevance: {r['score']:.2f})*")
                                st.caption(r["content"][:200] + "…")
                except Exception as e:
                    err = f"⚠️ Error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

if __name__ == "__main__":
    main()

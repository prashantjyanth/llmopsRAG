# import os
# import requests
# import streamlit as st

# ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8001")

# st.title("Multi-Agent Travel Planner")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Input box
# user_input = st.text_input("You:", "")

# if st.button("Send") and user_input:
#     # Append user message only once
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     payload = {"message": user_input, "thread_id": "ui"}
#     try:
#         response = requests.post(ORCHESTRATOR_URL, json=payload)
#         data = response.json()

#         # Process orchestrator responses (assistant/tool/etc.)
#         for msg in data.get("messages", []):
#             if "role" in msg:
#                 role = msg["role"]
#             elif msg.get("type") == "human":
#                 role = "user"
#             elif msg.get("type") == "ai":
#                 role = "assistant"
#             elif msg.get("type") == "tool":
#                 role = "tool"
#             else:
#                 role = "assistant"  # fallback

#             content = msg.get("content", "")
#             st.session_state.messages.append({"role": role, "content": content})

#     except Exception as e:
#         st.error(f"Error: {e}")

# # --- Show conversation (latest first) ---
# st.write("---")
# for msg in reversed(st.session_state.messages):  # <-- latest first
#     st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")

import os
import requests
import streamlit as st
import yaml
with open("configs/agents.yaml", "r") as f:
    config = yaml.safe_load(f)
agents = config.get("agents", {})
agent_names = list(agents.keys())

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8001")
EVALUATOR_URL = os.getenv("EVALUATOR_URL", "http://evaluation:8101")  # FastAPI evaluator service

st.title("Multi-Agent Travel Planner")

# --- Tabs ---
tab1, tab2 = st.tabs(["💬 Chat", "📊 Evaluation"])

# -------------------------------
# Tab 1: Chat
# -------------------------------
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("You:", "")

    if st.button("Send", key="chat_send") and user_input:
        # st.session_state.messages.append({"role": "user", "content": user_input})

        payload = {"message": user_input, "thread_id": "ui"}
        try:
            response = requests.post(ORCHESTRATOR_URL, json=payload)
            data = response.json()

            for msg in data.get("messages", []):
                if "role" in msg:
                    role = msg["role"]
                elif msg.get("type") == "human":
                    role = "user"
                elif msg.get("type") == "ai":
                    role = "assistant"
                elif msg.get("type") == "tool":
                    role = "tool"
                else:
                    role = "assistant"

                content = msg.get("content", "")
                st.session_state.messages.append({"role": role, "content": content})

        except Exception as e:
            st.error(f"Error: {e}")

    st.write("---")
    for msg in reversed(st.session_state.messages):
        st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")

# -------------------------------
# Tab 2: Evaluation
# -------------------------------
with tab2:
    st.subheader("Evaluate Agent with CSV")

    agent_name = st.selectbox("Select Agent", agent_names)
    uploaded_file = st.file_uploader("Upload CSV (agent_name, question, expected)", type=["csv"])

    if uploaded_file is not None and st.button("Evaluate", key="eval_button"):
        try:
            files = {"file": uploaded_file.getvalue()}
            data = {"agent_name": agent_name}
            resp = requests.post(f"{EVALUATOR_URL}/evaluate/agent", files=files, data=data)

            if resp.status_code == 200:
                result = resp.json()
                st.success("✅ Evaluation Completed")
                st.json(result)
            else:
                st.error(f"❌ Evaluation failed: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")
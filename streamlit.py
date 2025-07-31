import streamlit as st
import requests
import json
import time

# --- Configuration ---
API_BASE = "http://127.0.0.1:8000/api"
session_id_key = "user_session_id"
st.set_page_config(page_title="Smart Chat App", layout="wide")

# --- Session Init ---
if session_id_key not in st.session_state:
    st.session_state[session_id_key] = None
if "pdf_uploaded" not in st.session_state:
    st.session_state["pdf_uploaded"] = False
if "yt_embedded" not in st.session_state:
    st.session_state["yt_embedded"] = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Login UI ---
def login():
    st.title("🔐 Login")
    username = st.text_input("Enter a username to login")
    if st.button("Login"):
        if username:
            st.session_state[session_id_key] = f"{username}_session"
            st.success("✅ Logged in!")
            st.rerun()
        else:
            st.warning("Username is required")

# --- Main App UI ---
def main_app():
    st.sidebar.title("📚 Document Tools")

    # Upload PDF
    st.sidebar.subheader("Upload PDF")
    uploaded_pdf = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_pdf and not st.session_state["pdf_uploaded"]:
        with st.spinner("Uploading and embedding..."):
            files = {'file': (uploaded_pdf.name, uploaded_pdf, 'application/pdf')}
            res = requests.post(f"{API_BASE}/upload_pdf", files=files)
            if res.status_code == 200:
                st.sidebar.success("✅ PDF embedded successfully")
                st.session_state["pdf_uploaded"] = True
            else:
                st.sidebar.error(f"❌ Upload failed: {res.text}")

    # YouTube Link Embedding
    st.sidebar.subheader("Embed from YouTube")
    yt_link = st.sidebar.text_input("Paste YouTube link here")
    if st.sidebar.button("Submit YouTube Link") and yt_link and not st.session_state["yt_embedded"]:
        with st.spinner("Processing YouTube link..."):
            res = requests.post(f"{API_BASE}/youtube_link_embedding", json={"url": yt_link})
            if res.status_code == 200:
                st.sidebar.success("✅ You can now chat with the YouTube video")
                st.session_state["yt_embedded"] = True
            else:
                st.sidebar.error(f"❌ YouTube processing failed: {res.text}")

    # --- Chat UI ---
    st.title("💬 Smart Chat")

    # Display previous chat history
    for sender, message in st.session_state.chat_history:
        if sender == "user":
            st.chat_message("user", avatar="🧑").markdown(message)
        else:
            st.chat_message("assistant", avatar="🤖").markdown(message)

    # Chat input box
    user_input = st.chat_input("Type your question")

    if user_input:
        # Save user message
        st.session_state.chat_history.append(("user", user_input))

        # Display assistant container
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            context_placeholder = st.container()
            response_placeholder.markdown("🤔 Thinking...")

            headers = {"session-id": st.session_state[session_id_key]}

            try:
                response_text = ""
                context_text = ""

                with requests.post(
                    f"{API_BASE}/llm_response",
                    json={"user_input": user_input},
                    headers=headers,
                    stream=True,
                    timeout=60
                ) as res:

                    if res.status_code == 200:
                        for line in res.iter_lines(decode_unicode=True):
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    data_type = data.get("type", "chunk")

                                    if data_type == "context":
                                        context_text = data.get("context", "")
                                        context_placeholder.info(f"📄 **Context:** {context_text}")

                                    elif data_type == "chunk":
                                        text_chunk = data.get("response", "")
                                        response_text += text_chunk
                                        response_placeholder.markdown(response_text + "▌")

                                    elif data_type == "complete":
                                        response_placeholder.markdown(response_text)
                                        st.success("✅ Response completed!")
                                        break

                                    elif data_type == "error":
                                        st.error(f"❌ Error: {data.get('error', 'Unknown error')}")
                                        break

                                except json.JSONDecodeError:
                                    st.warning(f"⚠️ Malformed JSON: {line}")
                                except Exception as e:
                                    st.error(f"⚠️ Error processing: {str(e)}")
                    else:
                        st.error(f"❌ Server Error ({res.status_code}): {res.text}")

                # Save assistant response
                st.session_state.chat_history.append(("assistant", response_text))

            except requests.exceptions.Timeout:
                st.error("🕐 Request timed out.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Connection error.")
            except Exception as e:
                st.error(f"🚨 Unexpected error: {str(e)}")

# --- Run App ---
if st.session_state[session_id_key] is None:
    login()
else:
    main_app()

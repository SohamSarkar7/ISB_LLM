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

    # Chat UI
    st.title("💬 Ask a Question")
    user_input = st.text_input("Type your question", key="user_question")

    if st.button("Ask"):
        if user_input:
            # Create containers for response and context
            response_container = st.container()
            context_container = st.container()
            
            with response_container:
                st.markdown("### 🧠 Response:")
                response_placeholder = st.empty()
                
            with context_container:
                context_placeholder = st.empty()
            
            # Stream the response
            headers = {"session-id": st.session_state[session_id_key]}
            
            try:
                response_text = ""
                context_text = ""
                
                with requests.post(
                    f"{API_BASE}/llm_response",
                    json={"user_input": user_input},
                    headers=headers,
                    stream=True,
                    timeout=60  # Add timeout
                ) as res:
                    
                    if res.status_code == 200:
                        for line in res.iter_lines(decode_unicode=True):
                            if line.strip():  # Skip empty lines
                                try:
                                    data = json.loads(line)
                                    data_type = data.get("type", "chunk")
                                    
                                    if data_type == "context":
                                        context_text = data.get("context", "")
                                        with context_container:
                                            context_placeholder.info(f"📄 **Context:** {context_text}")
                                    
                                    elif data_type == "chunk":
                                        text_chunk = data.get("response", "")
                                        response_text += text_chunk
                                        
                                        # Update the response in real-time
                                        with response_container:
                                            response_placeholder.markdown(response_text + "▌")  # Add cursor
                                        
                                        # Small delay for better visual effect
                                        time.sleep(0.01)
                                    
                                    elif data_type == "complete":
                                        # Remove cursor when complete
                                        with response_container:
                                            response_placeholder.markdown(response_text)
                                        st.success("✅ Response completed!")
                                        break
                                    
                                    elif data_type == "error":
                                        error_msg = data.get("error", "Unknown error")
                                        st.error(f"❌ Error: {error_msg}")
                                        break
                                        
                                except json.JSONDecodeError as e:
                                    st.warning(f"⚠️ Received malformed JSON: {line}")
                                    continue
                                except Exception as e:
                                    st.error(f"⚠️ Error processing line: {str(e)}")
                                    continue
                    else:
                        st.error(f"❌ Server Error ({res.status_code}): {res.text}")
                        
            except requests.exceptions.Timeout:
                st.error("🕐 Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Connection error. Please check if the API server is running.")
            except Exception as e:
                st.error(f"🚨 Unexpected error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question first.")

# --- Run App ---
if st.session_state[session_id_key] is None:
    login()
else:
    main_app()
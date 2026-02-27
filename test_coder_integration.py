import streamlit as st 
from MainAgent import full_agent

st.title("🏀 Shot Quality Analysis Agent")
st.markdown("Ask questions about NBA players' shot data, request analysis, or create visualizations.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize thread config for agent continuity
if "config" not in st.session_state:
    st.session_state.config = {
        "configurable": {
            "thread_id": "streamlit_session"
        }
    }

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about NBA shot data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare input for agent
            inputs = {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Stream agent response
            response_text = ""
            placeholder = st.empty()
            
            try:
                for s in full_agent.stream(inputs, stream_mode="values", config=st.session_state.config):
                    message = s["messages"][-1]
                    
                    # Extract content from message
                    if hasattr(message, "content"):
                        content = message.content
                        if isinstance(content, str) and content.strip():
                            response_text = content
                            placeholder.markdown(response_text)
                
                # If no response text, show a default message
                if not response_text:
                    response_text = "Processing complete. Please check the output above."
                    placeholder.markdown(response_text)
                    
            except Exception as e:
                response_text = f"Error: {str(e)}"
                placeholder.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Sidebar with instructions
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    **Example Queries:**
    - "Analyze Devin Booker's shots in game 16 of his 2022-23 season"
    - "How did defenders guard him?"
    - "Create a shot chart"
    - "Show me LeBron's game against the Lakers"
    
    **Tips:**
    - The agent maintains conversation context
    - You can ask follow-up questions
    - Request visualizations anytime after loading data
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.config = {
            "configurable": {
                "thread_id": "streamlit_session"
            }
        }
        st.rerun()
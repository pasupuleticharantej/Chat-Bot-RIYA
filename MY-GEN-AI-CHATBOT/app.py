import os
import re
import gradio as gr
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import login

# Ensure required libraries are installed
os.system("pip install langchain langchain-community langchain-core huggingface_hub gradio")

# Load Model from Hugging Face
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1")

# Define Chatbot Personalities
PERSONAS = {
    "Casual": "You are Riya, a witty and energetic personal assistant who responds in a fun and engaging manner.",
    "Professional": "You are Riya, a highly professional assistant who provides clear and concise answers.",
    "Sarcastic": "You are Riya, a humorous assistant who responds with witty and sarcastic remarks.",
    "Motivational": "You are Riya, an encouraging assistant who provides uplifting and inspiring messages.",
    "Tech-Savvy": "You are Riya, a knowledgeable AI who gives detailed tech-related insights."
}

# Define Prompt Template
prompt_template = """Your persona is: {persona}.
Respond in this style without mentioning your persona explicitly.

User: {user_message}
Chatbot: """

prompt = PromptTemplate(
    input_variables=["user_message", "persona"], template=prompt_template
)

# Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_message")

# Initialize LLM Chain
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

# Chat History Storage
chat_history = []

# Function to Get AI Response
def get_text_response(user_message, persona):
    persona_style = PERSONAS.get(persona, PERSONAS["Casual"])
    response = llm_chain.predict(user_message=user_message, persona=persona_style)
    chatbot_response = re.split(r'Chatbot:\s?', response, maxsplit=1)[-1].strip()
    chat_history.append([user_message, chatbot_response])  
    return chat_history

# Function to Show Chat History
def show_chat_history():
    if chat_history:
        return "\n\n".join([f"User: {u}\nChatbot: {r}" for u, r in chat_history])
    return "No history available."

# Function to Reset Chat (Clears UI but Keeps History)
def reset_chat():
    memory.clear()  # Clear LangChain's memory to reset conversation
    return [], ""  # Clears chatbot UI and input box but keeps history available

# Function to Hide Chat History
def hide_chat_history():
    return gr.update(visible=False)

# Set Up Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Riya - Your AI Assistant")
    gr.Markdown("### Choose a chatbot personality & start chatting!")

    with gr.Row():
        persona_selector = gr.Dropdown(
            choices=list(PERSONAS.keys()), value="Casual", label="Chatbot Personality"
        )
    
    chatbot = gr.Chatbot(label="Chat with Riya")
    user_input = gr.Textbox(placeholder="Type your message...")
    send_button = gr.Button("Send")
    
    with gr.Row():
        reset_button = gr.Button("Reset Chat", variant="secondary")
        show_history_button = gr.Button("Show History", variant="primary")

    with gr.Column(visible=False) as history_section:
        history_output = gr.Textbox(label="Chat History", interactive=False, lines=10)
        clear_history_button = gr.Button("Clear History", variant="secondary")
        close_history_button = gr.Button(" CLOSE ", elem_id="close-btn")
    
    # Inject inline CSS for the close button
    gr.HTML("""
    <style>
        #close-btn {
            background-color: red !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 6px 12px;
            border: none;
            cursor: pointer;
        }
    </style>
    """)

    # Button Click Actions
    send_button.click(fn=get_text_response, inputs=[user_input, persona_selector], outputs=[chatbot])
    reset_button.click(fn=reset_chat, inputs=[], outputs=[chatbot, user_input])
    show_history_button.click(fn=show_chat_history, inputs=[], outputs=[history_output]).then(
        lambda: gr.update(visible=True), inputs=[], outputs=[history_section]
    )
    close_history_button.click(fn=hide_chat_history, inputs=[], outputs=[history_section])
    clear_history_button.click(fn=reset_chat, inputs=[], outputs=[history_output])

# Launch Gradio App
demo.launch(share=True, debug=False)

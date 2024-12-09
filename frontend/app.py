import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Financial Q&A Chatbot",
    page_icon="💵",
    layout="wide",
)

# Check if 'menu_selection' exists in session state, if not, set it to "Home"
if "menu_selection" not in st.session_state:
    st.session_state.menu_selection = "Home"

# Check if file is already uploaded in session state
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None  # No file uploaded by default

# Function to log the sidebar state in the browser's console
def log_sidebar_state_to_console():
    sidebar_state = {
        'menu_selection': st.session_state.menu_selection,
    }
    sidebar_state_json = json.dumps(sidebar_state, default=str)
    
    # Inject JavaScript to log the session state to the console
    st.markdown(f"""
        <script>
        console.log("MoneyChat Sidebar State: ", {sidebar_state_json});
        </script>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("MoneyChat Navigation 💬")
    # Use the session state value to pre-select the sidebar option
    menu_selection = st.radio(
        "Explore the sections:",
        options=["Home", "Upload Financial Data", "Ask Financial Questions", "Visualizations and Insights"],
        index=["Home", "Upload Financial Data", "Ask Financial Questions", "Visualizations and Insights"].index(st.session_state.menu_selection),
    )
    # Store the selected option in session state
    st.session_state.menu_selection = menu_selection

# Main Page Content
if menu_selection == "Home":
    st.title("Welcome to MoneyChat 💵🤖")
    st.write("""
        MoneyChat is your personal financial assistant.
        - Upload your financial data to get started.
        - Ask specific questions to gain insights.
        - Visualize trends and uncover meaningful patterns.
        Use the menu on the left to navigate through the features.
    """)

elif menu_selection == "Upload Financial Data":
    st.title("Upload Your Financial Data 📂")
    uploaded_file = st.file_uploader("Upload a file (e.g., CSV, Excel, or PDF):", type=["csv", "xlsx", "txt", "pdf"])

    # If a file is uploaded, save it in the session state to persist after refresh
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.success("Your file has been uploaded successfully!")
    elif st.session_state.uploaded_file:
        st.warning("Your previously uploaded file is still available.")
    else:
        st.info("Please upload a file to proceed.")
        
    # You can allow the user to interact with the uploaded file, e.g., preview it.
    if st.session_state.uploaded_file:
        st.write(f"File name: {st.session_state.uploaded_file.name}")

elif menu_selection == "Ask Financial Questions":
    st.title("Ask Your Financial Questions 💬")
    
    # Initialize session state for storing questions and answers
    if "questions" not in st.session_state:
        st.session_state.questions = []  # List to store user questions
    if "answers" not in st.session_state:
        st.session_state.answers = []  # List to store model-generated answers

    # Input for user query
    user_query = st.text_input("Type your question about finances:")

    # Submit button to handle query
    if st.button("Submit"):
        if user_query.strip():
            # Add question and placeholder answer to session state
            st.session_state.questions.append(user_query)
            generated_answer = f"Model's response to: '{user_query}'"  # Replace with model logic
            st.session_state.answers.append(generated_answer)
        else:
            st.warning("Please enter a question before submitting.")

    # Display all questions and their corresponding answers
    if st.session_state.questions:
        st.write("### Previous Questions and Answers")
        for i, question in enumerate(st.session_state.questions):
            st.write(f"**Q{i + 1}:** {question}")
            st.write(f"**A{i + 1}:** {st.session_state.answers[i]}")
    else:
        st.info("No questions asked yet. Start typing above!")

elif menu_selection == "Visualizations and Insights":
    st.title("Explore Visualizations and Insights 📊")

    # Placeholder for insights and charts
    st.write("""
        This section provides meaningful insights and trends based on your financial data.
    """)
    
    # Example Visualization (Replace with actual charts)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Sample data for visualization
    st.subheader("Sample Expense Categories")
    data = {
        "Category": ["Housing", "Transportation", "Food", "Utilities", "Entertainment"],
        "Expenses": [1200, 300, 500, 200, 150],
    }
    df = pd.DataFrame(data)
    
    st.bar_chart(df.set_index("Category"))

    # Placeholder for user-uploaded file insights
    if st.session_state.uploaded_file:
        st.write("Visualizations based on your uploaded data will appear here.")
    else:
        st.info("Upload financial data to generate personalized insights and visualizations.")

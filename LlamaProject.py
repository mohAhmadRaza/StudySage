import streamlit as st
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool, tool
import os
from dotenv import load_dotenv
import time

# Initialize session state for the sidebar buttons
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = "Home"  # Default view

# Sidebar with buttons
st.sidebar.header("Menu")
if st.sidebar.button("Home"):
    st.session_state.button_clicked = "Home"
if st.sidebar.button("Student's Solutions"):
    st.session_state.button_clicked = "Student's Solutions"
if st.sidebar.button("Contact"):
    st.session_state.button_clicked = "Contact"

# Default content
if st.session_state.button_clicked == "Home":
    st.markdown("""
    <div style="background-color:Black; border-radius:5px; padding:10px;margin-bottom:35px;">
       <h1 style="color: white; font-size:50px; text-align:center;">AI StudySage</h1> 
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "This app is an AI-powered assistant designed to help you achieve your goals by providing detailed solutions and resources.")

    st.header("Features")
    st.markdown("""
    - **Goal-Oriented Responses:** Get detailed solutions to your problems.
    - **Resource Integration:** Access relevant YouTube videos and articles.
    - **User-Friendly Interface:** Easy to use and navigate.
    - **Decision Making:** Helps in building decisions and helps to achieve goals.
    - **Formatted Blogs:** Here you can find formatted blogs highlighting solutions to your problems.
    """)

if st.session_state.button_clicked == "Student's Solutions":
    # Load API from environment as it is secret information
    load_dotenv()
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    # Create A Language model for further processing
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=GROQ_API_KEY)

    def create_agent(role, goal, backstory):
        return Agent(
            llm=llm,
            role=role,
            goal=goal,
            backstory=backstory,
            allow_delegation=False,
            verbose=True,
        )

    planner = create_agent(
        role="Content Planner",
        goal="Plan engaging and factually accurate content on {topic}",
        backstory="You are planning a blog article about {topic}. You collect information that helps the audience learn and make informed decisions. Your work serves as a foundation for the Content Writer.",
    )

    writer = create_agent(
        role="Content Writer",
        goal="Write an insightful and factually accurate opinion piece on {topic}",
        backstory="You are writing an opinion piece on {topic}, based on the planner's outline. You provide objective insights and acknowledge opinions.",
    )

    editor = create_agent(
        role="Editor",
        goal="Edit the blog post to align with the organization's writing style.",
        backstory="You review the blog post from the writer, ensuring it follows best practices, provides balanced viewpoints, and avoids major controversial topics.",
    )

    def create_task(description, expected_output, agent):
        return Task(description=description, expected_output=expected_output, agent=agent)

    plan = create_task(
        description=(
            "1. Prioritize the latest trends, key players, and news on {topic}.\n"
            "2. Identify the target audience, their interests, and pain points.\n"
            "3. Develop a detailed content outline with an introduction, key points, and a call to action.\n"
            "4. Include SEO keywords and relevant data or sources."
        ),
        expected_output="A comprehensive content plan with an outline, audience analysis, SEO keywords, and resources.",
        agent=planner,
    )

    write = create_task(
        description=(
            "1. Use the content plan to craft a compelling blog post on {topic}.\n"
            "2. Incorporate SEO keywords naturally.\n"
            "3. Name sections/subtitles engagingly.\n"
            "4. Structure the post with an engaging introduction, insightful body, and summarizing conclusion.\n"
            "5. Proofread for grammatical errors and brand voice alignment."
        ),
        expected_output="A well-written blog post in markdown format, ready for publication, with each section having 2-3 paragraphs.",
        agent=writer,
    )

    edit = create_task(
        description="Proofread the given blog post for grammatical errors and brand voice alignment.",
        expected_output="A well-written blog post in markdown format, ready for publication, with each section having 2-3 paragraphs.",
        agent=editor,
    )

    crew = Crew(agents=[planner, writer, editor], tasks=[plan, write, edit], verbose=2)

    st.markdown("""
    <div style="background-color:Black; border-radius:5px; padding:10px;margin-bottom:35px;">
       <h1 style="color: white; font-size:50px; text-align:center;">AI Content Creation</h1> 
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for all input fields
    if "grade_level" not in st.session_state:
        st.session_state.grade_level = ""
    if "favorite_subjects" not in st.session_state:
        st.session_state.favorite_subjects = ""
    if "performance" not in st.session_state:
        st.session_state.performance = ""
    if "goals" not in st.session_state:
        st.session_state.goals = ""
    if "interest" not in st.session_state:
        st.session_state.interest = ""
    if "challenges" not in st.session_state:
        st.session_state.challenges = ""

    # Input fields
    st.session_state.grade_level = st.text_input("What's the current grade/year level?", st.session_state.grade_level)
    st.session_state.favorite_subjects = st.text_input("What's your favorite subjects?", st.session_state.favorite_subjects)
    st.session_state.performance = st.text_input("What's your current performance?", st.session_state.performance)
    st.session_state.goals = st.text_input("What's your long-term academic goals?", st.session_state.goals)
    st.session_state.interest = st.text_input("In which fields of study are you interested?", st.session_state.interest)
    st.session_state.challenges = st.text_input("What are the challenges you are currently facing?", st.session_state.challenges)

    topic = (
        f"The student is currently in {st.session_state.grade_level}.\n"
        f"Their favorite subjects are {st.session_state.favorite_subjects}.\n"
        f"Their current performance is {st.session_state.performance}.\n"
        f"Their long-term academic goals are {st.session_state.goals}.\n"
        f"They are interested in the following fields of study: {st.session_state.interest}.\n"
        f"They are currently facing these challenges: {st.session_state.challenges}.\n\n"
        "Based on this information, provide advice and suggestions to help the student improve their academic performance and achieve their long-term goals. "
        "Consider their interests and the challenges they are facing. Add the proper headings, hyperlinks to the books you refer to read, any website links, or any other resources you can suggest."
    )

    if st.button("Start Workflow"):
        with st.spinner("Running the content creation workflow..."):
            result = crew.kickoff(inputs={"topic": topic})
        result_str = str(result)  # Convert to string if it’s a textual representation
        result_bytes = result_str.encode('utf-8')  # Convert string to bytes
        st.sidebar.download_button(
            label="Download Response",
            data=result_bytes,
            file_name="ChatBot_Response.txt"  # Use .txt or another appropriate extension
        )
        st.markdown(result)
        st.success("Workflow completed")

import streamlit as st

# Check if the button for 'Contact' is clicked
if st.session_state.button_clicked == "Contact":
    st.title(":mailbox: Get in touch with us!")
    st.write(
        "If you are not getting required results from this app, don't feel sad, contact us on our given email any time of the week and any time of the day.")

    # Add custom CSS
    custom_css = """
    <style>
    .contact-form {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .contact-form input, .contact-form textarea {
        width: 100%;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    .contact-form button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .contact-form button:hover {
        background-color: #45a049;
    }
    .contact-form textarea {
        height: 150px;
        resize: none;
    }
    </style>
    """

    # Contact form with CSS class
    contact_form = """
    <form action="https://formsubmit.co/sktfscm21557034@gmail.com" method="POST" class="contact-form">
         <input type="text" name="Name" placeholder="Your Name" required>
         <input type="email" name="email" placeholder="Your Email" required> 
         <textarea name="message" placeholder="Details Of Your Problem"></textarea> 
         <button type="submit">Send</button>
    </form>     
    """

    # Combine the CSS and form HTML
    form_with_css = custom_css + contact_form

    # Render the form
    st.markdown(form_with_css, unsafe_allow_html=True)

    # Add some spacing
    st.write(" \n" "")

    # Add LinkedIn link
    st.markdown('[LinkedIn](https://www.linkedin.com/in/mohAhmadRaza) :house:')


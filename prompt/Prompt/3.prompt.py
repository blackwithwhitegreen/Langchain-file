from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import streamlit as st 
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv(find_dotenv())

st.header("TOOl")

model = ChatGoogleGenerativeAI(
    model = "models/gemini-1.5-flash-latest",
    temperature = 1.5
)

poem_type = st.selectbox(
    "Select Poem Type",
    [
        "Love",
        "Anger",
        "Anxiety",
        "Loneliness",
        "Happiness"
    ]
)

poem_input = st.selectbox(
    "Select Poem Title",
    [
        "The Road Not Taken – Robert Frost",
        "Still I Rise – Maya Angelou",
        "If – Rudyard Kipling",
        "Daffodils – William Wordsworth",
        "Annabel Lee – Edgar Allan Poe",
        "Wish I Could Tell You - Durjoy Datta"
    ]
)
  
style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Literary Analysis", "Storytelling"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt('template.json')


if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'poem_input': poem_input,
        'style_input': style_input,
        'length_input':length_input
    })
    st.write(result.content)
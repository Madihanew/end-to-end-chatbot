# import all necessary libraries
import streamlit as st
import nltk
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")

#Defining intents
#Each intent consists of a tag, patterns (phrases representing the intent), and responses (appropriate replies for the intent).

intents = [
    {
        "tag": "greeting",
        "patterns": ["How are you?", "Are you ok?", "Hey", "How are you", "What's going on?"],
        "responses": ["Hi there", "Hi", "Hey", "I'm fine, thank you", "Nothing"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am NeuraPulse AI chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "AI",
        "patterns": ["what is AI?", "Define AI?", "What is the definition of AI?"],
        "responses": ["Artificial Intelligence (AI) refers to the simulation of human intelligence processes by machines, particularly computer systems", "AI encompasses a broad range of techniques and approaches, from simple algorithms to complex neural networks, aimed at enabling machines to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. "]
    },
    {
        "tag": "sub-fields",
        "patterns": ["what are different sub-fields of AI?", "Name some forms of AI?", "where can we use AI?"],
        "responses": ["Surely, these are some sub-fields of AI: Machine Learning, Deep learning, Natural Language Processing, Computer Vision, and robotics ", "Sure, AI has different categories including ML,DL, NLP, and CV"]
    },
    {
        "tag": "Applications",
        "patterns": ["what are some applications of AI?", "Name some applications of AI?", "where can we use AI?"],
        "responses": ["Machine Learning, Deep learning, Natural Language Processing, Computer Vision, and robotics ", "Sure AI can be used in health care, finance, transportation, education, and entertainment"]
    },
    {
        "tag": "NeuraPulse",
        "patterns": ["What is neurapulse AI?", "what is the purpose of Neurapulse AI?", "what are the expertise of NeuraPulse AI"],
        "responses": ["NeuraPulse AI is a software development company", "NeuraPulse create AI-powered software solutions", "NeuraPulse AI have a team of experts in machine learning, natural language processing, computer vision, and data analytics"]
    },
    {
        "tag": "vacancy",
        "patterns": ["Is there any job opportunuties in Neurapulse AI", "any job vacancy in Neurapulse AI?", "any vacancy in Neurapulse AI"],
        "responses": ["Please visit website and social media apps for career opportunities in Neurapulse AI", "Please visit our website for latest update"]
    },
   
    {"tag": "greetings",
      "patterns": ["Hello there", "Hey, How are you", "Hey", "Hi", "Hello", "Anybody", "Hey there"],
      "responses": ["Hello, I'm your helping bot", "Hey it's good to see you", "Hi there, how can I help you?"],
      "context": [""]
    },
    {"tag": "thanks",
      "patterns": ["Thanks for your quick response", "Thank you for providing the valuable information", "Awesome, thanks for helping"],
      "responses": ["Happy to help you", "Thanks for reaching out to me", "It's My pleasure to help you"],
      "context": [""]
    },
    {"tag": "no_answer",
      "patterns": [],
      "responses": ["Sorry, Could you repeat again", "provide me more info", "can't understand you"],
      "context": [""]
    },
    {"tag": "support",
      "patterns": ["What help you can do?", "What are the helps you provide?", "How you could help me", "What support is offered by you"],
      "responses": [ "I can guide you about AI", "I can assist you in understanding basic concepts of AI"],
      "context": [""]
    },
    {"tag": "goodbye",
        "patterns": ["bye bye", "Nice to chat with you", "Bye", "See you later buddy", "Goodbye"],
        "responses": [ "bye bye, thanks for reaching", "Have a nice day there", "See you later"],
        "context": [""]
    }
]

vectorizer = TfidfVectorizer()
classifier = LogisticRegression(random_state=0, max_iter=10000)
tags=[]
patterns=[]
for i in intents:
    for pattern in i['patterns']:
        tags.append(i["tag"])                        #extracting tags from intents
        patterns.append(pattern)                     #extracting patterns from intents
x=vectorizer.fit_transform(patterns)                 #declaring x as an independent variable
y=tags                                               #declaring y as dependent variable
classifier.fit(x,y)                                  #using x and y to train logistic regression model

def chatbot_response(text):                          #defining a function to get chatbot response bases on user_input
    input_text=vectorizer.transform([text])          #vectoring the input into numerical form
    tagsdata=classifier.predict(input_text)[0]       #using trained classifier to predict the tag
    for i in intents:
        if i["tag"] == tagsdata:
            response=random.choice(i['responses'])   #randomly choosing a response from responses in intents
            return response 
        

st.title('This is chatbot for NeuraPulse AI')

if "messages" not in st.session_state:              # Initialize chat history
    st.session_state.messages = []

for message in st.session_state.messages:           # Display chat messages from history on app rerun
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Welcome to Neurapulse"): # React to user input
    with st.chat_message("user"):                    # Display user message in chat message container
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})# Add user message to chat history
    response = f"Echo: {prompt}"
   
    with st.chat_message("assistant"):               # Display assistant response in chat message container
        response=chatbot_response(prompt)
        st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})# Add assistant response to chat history


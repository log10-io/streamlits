import streamlit as st

st.title('Hello Streamlit!')

name = st.text_input('What is your name?', 'Type your name here...')
if name:
    st.write(f'Hello, {name}!')
else:
    st.write('Please enter your name above to see the greeting.')
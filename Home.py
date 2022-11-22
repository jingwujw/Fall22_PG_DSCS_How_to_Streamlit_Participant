import streamlit as st

col1, col2 = st.columns(2)
col2.image('./home/profile.png')

col1.header('Marc-Robin Grüner')
col1.markdown('''
Hello everyone,

My name is Marc and I love working on data science projects.
On this webiste, I share some of my recent journeys.\n
You can find them on the left!

In case you find any of these interesting, feel free to reach out:

[LinkedIn](https://www.linkedin.com/in/marc-robin-gruener/)    [Email](mailto:marc-robin.gruener@student.unisg.ch)
''')
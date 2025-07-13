import streamlit as st

pages = [
    st.Page("main.py", title="사업소 소개", icon="🟠", default=True),
    st.Page("data.py", title="사업소별 데이터", icon="🟢"),
    st.Page('machinelearning.py', title = 'NOX 배출량 예측', icon="🔵"),
    # st.Page("hyper.py", title="머신러닝", icon="🤖")
]

selected_page = st.navigation(pages)

selected_page.run()
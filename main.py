import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
import matplotlib.font_manager as fm

st.set_page_config(layout="wide")

st.title('메인페이지')

def load_data():
    df = pd.read_csv("기상청_alldata.csv")
    return df

df = load_data()

color_map = {
    '분당': '#a6cee3',     # 부드러운 하늘색
    '삼천포': '#b2df8a',   # 파스텔 연두색
    '영동': '#fb9a99',     # 연한 핑크
    '영흥': '#fdbf6f',     # 따뜻한 오렌지
    '여수': '#cab2d6'      # 연한 보라색
}

df_bundang = df[df['사업소'] == '분당'].reset_index(drop = True)
df_sancheonpo = df[df['사업소'] == '삼천포'].reset_index(drop = True)
df_yeosu = df[df['사업소'] == '여수'].reset_index(drop = True)
df_yeongdong = df[df['사업소'] == '영동'].reset_index(drop = True)
df_yeongheung = df[df['사업소'] == '영흥'].reset_index(drop = True)

font_path = 'NanumGothic.ttf'
fm.fontManager.addfont(font_path)
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

col1, col2 = st.columns([2,3])

if 'selected_location' not in st.session_state:
    st.session_state.selected_location = '분당발전본부'

selected_location = st.session_state.selected_location  # 이 줄 추가

with col1:
    with st.container(border=True):
        st.subheader('◼ 사업소')
        col1_1, col1_2, col1_3 = st.columns(3)

        with col1_1:
            file_path = f"{selected_location}.png"  # 세션 상태에서 가져온 선택값 사용
            image = Image.open(file_path)
            st.image(image, width = 130)

        with col1_2:
            selected = st.session_state.selected_location = st.radio(
                "발전 본부 선택",
                ['분당발전본부', '삼천포발전본부', '여수발전본부', '영동에코발전본부', '영흥발전본부'],
                index=['분당발전본부', '삼천포발전본부', '여수발전본부', '영동에코발전본부', '영흥발전본부'].index(st.session_state.selected_location)
            )

            if selected != selected_location:
                st.session_state.selected_location = selected
                st.rerun()
        
        with col1_3 :
            if selected_location == '영동에코발전본부' :
                st.markdown('**⚙ 영동에코발전본부**\n\n☎ 전화번호 : 070-8898-4000\n\n 🔍 주소 : 강원특별자치도 강릉시 강동면 염전길 99\n\n 🏭 발전소 : 2기')
            
            elif selected_location == '영흥발전본부' :
                st.markdown('**⚙ 영흥발전본부**\n\n☎ 전화번호 : 070-8898-3000\n\n 🔍 주소 : 인천광역시 옹진군 영흥면 영흥남로 293번길 75\n\n 🏭 발전소 : 6기')

            elif selected_location == '삼천포발전본부' :
                st.markdown('**⚙ 삼천포발전본부**\n\n☎ 전화번호 : 070-8898-2000\n\n 🔍 주소 : 경남 고성군 하이면 하이로 1\n\n 🏭 발전소 : 8기')

            elif selected_location == '분당발전본부' :
                st.markdown('**⚙ 분당발전본부**\n\n☎ 전화번호 : 070-8898-6000\n\n 🔍 주소 : 경기도 성남시 분당구 분당로 336\n\n 🏭 발전소 : 8기')

            else :
                st.markdown('**⚙ 여수발전본부**\n\n☎ 전화번호 : 070-8898-5000\n\n 🔍 주소 : 전남여수시 여수산단로 727\n\n 🏭 발전소 : 2기')

    with st.container(border=True) :
        st.subheader('◼ 대기오염물질 누적 비율')
        grouped = df.groupby('사업소')[['SOX', 'NOX', '먼지', '산소']].sum()

        grouped_T = grouped.transpose()
        grouped_T_percent = grouped_T.div(grouped_T.sum(axis=1), axis=0) * 100
        grouped_T_percent = grouped_T_percent.reset_index().melt(id_vars='index', var_name='사업소', value_name='비율')
        grouped_T_percent.rename(columns={'index': '오염물질'}, inplace=True)

        fig = px.bar(
            grouped_T_percent,
            x="오염물질",
            y="비율",
            color="사업소",
            color_discrete_map=color_map
        )
        fig.update_layout(
                xaxis_title=None,
                yaxis_title=None,
                yaxis=dict(range=[0, 100])
                )
        st.plotly_chart(fig)

with col2 :
    with st.container(border=True) :
        st.subheader('◼ 대기오염물질 발생 사업소 순위')
        st.write('1순위 : 삼천포ㅤㅤㅤ2순위 : 영흥ㅤㅤㅤ3순위 : 분당ㅤㅤㅤ4순위 : 영동ㅤㅤㅤ5순위 : 여수')
    with st.container(border=True) :
        feature = st.radio(
            '',
            ['NOX', 'SOX', '먼지', '산소'],
            horizontal=True,
            label_visibility = 'collapsed'
    )
        
    col2_1, col2_2 = st.columns(2, border=True)
    
    with col2_1 :
        st.subheader('◼ 대기오염물질 평균(분당)')
        fig1, ax = plt.subplots(figsize = (5, 2))
        sns.barplot(df_bundang.groupby('호기')[feature].mean(), label = feature, color = color_map['분당'])
        plt.grid(axis = 'y', color = 'grey', alpha = 0.5, linestyle = '--')
        plt.legend()
        plt.xlabel('')
        plt.ylabel('')
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        st.pyplot(fig1)

    with col2_2 :
        st.subheader('◼ 대기오염물질 평균(삼천포)')
        fig2, ax = plt.subplots(figsize = (5, 2))
        sns.barplot(df_sancheonpo.groupby('호기')[feature].mean(), label = feature, color = color_map['삼천포'])
        plt.grid(axis = 'y', color = 'grey', alpha = 0.5, linestyle = '--')
        plt.legend()
        plt.xlabel('')
        plt.ylabel('')
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        st.pyplot(fig2)
        
    col2_3, col2_4 = st.columns(2, border=True)

    with col2_3 :
        st.subheader('◼ 대기오염물질 평균(여수/영동)')    
        fig3, axs = plt.subplots(1, 2, figsize = (5, 2))
        axs[0].bar(df_yeongdong['호기'].unique(), df_yeosu.groupby('호기')[feature].mean(), label = feature, color = color_map['여수'])
        axs[0].legend()
        axs[0].grid(axis = 'y', color = 'grey', alpha = 0.5, linestyle = '--')
        for s in ['top', 'bottom', 'left', 'right']:
            axs[0].spines[s].set_visible(False)

        axs[1].bar(df_yeongdong['호기'].unique(), df_yeongdong.groupby('호기')[feature].mean(), label = feature, color = color_map['영동'])
        axs[1].grid(axis = 'y', color = 'grey', alpha = 0.5, linestyle = '--')
        axs[1].legend()
        for s in ['top', 'bottom', 'left', 'right']:
            axs[1].spines[s].set_visible(False)
        st.pyplot(fig3)

    with col2_4 :
        st.subheader('◼ 대기오염물질 평균(영흥))')
        fig5, ax = plt.subplots(figsize = (5, 2))
        sns.barplot(df_yeongheung.groupby('호기')[feature].mean(), label = feature, color = color_map['영흥'])
        plt.grid(axis = 'y', color = 'grey', alpha = 0.5, linestyle = '--')
        plt.legend()
        plt.xlabel('')
        plt.ylabel('')
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        st.pyplot(fig5)
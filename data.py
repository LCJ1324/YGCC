import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

st.set_page_config(layout="wide")

st.title('사업소별 데이터')

def load_data():
    df = pd.read_csv("기상청_alldata.csv")
    return df

df = load_data()

def parse_date_safe(x):
    try:
        return pd.to_datetime(str(int(x)), format='%Y%m%d')
    except:
        return pd.to_datetime(x, errors='coerce')

df['일자'] = df['일자'].apply(parse_date_safe)
df['연도'] = df['일자'].dt.year

font_path = 'NanumGothic.ttf'
fm.fontManager.addfont(font_path)
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

with st.sidebar :
    side_col1, side_col2 = st.columns(2)

    with side_col1 :
        location = st.selectbox('사업소', ['분당', '삼천포', '영동', '영흥', '여수'])
        selected_df = df[df['사업소'] == location]

    with side_col2 :
        unit = st.selectbox('', selected_df['호기'].unique())

    feature = st.selectbox(
        '특성', 
        ['NOX', 'SOX', '먼지', '산소',
        '유량', '온도', '용량(MW)', '열효율(%)',
        '이용률(%)', '평균기온(°C)']
        )

new_df = selected_df[selected_df['호기'] == unit]

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

col1 = st.container(border = True)

with col1 :
    st.subheader('◼ 사업소별 데이터 평균값')
    col1_3, col1_4, col1_5, col1_6, col1_7 = st.columns(5, border=True)
    
    col1_3.metric(
        '**⚙ 분당발전본부**',
        "{:.2f}".format(df_bundang[feature].mean())
    )
    
    col1_4.metric(
        '**⚙ 삼천포발전본부**',
        "{:.2f}".format(df_sancheonpo[feature].mean())
    )

    col1_5.metric(
        '**⚙ 여수발전본부**',
        "{:.2f}".format(df_yeosu[feature].mean())
    )

    col1_6.metric(
        '**⚙ 영동에코발전본부**',
        "{:.2f}".format(df_yeongdong[feature].mean())
    )

    col1_7.metric(
        '**⚙ 영흥발전본부**',
        "{:.2f}".format(df_yeongheung[feature].mean())
    )

    df_filtered = df.copy()
    df_filtered["사업소_호기"] = df_filtered["사업소"] + "(" + df_filtered["호기"] + ")"
    grouped_feature = df_filtered.groupby(["사업소", "사업소_호기"])[feature].mean().reset_index()
    grouped_feature = grouped_feature.sort_values(by=feature, ascending=False)

    colors = [color_map.get(biz, "#cccccc") for biz in grouped_feature["사업소"]]

    fig, ax = plt.subplots(figsize=(16, 2))
    sns.barplot(
        data=grouped_feature,
        x="사업소_호기",
        y=feature,
        palette=colors,
        ax=ax
    )
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(axis='y', color='grey', alpha=0.5, linestyle='--')
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    st.pyplot(fig)

col2, col3 = st.columns([2, 4], border = True)

with col2 :
    st.subheader('◼ 구간별 데이터 비율')
    if (location == '분당' and (feature == 'SOX' or feature == '먼지')) :
        st.warning('⚠ 분당사업소는 SOX와 먼지 값이 존재하지 않습니다.')

    else :
        # IQR 계산
        Q1 = selected_df[feature].quantile(0.25)
        Q3 = selected_df[feature].quantile(0.75)
        IQR = Q3 - Q1

        # IQR 기반 정상 데이터
        filtered = selected_df[(selected_df[feature] >= Q1 - 1.5 * IQR) & (selected_df[feature] <= Q3 + 1.5 * IQR)]

        # 구간 설정용 min/max
        min_val = filtered[feature].min()
        iqr_max = filtered[feature].max()
        true_max = selected_df[feature].max()  # 전체 최댓값

        # 구간 생성 (정상 범위 기준 4등분)
        bins = pd.interval_range(start=min_val, end=iqr_max, periods=4)
        bins = list(bins)

        # 마지막 구간 확장: 이상치 포함
        last_bin = pd.Interval(left=bins[-1].left, right=true_max, closed='right')
        bins[-1] = last_bin

        # 최종 구간 리스트 및 라벨
        cut_bins = [b.left for b in bins] + [bins[-1].right]
        labels = [f"{b.left:.2f} ~ {b.right:.2f}" for b in bins]

        # 구간 적용
        selected_df['range'] = pd.cut(selected_df[feature], bins=cut_bins, labels=labels, include_lowest=True)

        # 비율 계산
        pie_data = selected_df['range'].value_counts(normalize=True).sort_index() * 100

        # 파이차트 그리기
        fig1, ax = plt.subplots(figsize=(3, 3))
        colors = ["#28A0FF", '#00BFFF', '#87CEEB', '#AFEEEE']
        ax.pie(
            pie_data.values,
            labels=pie_data.index,  # 범위만 표시
            autopct='%1.1f%%',      # 퍼센트 내부 출력
            colors=colors[:len(pie_data)],
            textprops={'fontsize': 6},
        )

        ax.set_title(f"{location}사업소 - {feature}", fontsize=8)
        st.pyplot(fig1)

with col3 :
    col3_1, col3_2 = st.columns([3,1])
    with col3_1 :
        st.subheader('◼ NOX 기준치 초과 비율')
        fig2, ax = plt.subplots(figsize = (6, 2))
        plt.plot(new_df['일자'], new_df['NOX'], color = color_map[location])
        plt.axhline(new_df['NOX기준'].unique()[0], color = 'r', linestyle = '--', alpha = 0.7)
        plt.grid(axis = 'y', color = 'grey', alpha = 0.5, linestyle = '--')
        plt.title(f'{location} - {unit}')
        plt.xticks(rotation=45, ha='right', fontsize = 8)
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        st.pyplot(fig2)

    if (location == '여수') and (unit == '-') :
        with col3_2 :
            st.warning('⚠기준치가 존재하지 않습니다.')
                
    else :
        with col3_2 :
            with st.container(border=True) :
                st.metric('**⚙ 기준치 값**' ,
                        int(new_df['NOX기준'].unique()[0]))
                
            with st.container(border=True) :
                if len(new_df['NOX기준초과'].unique()) == 1 :
                    st.metric('**⚙ 기준치 초과 일수**' ,
                        "0일")
                else :
                    st.metric('**⚙ 기준치 초과 일수**' ,
                        f"{new_df['NOX기준초과'].value_counts()[1]}일")
                
            with st.container(border=True) :
                if len(new_df['NOX기준초과'].unique()) == 1 :
                    st.metric('**⚙ 기준치 초과 비율**' ,
                        "0%")
                else :
                    st.metric('**⚙ 기준치 초과 비율**' ,
                        "{:.2f}%".format(new_df['NOX기준초과'].value_counts()[1] / len(new_df) * 100))
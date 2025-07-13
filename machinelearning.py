import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.font_manager as fm

st.set_page_config(layout="wide")

st.title('NOX 배출량 예측')

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

df['NOX_per_MWh'] = df['NOX'] / (df['발전량(MWh)'] + 1e-5)
df['이용률_MW'] = df['발전량(MWh)'] / (df['용량(MW)'] + 1e-5)
df['유량당_발전량'] = df['발전량(MWh)'] / df['유량']
df['단위용량당_발전량'] = df['발전량(MWh)'] / df['용량(MW)']
df['이용률_MW'] = df['발전량(MWh)'] / (df['용량(MW)'] + 1e-5)

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

score = pd.DataFrame({
    '영흥(1호기)' : 0.434,
    '영흥(2호기)' : 0.312,
    '영흥(3호기)' : 0.639,
    '영흥(4호기)' : 0.775,
    '영흥(5호기)' : 0.932,
    '영흥(6호기)' : 0.694,
    '영동(1호기)' : 0.644,
    '영동(2호기)' : 0.884,
    '분당(1호기)' : 0.721,
    '분당(2호기)' : 0.739,
    '분당(3호기)' : 0.915,
    '분당(4호기)' : 0.729,
    '분당(5호기)' : 0.937,
    '분당(6호기)' : 0.801,
    '분당(7호기)' : 0.829,
    '분당(8호기)' : 0.892,
    '삼천포(3A호기)' : 0.871,
    '삼천포(3B호기)' : 0.821,
    '삼천포(4A호기)' : 0.612,
    '삼천포(4B호기)' : 0.581,
    '삼천포(5A호기)' : 0.327,
    '삼천포(5B호기)' : 0.416,
    '삼천포(6A호기)' : 0.518,
    '삼천포(6B호기)' : 0.337,
    '여수(-)' : 0.984,
    '여수(1호기)' : 0.730,
}, 
index = ['score']).T

score['사업소'] = score.index.str.split('(').str[0]
score['color'] = score['사업소'].map(color_map)
score = score.sort_values(by = 'score', ascending = False)

with st.container(border = True) :
    st.subheader('◼ NOX 예측 모델 정확성')
    fig, ax = plt.subplots(figsize=(16, 2))
    sns.barplot(
        data=score,
        x=score.index,
        y='score',
        palette=score['color'].to_dict(),
        ax=ax
    )
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(axis='y', color='grey', alpha=0.5, linestyle='--')
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    st.pyplot(fig)

col1, col2 = st.columns(2, border=True)

new_df.replace([np.inf, -np.inf], np.nan, inplace=True)

if location == '분당' :
    features = ['산소', '유량', '온도', '용량(MW)',
            '발전량(MWh)', '열효율(%)', '이용률(%)', '평균기온(°C)', '일강수량(mm)',
            '평균 풍속(m/s)', '평균 이슬점온도(°C)', '평균 상대습도(%)', '평균 증기압(hPa)',
            '이용률_MW', '유량당_발전량', '단위용량당_발전량']
    
elif location == '삼천포' :
    features = ['산소', '유량', '온도', 'SOX', '먼지']

else :
    features = ['산소', '유량', '온도', '용량(MW)', 'SOX', '먼지',
            '발전량(MWh)', '열효율(%)', '이용률(%)', '평균기온(°C)', '일강수량(mm)',
            '평균 풍속(m/s)', '평균 이슬점온도(°C)', '평균 상대습도(%)', '평균 증기압(hPa)' ,
            '이용률_MW', '유량당_발전량', '단위용량당_발전량']

param = pd.DataFrame([
    ['영흥', '1호기', {'subsample': 0.6, 'reg_lambda': 1.0, 'reg_alpha': 0.1, 'num_leaves': 15, 'n_estimators': 500, 'min_child_samples': 10, 'max_depth': None, 'learning_rate': 0.02, 'colsample_bytree': 1.0}],
    ['영흥', '2호기', {'subsample': 0.8, 'reg_lambda': 0, 'reg_alpha': 0.5, 'num_leaves': 50, 'n_estimators': 1000, 'min_child_samples': 10, 'max_depth': 7, 'learning_rate': 0.02, 'colsample_bytree': 1.0}],
    ['영흥', '3호기', {'random_state' : 42}],
    ['영흥', '4호기', {'random_state' : 42}],
    ['영흥', '5호기', {'subsample': 1.0, 'reg_lambda': 0, 'reg_alpha': 0.1, 'num_leaves': 70, 'n_estimators': 1000, 'min_child_samples': 10, 'max_depth': -1, 'learning_rate': 0.005, 'colsample_bytree': 0.8}],
    ['영흥', '6호기', {'bootstrap': False, 'max_depth': None, 'max_features': 0.8, 'min_samples_leaf': 6, 'min_samples_split': 6, 'n_estimators': 283}],
    ['영동', '1호기', {'random_state' : 42}],
    ['영동', '2호기', {'bootstrap': False, 'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 352}],
    ['분당', '1호기', {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 360}],
    ['분당', '2호기', {'subsample': 0.6, 'reg_lambda': 0.1, 'reg_alpha': 0, 'num_leaves': 15, 'n_estimators': 500, 'min_child_samples': 10, 'max_depth': 10, 'learning_rate': 0.01, 'colsample_bytree': 1.0}],
    ['분당', '3호기', {'bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 519}],
    ['분당', '4호기', {'bootstrap': False, 'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 352}],
    ['분당', '5호기', {'bootstrap': True, 'max_depth': 30, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 446}],
    ['분당', '6호기', {'subsample': 0.6, 'reg_lambda': 1.0, 'reg_alpha': 0, 'num_leaves': 120, 'n_estimators': 300, 'min_child_samples': 10, 'max_depth': 7, 'learning_rate': 0.07, 'colsample_bytree': 0.7}],
    ['분당', '7호기', {'subsample': 0.9, 'reg_lambda': 0.05, 'reg_alpha': 0.01, 'num_leaves': 70, 'n_estimators': 700, 'min_child_samples': 5, 'max_depth': -1, 'learning_rate': 0.01, 'colsample_bytree': 0.8}],
    ['분당', '8호기', {'subsample': 0.7, 'reg_lambda': 0.5, 'reg_alpha': 0.01, 'num_leaves': 90, 'n_estimators': 700, 'min_child_samples': 5, 'max_depth': 15, 'learning_rate': 0.03, 'colsample_bytree': 0.7}],
    ['삼천포', '3A호기', {'random_state' : 42}],
    ['삼천포', '3B호기', {'subsample': 0.8, 'reg_lambda': 1.0, 'reg_alpha': 0, 'num_leaves': 120, 'n_estimators': 1000, 'min_child_samples': 5, 'max_depth': 5, 'learning_rate': 0.01, 'colsample_bytree': 0.8}],
    ['삼천포', '4A호기', {'objective' : 'regression', 'n_estimators' : 500, 'learning_rate' : 0.01, 'max_depth' : 8, 'subsample' : 0.8, 'colsample_bytree' : 0.8, 'reg_alpha' : 1.0, 'reg_lambda' : 1.0, 'random_state' : 42}],
    ['삼천포', '4B호기', {'objective' : 'regression', 'n_estimators' : 400, 'learning_rate' : 0.03, 'max_depth' : 8, 'subsample' : 0.9, 'colsample_bytree' : 0.8, 'reg_alpha' : 1.0, 'reg_lambda' : 1.0, 'random_state' : 42}],
    ['삼천포', '5A호기', {'bootstrap': True, 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 349}],
    ['삼천포', '5B호기', {'random_state' : 42}],
    ['삼천포', '6A호기', {'objective' : 'regression', 'n_estimators' : 800, 'learning_rate' : 0.01, 'max_depth': 10, 'subsample' : 0.9, 'colsample_bytree' : 0.9, 'reg_alpha' : 1.0, 'reg_lambda' : 1.0, 'random_state' : 42, 'n_jobs' : -1}],
    ['삼천포', '6B호기', {'colsample_bytree': np.float64(0.8092261699076477), 'gamma': np.float64(0.3588304841235025), 'learning_rate': np.float64(0.13658008112196623), 'max_depth': 11, 'min_child_weight': 2, 'n_estimators': 595, 'reg_alpha': np.float64(0.45194860129903924), 'reg_lambda': np.float64(1.927564909710652), 'subsample': np.float64(0.9598669781168693)}],
    ['여수', '-', {'subsample': 0.6, 'reg_lambda': 0.1, 'reg_alpha': 0, 'num_leaves': 50, 'n_estimators': 300, 'min_child_samples': 5, 'max_depth': 10, 'learning_rate': 0.07, 'colsample_bytree': 1.0}],
    ['여수', '1호기', {'random_state' : 42}]
], 
columns = ['사업소', '호기', 'best_params'],
index = range(26))

target = 'log_NOX'

new_df = new_df[new_df['NOX'] <= 30]
new_df['log_NOX'] = np.log1p(new_df['NOX'])
new_df = new_df.dropna(subset=features + [target])

X = new_df[features]
y = new_df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

row = param[(param['사업소'] == location) & (param['호기'] == unit)]
best_params = row.iloc[0]['best_params']

if ((location == '분당') and (unit in ['1호기', '3호기', '4호기', '5호기'])) or ((location == '삼천포') and (unit in ['3A호기', '5A호기', '5B호기'])) or (location == '영동') or ((location == '영흥') and (unit == '6호기')) :
    model = RandomForestRegressor(**best_params)

elif ((location == '분당') and (unit in ['2호기', '6호기', '7호기', '8호기'])) or ((location == '삼천포') and (unit in ['3B호기', '4A호기', '4B호기', '6A호기'])) or ((location == '여수') and (unit == '-')) or ((location == '영흥') and (unit in ['1호기', '2호기', '5호기'])) :
    model = LGBMRegressor(**best_params)

else :
    model = XGBRegressor(**best_params)

model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_true = np.expm1(y_test)

r2 = r2_score(y_test_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))

with col1 :
    st.subheader('◼ 특성 중요도')
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance')
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title(f'{location} {unit} - Feature Importance')
    plt.ylabel('')
    for s in ['top', 'right']:
        plt.gca().spines[s].set_visible(False)
    st.pyplot(plt)

with col2 :
    st.subheader('◼ NOX 예측')
    plt.figure(figsize=(8, 4))
    plt.scatter(y_test_true, y_pred, alpha=0.7, color=color_map[location])
    plt.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], '--r')
    plt.xlabel('Actual NOX')
    plt.ylabel('Predicted NOX')
    plt.title(f'{location} {unit} - Actual vs Predicted (R²={r2:.3f})')
    for s in ['top', 'right']:
        plt.gca().spines[s].set_visible(False)
    st.pyplot(plt)
    
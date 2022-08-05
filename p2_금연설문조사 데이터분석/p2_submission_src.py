
# base tool
import pandas as pd
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows',100)
import numpy as np
from sklearn.model_selection import train_test_split
import copy

import warnings
warnings.filterwarnings('ignore')

#visualization
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
pio.templates.default = "plotly_white"

# test
from imblearn.under_sampling import RandomUnderSampler
import pingouin as pg
from scipy.stats import chi2_contingency,shapiro

from statsmodels.stats.outliers_influence import variance_inflation_factor

# resampling
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

# modeling
from pandas.api.types import CategoricalDtype
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.ensemble import RandomForestClassifier

# evaluation
from sklearn.metrics import accuracy_score,confusion_matrix, plot_confusion_matrix,f1_score, classification_report




def summary(df, pred=None):
    obs = df.shape[0]
    Types = df.dtypes
    Counts = df.apply(lambda x: x.count())
    Min = df.min()
    Max = df.max()
    Uniques = df.apply(lambda x: x.unique().shape[0])
    Nulls = df.apply(lambda x: x.isnull().sum())
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['Types', 'Counts', 'Uniques', 'Nulls', 'Min', 'Max']
        st = pd.concat([Types, Counts, Uniques, Nulls, Min, Max], axis = 1, sort=True)

    st.columns = cols
    print('___________________________\nData Types:')
    print(st.Types.value_counts())
    print('___________________________')
    return st


def vif(df:pd.DataFrame)->None:
    print('vif score')
    df = df.iloc[:,:-1]
    vif_scores = pd.DataFrame() 

    vif_scores["Attribute"] = df.columns 
    # calculating VIF for each feature 
    vif_scores["VIF Scores"] = [round(variance_inflation_factor(df.values, i) ,2)for i in range(len(df.columns))] 
    display(vif_scores)
    print('-'*50)



def pairwise(df:pd.DataFrame,disp:bool =False)->None:
    print('kendall correlation')
    print()
    df = df.iloc[:,:]
    corr = df.corr(method='kendall').round(3)
    display(corr.iloc[:-1,[-1]])
    # (1,2) (2,4) (4,1)? (6,7)
    # 1,2,4 번은 모두 상담사의 관한 질문
    if disp:
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.columns,
            colorscale=px.colors.diverging.RdBu,
            zmin=-1,
            zmax=1
        ))
        fig.show()
    print('-'*50)



def cronbach (df:pd.DataFrame)->None:
    print('cronbach-alpha test:',pg.cronbach_alpha(data=df,ci=0.95))
    print('-'*50)



def chi2(df:pd.DataFrame)->None:
    print('chi-square test')
    chi_df = df.iloc[:,:-1]
    y_ = df.iloc[:,-1]
    for i in range(len(chi_df.columns)):
        cross_tb = pd.crosstab(chi_df.iloc[:,i],y_)
        chi, p, dof, expected = chi2_contingency(cross_tb)
        print('문항',str(i+1),':',end=' ')
        print(f"chi 스퀘어 값: {round(chi,2)}",
            f"p-value (0.05): {p}",
            # f"자유도 수: {dof}",
            # f"기대값: \n{pd.DataFrame(expected)}",
            # f"측정값: \n{cross_tb}", sep = "\n" 
            )
    print('-'*50)




def plot_cm(y_test,y_pred):
    print('test confusion matrix:')
    z = confusion_matrix(y_true=y_test,y_pred= y_pred)



    x = ['0','1','2','3','4']
    y = ['0','1','2','3','4']

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()


def modeling(df:pd.DataFrame)->None:
    ## ordinal regression
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=0,stratify=y)
    shapiro_y = copy.deepcopy(y_train)
    print('count of train y:',sorted(Counter(y_train).items()))
    cat_type = CategoricalDtype(categories=[0,1,2,3,4], ordered=True)
    y_train = y_train.astype(cat_type)

    # for distr in ['probit','logit']:
    for distr in ['logit']:
        print('Ordinal regression',distr)


        mod = OrderedModel(y_train,
                            X_train,
                            distr=distr)
        res = mod.fit(method='bfgs')
        display(res.summary())

        # train data
        predicted = res.model.predict(res.params, exog=X_train)
        y_pred = pd.Series([i.argmax() for i in predicted]).ravel()
        if distr =='probit':
            residual  = shapiro_y -y_pred
            print(shapiro(residual))
            fig = px.histogram(residual)
            fig.show()

        # test data
        predicted = res.model.predict(res.params, exog=X_test)
        y_pred = pd.Series([i.argmax() for i in predicted]).ravel()
        

        print('-'*50)

        plot_cm(y_test=y_test,y_pred=y_pred)
        print(classification_report(y_test, y_pred))
        print('-'*50)
        print()

    
        
    
    ## random forest
    print('random forest:')
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)

    display(pd.DataFrame({'index':X.columns,'feature importance':clf.feature_importances_}).round(2))
    y_pred  = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print()
    plot_cm(y_test=y_test,y_pred=y_pred)

    print('-'*50)


    



def smoteEnn(df:pd.DataFrame):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]

    smote_enn = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    print(sorted(Counter(y).items()))
    print(sorted(Counter(y_resampled).items()))
    resampled_df = X_resampled.merge(y_resampled,left_index=True,right_index=True)
    return resampled_df


def pipeline(df:pd.DataFrame) ->None:
    cronbach(df)
    vif(df)
    # chi2(df)
    pairwise(df)
    modeling(df)


rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)


sorted(Counter(rawData.iloc[:,-1]).items())

 
# # Introduction

 
# # 1. Data Overview

 
# ## 1.1 data


# https://www.data.go.kr/data/15092388/fileData.do
# 한국건강증진개발원_국가금연지원서비스 등록정보(만족도)
raw_data = pd.read_csv('./만족도평가(2020).csv',encoding='cp949')
rawData = pd.read_csv('./만족도평가(2020).csv',encoding='cp949')



display(raw_data.head())
print(raw_data.shape)


raw_data = raw_data.drop(['문항6'],axis=1)


summary(raw_data)

 
# # 2.EDA


x = list(raw_data.columns)
x = x[7:]
fig = make_subplots(rows=2, cols=3,subplot_titles=x,vertical_spacing=0.1,x_title='설문조사 결과 히스토그램')

trace0 = go.Histogram( x=raw_data[x[0]],)
trace1 = go.Histogram( x=raw_data[x[1]],)
trace2 = go.Histogram( x=raw_data[x[2]],)
trace3 = go.Histogram( x=raw_data[x[3]],)
trace4 = go.Histogram( x=raw_data[x[4]],)
trace5 = go.Histogram( x=raw_data[x[5]],)


fig.append_trace(trace0,1,1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 3)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 2, 3)
fig.layout.update(height=700)
fig.update_layout(bargap=0.2)
fig.show()

 
# # Single Response


# survey data slicing
survey_data= raw_data.iloc[:,7:]
# count unique value in row
val_num = survey_data.stack().groupby(level=0).apply(lambda x: len(x.unique().tolist()))
survey_data['unique_val'] = val_num
# replace num of unique val to specific unique value
survey_data['unique_val']=survey_data['unique_val'].map(lambda x: 'multiple_val' if x !=1 else x)
survey_data['unique_val'] = survey_data.apply(lambda x: 'only '+str(x['문항1']) if x['unique_val'] ==1 else x,axis=1)['unique_val']
print(survey_data['unique_val'].value_counts())


px.pie(survey_data,names='unique_val')

 
#  H0: 항목1~5 has same value, then  항목7 has same value


# survey data slicing
survey_data= raw_data.iloc[:,7:-1]
# count unique value in row
val_num = survey_data.stack().groupby(level=0).apply(lambda x: len(x.unique().tolist()))
survey_data['unique_val'] = val_num
temp = survey_data.merge(raw_data['문항7'],left_index=True,right_index=True)[survey_data['unique_val']==1]
pred_val = temp['문항1'].ravel()
true_val =temp['문항7'].ravel()
accuracy_score(true_val,pred_val)



#drop
raw_data = raw_data[survey_data['unique_val']!=1]


# dummy 성별, 출생년도
raw_data['성별'].replace({'남':1,'여':0},inplace=True)
year_col = sorted(raw_data['출생년도'].unique(),reverse=True)
raw_data['출생년도'].replace(year_col,range(len(year_col)),inplace=True)
raw_data = raw_data.loc[:,['출생년도','성별','문항1','문항2','문항3','문항4','문항5','문항7']]



#문항 5외 다른 점수 점수 반대로 설정
for col in ['문항1','문항2','문항3','문항4','문항7']:
    raw_data[col] = raw_data[col].replace([0,1,2,3,4],[4,3,2,1,0])


raw_data.to_csv("preprocessed_data.csv", index=False, encoding="utf-8-sig")

 
# # 모델링


rawData = pd.read_csv('preprocessed_data.csv',encoding='utf-8')


rawData.shape


# full model without re-sampling
pipeline(rawData)


# 출생년도, 성별 제거 model without re-sampling

rawData = rawData.iloc[:,2:]
pipeline(rawData)


 
# ## TokEE


rawData = pd.read_csv('preprocessed_data.csv',encoding='utf-8')
rawData = rawData.iloc[:,2:]



X=rawData.iloc[:,:-1]
y=rawData.iloc[:,-1]

smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
print(sorted(Counter(y).items()))
print(sorted(Counter(y_resampled).items()))
r_resampled = X_resampled.merge(y_resampled,left_index=True,right_index=True)



temp =r_resampled.iloc[:,:]
x = list(temp.columns)
fig = make_subplots(rows=1, cols=5,subplot_titles=x,vertical_spacing=0.1,x_title='설문조사 결과 히스토그램')

trace0 = go.Histogram( x=temp[x[0]],)
trace1 = go.Histogram( x=temp[x[1]],)
trace2 = go.Histogram( x=temp[x[2]],)
trace3 = go.Histogram( x=temp[x[3]],)
trace4 = go.Histogram( x=temp[x[4]],)


fig.append_trace(trace0,1,1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1,3)
fig.append_trace(trace3, 1, 4)
fig.append_trace(trace4, 1, 5)
fig.layout.update(height=400)
fig.update_layout(bargap=0.2)
fig.show()


# full model with re-sampling
pipeline(r_resampled)



# 출생년도, 성별, 항목4 제거 model with re-sampling
pipeline(r_resampled.iloc[:,[0,1,2,4,5]])




# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:12:07 2020

@author: lx
"""
#################################   Pandas数据预处理    ############################################
#########################第一讲 数据框###########################################
##设置路径读取数据#######################
import os  ##查看当前路径
os.getcwd()
os.chdir('C:\\Users\\lx\\Desktop')##更改路径至桌面
import pandas as pd
data=pd.read_csv('data.csv')      ##利用pandas读取数据
data
data1=pd.read_csv('data2.csv',sep=',',encoding='unicode_escape')
data=pd.read_table('data.txt',sep=',',encoding='gbk') 
data.shape
data.head(10) 
data.to_csv('data_feiyan.csv',encoding='gbk')    ##输出数据
##行列选择##############################
data[0:1]      ##根据位置
data.loc[1]    ##根据行名
data.iloc[1]   ##根据行号
data['a1']     ##选区某一列
data.iloc[:,0:1]
data[['a1','a2']]
##行列筛选################################
data[data['a1']=='a']
data[data['a1'].isin(['a','b'])]
##创建新列#################################
data.a5=data.a2*data.a3
##创建数据框###############################
frame=pd.DataFrame({'id':['a','b','c'],'price':[1,2,3]},index=[1,2,3])
frame.shape#行列数
type(frame)#数据类型
##数据框重命(列)名##############################
frame.columns=['hehe','haha']
##数据框合并#######################################
##输出csv文件###############################
frame.to_csv("newdata.csv",sep=';',index=False)#"False"表示行名不导入
##读取excel文件###########################
pd.read_excel('newdata.excel')

#############################第二讲 pandas数据的查改增删#########################
##查找数据##
frame['price']
frame['id']
frame[['price','id']][0:2]
frame.loc[1:2,'id':'price']
##修改添加数据##
frame['haha']=0 #添加空列
frame.columns#查看信息
frame.loc[:,'haha']=frame['id']
frame
##删除数据##
frame['gaogao']='wuwuwu'
frame.drop(labels='gaogao',axis=1,inplace=True)#删除指定列
frame.drop(labels=1,axis=0,inplace=True)#删除指定行
#排序#
frame.sort_values(by='price',ascending=False)
#修改索引#
frame.set_index('id',inplace=False)

#############################第三讲 pandas\numpy描述性统计分析，分组聚合操作#######
import pandas as pd
import numpy as np
data=pd.DataFrame({'id':['a','b','c'],'price':[1,2,3]},index=[1,2,3])
data
##################描述性统计####################
##numpy方法##
np.min(data['price'])
data['haha']=0
np.min(data[['price','haha']])
np.mean(data[['price','haha']])
##pandas方法##
data[['price','haha']].mean()
##描述性统计函数##
data[['price','haha']].describe()
##类别型数据的描述性统计##
data['id'].value_counts()
##################处理与时间相关的数据##################
data1=pd.read_csv('data2.csv',sep=',',encoding='unicode_escape') 
data1.dtypes
data1['EFFECTDATE']=pd.to_datetime(data1['EFFECTDATE']) ##转换为时间序列数据
data1['EFFECTDATE'].dtypes
year1=[i.year for i in data1['EFFECTDATE']]##提取时间数据中的信息
year1
week=[i.week for i in data1['EFFECTDATE']]
week
data1.loc[:,'EFFECTDATE'].head()+pd.Timedelta(days=1)##加减时间数据
data1.head(5)
data1.tail(5)
##################分组聚合操作##################
##分组##
data1Group=data1[['No','TOACCOUNT','TRANAMT']].groupby(by='No')
type(data1Group)
data1Group ##不能直接查看，因为没有聚合
##聚合##
data1Group.sum() ##直接对所有数据求和
data1Group.agg(np.sum) ##用agg方法求和
data1Group.agg({'TOACCOUNT':np.max,'TRANAMT':np.mean})
#####################类型转换##########################
data1['No'].dtypes
data1['No']=data1['No'].astype(float)
data1['No'].dtypes

###################################第四讲 欧洲杯案例分析######################################
#导入相关数据及库#
import pandas as pd
import numpy as np
euro12=pd.read_csv('Euro2012.csv')
euro12.shape
euro12.columns
euro12.head(3)
#只选取Goals这一列#
euro12['Goals']
#有多少球队参加了2012世界杯#
euro12['Team'].unique().shape[0]
#一共有多少列#
euro12.shape[1]
#将列Team,红牌，黄牌单独存为一个名为discipline的数据框#
discipline=euro12[['Team','Red Cards','Yellow Cards']]
discipline.shape
#排序#
discipline.sort_values(by=['Red Cards','Yellow Cards'],ascending=False)
#计算每个球队拿到黄牌数的平均值#
discipline['Yellow Cards'].mean()
euro12.groupby(by='Team').agg({'Goals':np.sum}) #另一种理解，分组聚合典型操作
#进球数超过6个的球队#
index=euro12['Goals']>6
euro12.loc[index,:]
#选取以字母G开头的球队数据#
index1=euro12['Team'].str[0]=='G'
euro12.loc[index1,:]
#选取前7列#
euro12.iloc[:,0:7]
#选取除了最后三列之外的全部列#
euro12.iloc[:,0:-3]
#找到英格兰，意大利，俄罗斯的射正率#
index2=euro12['Team'].agg(lambda x:x in ['England','Italy','Russia'])
euro12.loc[index2,'Shooting Accuracy']

####################################第五讲 数据清洗##########################################
###########数据清洗(出现重复值导致方差变小，分布改变；空值导致样本量减少，分析结果偏差；异常值导致伪回归)################
#检测和处理重复值#
import pandas as pd
import numpy as np
d1={'a':[1,2,3,3,4,5,8,9,10],'b':[1,2,3,3,4,5,6,7,8]}
df1=pd.DataFrame(d1)
df1
#numpy法
df1['a'].unique()
#pandas法
df1.drop_duplicates(keep='first') #对数据框操作
#python法
set(df1['a'])
#特征重复
d2={'a':[1,2,3,4],'b':[1,2,3,4],'c':[4,2,3,1]}
df2=pd.DataFrame(d2)
df2[['a','b']].corr()
df2.drop(labels='b',axis=1)
#检测与处理缺失值#
d3={'姓名':['张三','李四','王五','刘六',np.nan,np.nan],'成绩':[20,30,50,95,np.nan,np.nan],'性别':['男','女',np.nan,'男',np.nan,np.nan],'备注':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]}
df4=pd.DataFrame(d3)
df4['姓名'].isnull()          #检测
df4['性别'].notnull().sum()
df4.dropna(axis=1,how='all') #处理
df4.dropna(axis=0,how='all')
df4.dropna(axis=1,how='any') 
df4.dropna(axis=0,how='any')
df4['姓名'].fillna('未知')
df4['成绩'].fillna(df4['成绩'].mean())
df4['性别'].replace(np.nan,'女')
#检测与处理异常值\离群点#
a=[5,6,7,8,3,5,123,0.00001]
s1=pd.Series(a)
s1.describe()
#1.找四分位数
l_value=s1.quantile(0.25)
u_value=s1.quantile(0.75)
c_value=u_value-l_value
#2.替换
s1.loc[s1>u_value+1.5*c_value]=u_value
s1.loc[s1<l_value-1.5*c_value]=l_value
s1

####################################第五讲 案例分析##########################################
import pandas as pd
import numpy as np
import datetime 
#导入风速数据,并合并前三列#
wind=pd.read_csv('wind.csv',sep='\s+',parse_dates=[[0,1,2]])
wind.shape
#将2061改为2016#
def fix_year(x):
    if x.year>2000:
        mid=x.year-100
    else:
        mid=x.year
    new_date=pd.to_datetime(datetime.date(mid,x.month,x.day))
    return new_date
wind['Yr_Mo_Dy']=wind['Yr_Mo_Dy'].agg(fix_year)
#将日期作为索引#
wind=wind.set_index('Yr_Mo_Dy')
wind
#查看缺失值和完整值的个数#
wind.isnull().sum()
wind.notnull().sum()
#计算风速的平均值#
wind.mean()
wind.mean(axis=1)
#计算最小值、最大值、平均值、标准差#
loc_stats=wind.agg([np.min,np.max,np.mean,np.std])
day_stats=wind.T.agg([np.min,np.max,np.mean,np.std])
#对于每个location,计算一月份的平均风速#
index=wind.index.month==1
wind_new=wind[index]
wind_new.groupby(by=[wind_new.index.year,wind_new.index.month]).mean()
#按年月日取样本#
wind.asfreq('Y')
wind.asfreq('2Y')
wind.asfreq('3m')
wind.asfreq('3d')
wind.resample('2d',closed='right',label='right').mean() #每两天求均值

####################################第六讲 pandas数据合并##########################################
import pandas as pd
import numpy as np
import warnings        #忽视报警信息
warnings.filterwarnings('ignore')
##堆叠合并##
d1={'A':['A1','A2','A3'],'B':['B1','B2','B3'],'C':['C1','C2','C3']}
df1=pd.DataFrame(d1)
df1
d2={'A':['A4','A5','A6'],'B':['B4','B5','B6'],'C':['C4','C5','C6'],'D':['D4','D5','D6']}
df2=pd.DataFrame(d2)
df2
pd.concat([df1,df2],ignore_index=True) #忽略原始索引且合并
pd.concat([df1,df2],ignore_index=True,join='inner') #是否存在缺少列
pd.concat([df1,df2],ignore_index=True,join='outer')
pd.concat([df1,df2],ignore_index=True,join='outer',axis=1) #列堆叠
s1=pd.Series(['e1','e2','e3'],name='E')
s1
pd.concat([df1,s1],axis=1)
##追加合并##
df3=pd.DataFrame(columns=['A','B'])#生成一个空Dataframe
df3.append({'A':1,'B':2},ignore_index=True)
l1=['张三','李四','王五']
l2=['绘画','摄影','舞蹈']
#法一
for i in range(3):                 
    df3=df3.append({'A':l1[i],'B':l2[i]},ignore_index=True)
df3
#法二
df4=pd.concat([pd.DataFrame([[l1[i],l2[i]]],columns=['A','B']) for i in range(3)],ignore_index=True)
df4
##主键合并##
stu_d={'xh':[1,2,3,4],'xm':['a','b','c','d'],'kch':['k1','k2','k2','k3']}
sco_d={'kch':['k1','k2','k3','k4'],'kcm':['语文','数学','英语','政治'],'jsh':['j1','j1','j2','j3']}
tea_d={'jsh':['j1','j2','j3','j4'],'jsm':['赵四','刘能','谢广坤','王大拿']}
df_stu=pd.DataFrame(stu_d)
df_sco=pd.DataFrame(sco_d)
df_tea=pd.DataFrame(tea_d)
#学生和课程通过主键kch合并#
df_mer1=pd.merge(df_stu,df_sco,left_on='kch',right_on='kch')
#课程和教师通过jsh合并#
df_mer2=pd.merge(df_sco,df_tea,left_on='jsh',right_on='jsh')
df_mer1.loc[df_mer1['kcm']=='数学',:]
##重叠合并(结构相同内容互补的表进行合并)##
dict1={'id':[1,2,3,4],'system':['win10',np.nan,'win7',np.nan],'cup':['i7',np.nan,'i5',np.nan]}
dict2={'id':[1,2,3,4],'system':[np.nan,'win7',np.nan,'win10'],'cup':[np.nan,'i5',np.nan,'i7']}
df_A=pd.DataFrame(dict1)
df_B=pd.DataFrame(dict2)
df_A.combine_first(df_B)


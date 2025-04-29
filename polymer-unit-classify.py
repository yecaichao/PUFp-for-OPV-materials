#!/usr/bin/env python
# coding: utf-8

# In[1]:


#前面一步生成polymer unit还需要去除自由基


# In[1]:


import csv,os
import numpy as np
import pandas as pd
import itertools
import random


# In[2]:


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole 
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions 
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs


# In[3]:


#导入数据
file=open('ring_total_list.csv')
fileReader=csv.reader(file)
filedata=list(fileReader)
arr=np.array(filedata)[1:]


# In[4]:


#建立一个存储polymer_unit性质的DataFrame
ring_df=pd.DataFrame(arr[:,1],columns=['smiles'])
ring_df.index.name='index'


# In[5]:


#按单环，双环，支链，稠环进行划分
num_list=['1','2','3','4','5','6','7','8','9','%']
ring_type=[]
for i in arr:
    i=i.tolist()
    smi=i[1]
    num=[]
    for j in smi:
        if j in num_list:
            num.append(j)
    if len(num)==0:
        ring_type.append('bratch')
    elif len(num)==2:
        ring_type.append('single')
    elif len(num)==4:
        ring_type.append('double')
    elif len(num)>4:
        ring_type.append('fused')
    else:
        ring_type.append('unknow')

#给ring_df加上一列‘ring_type’
ring_df['ring_type']=ring_type


# In[6]:


#是否含硫
sulfur=[]
for i in arr:
    i=i.tolist()
    smi=i[1]
    if 's'in smi:
        s_num=smi.count('s')
        if 'se'in smi:
            se_num=smi.count('se')
            if s_num>se_num:
                sulfur.append('YES')
            elif s_num==se_num:
                sulfur.append('NO')
        else:
            sulfur.append('YES')
    elif 'S'in smi:
        S_num=smi.count('S')
        if 'Si'in smi:
            Si_num=smi.count('i')
            if S_num>Si_num:
                sulfur.append('YES')
            elif S_num==Si_num:
                sulfur.append('NO')
        else:
            sulfur.append('YES')
    else:
        sulfur.append('NO')
ring_df['sulfur']=sulfur


# In[7]:


#是否含硒
selenium=[]
for i in arr:
    i=i.tolist()
    smi=i[1]
    if 'se'in smi:
        selenium.append('YES')
    else:
        selenium.append('NO')
ring_df['selenium']=selenium


# In[8]:


#是否含硅
silicon=[]
for i in arr:
    i=i.tolist()
    smi=i[1]
    if 'Si'in smi:
        silicon.append('YES')
    else:
        silicon.append('NO')
ring_df['silicon']=silicon


# In[9]:


#是否含氧
oxygen=[]
for i in arr:
    i=i.tolist()
    smi=i[1]
    if 'o'in smi:
        oxygen.append('YES')
    elif 'O'in smi:
        oxygen.append('YES')
    else:
        oxygen.append('NO')
ring_df['oxygen']=oxygen


# In[10]:


#是否含氮
nitrogen=[]
for i in arr:
    i=i.tolist()
    smi=i[1]
    if 'n'in smi:
        nitrogen.append('YES')
    elif 'N'in smi:
        nitrogen.append('YES')
    else:
        nitrogen.append('NO')
ring_df['nitrogen']=nitrogen


# In[11]:


#是否含氯
chlorine=[]
for i in arr:
    i=i.tolist()
    smi=i[1]
    if 'Cl'in smi:
        chlorine.append('YES')
    else:
        chlorine.append('NO')
ring_df['chlorine']=chlorine


# In[12]:


#是否含氟
fluorine=[]
for i in arr:
    i=i.tolist()
    smi=i[1]
    if 'F'in smi:
        fluorine.append('YES')
    else:
        fluorine.append('NO')
ring_df['fluorine']=fluorine


# In[13]:


#根据属性进行总体分类
ring_df['polymer_type']=''

#判断polymer_unit的类型
def type_judge(x):
    ring_type=x['ring_type']
    sulfur=x['sulfur']
    oxygen=x['oxygen']
    nitrogen=x['nitrogen']
    chlorine=x['chlorine']
    fluorine=x['fluorine']
    selenium=x['selenium']
    silicon=x['silicon']
    SMILES=x['smiles']
    smiles=SMILES.lower()
    #判断有哪些元素
    element=[]
    if sulfur=='YES':
        element.append('S')
    if nitrogen=='YES':
        element.append('N')
    if fluorine=='YES':
        element.append('F')
    if oxygen=='YES':
        element.append('O')
    if selenium=='YES':
        element.append('se')
    if silicon=='YES':
        element.append('Si')
    if chlorine=='YES':
        element.append('Cl')
    long=len(element)
    type_name=ring_type
    i=0
    while i < long:
        type_name=type_name+'-'+element[i]
        i=i+1

    return(type_name)
    
    
f = lambda x: type_judge(x)

ring_df['polymer_type']=ring_df.apply(f,axis='columns')

ring_df.to_csv('ring_df.csv')


# In[14]:


#给数据的polymer_unit分类
file=open('index_data.csv')
fileReader=csv.reader(file)
filedata=list(fileReader)
arr=np.array(filedata)[1:]
index_frame=pd.DataFrame(arr[:,1:],index=arr[:,0])
long=arr.shape[1]-1
type_frame=pd.DataFrame(index=arr[:,0],columns=np.arange(long))
data_long=arr.shape[0]
i=0
while i<data_long:
    j=0
    while j<long:
        if index_frame[j][i]!='none':
            num=int(index_frame[j][i])
            type=ring_df['polymer_type'][num]
            type_frame[j][i]=type
        j=j+1
    i=i+1
type_frame.to_csv('type_frame.csv')


# In[ ]:





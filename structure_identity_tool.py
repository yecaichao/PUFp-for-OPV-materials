#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import csv, os
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Chem.Draw import IPythonConsole 
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions 
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit import Chem
from rdkit.Chem import Draw


# In[ ]:


def get_bracket_index(s):
    l_list=[0]
    r_list=[] 
    i_list=[0]
    i=0
    while i < len(s):
        j=s[i]
        if j =="(":
            i_list.append(i)
            l_list.append(i)
        if j==")":
            i_list.append(i)
            r_list.append(i)
        i=i+1
    l_list=list(reversed(l_list))
    r_list.append(len(s)-1)
    i_list.append(len(s)-1)
    return(l_list,r_list,i_list)


# In[ ]:


def smallest(cp_arr,index_arr):
    smallest_list=[]
    for i in cp_arr:
        l=i[0]
        r=i[1]
        arr1=index_arr[index_arr>l]
        arr2=arr1[arr1<r]
        if len(arr2)==0:
            smallest_list.append(i)
    return(smallest_list)

def link_c(string):
    num_list=['1','2','3','4','5','6','7','8','9','%']
    if len(string)>1:
        if string[0]=="C" and string[1] in num_list:
            string="C"+string
    return(string)



def found_independent_ring_in_same_str2(string_list):
    total_independent_string=[]
    for string in string_list:
        independent_string=[]
        if "/" in string:
            string=string.replace('/','')
        if "\\" in string:
            string=string.replace('\\','')
  
        index_num={}
        num=['0','1','2','3','4','5','6','7','8','9']
        num_list=[]
        index_list=[]
        cp_list=[]
     
        
        i=0
    
        while i < len(string):    
            j=string[i]
            if j in num:
                index_list.append(i)
                i=i+1
            elif j == "%":
                index_list.append(i)
                i=i+3
            else:
                i=i+1
        

        if string[1] not in num:
            index_list.append(0)
            index_num[0]=[]
        i=0
        while i < len(string):
            j=string[i]    
            if j in num:
                if j not in num_list:
                    num_list.append(j)
                elif j in num_list:
                    num_list.remove(j)
                index_num[i]=num_list[:]
                cp_like=[]
                for k,v in index_num.items():
                    if v==num_list and k!=i:
                        if k!=index_list[-1]:
                            cp_index=index_list.index(k)+1
                            cp_num=index_list[cp_index]
                            cp_like.append(cp_num)
                if len(cp_like)>0:
                    cp_like.sort()
                    cp=[cp_like[-1],i]
                    cp_list.append(cp)
                i=i+1        
            elif j =="%":
                number=string[i:i]
                if number not in num_list:
                    num_list.append(number)
                elif number in num_list:
                    num_list.remove(number)
                index_num[i]=num_list[:]
                cp_like=[]
                for k,v in index_num.items():
                    if v==num_list and k!=i:
                        if k!=index_list[-1]:
                            cp_index=index_list.index(k)+1
                            cp_num=index_list[cp_index]
                            cp_like.append(cp_num)
                if len(cp_like)>0:
                    cp_like.sort()
                    cp=[cp_like[-1],i]
                    cp_list.append(cp)
                i=i+3        
            else:
                i=i+1  
               
        
        new_cp_list=[]
        cp_arr=np.array(cp_list)
        for i in cp_arr:
            cp_arr2=cp_arr[cp_arr[:,0]<i[0]]
            cp_arr3=cp_arr2[cp_arr2[:,1]>i[1]]
            if len(cp_arr3)==0:
                i=i.tolist()
                new_cp_list.append(i)
        cp_list=new_cp_list
    
        be_string={}
        for i in cp_list:   
            s=""
            name=str(i[0])+"-"+str(i[1])
            if i==cp_list[0] and index_num[i[1]]==[]:
                if i[1]<len(string)-2:
                    if string[i[1]+1]=="=" and string[i[1]+2]=="O":
                        s=s+string[i[0]-1:i[1]+3]
                        
                else:
                    s=s+string[i[0]-1:i[1]+1]
                be_string[name]=[s,[0,i[1]]]
            else:
                if string[i[1]]!="%":
                    if i[1]<len(string)-2:
                        if string[i[1]+1]=="=" and string[i[1]+2]=="O":
                            s=s+string[i[0]-1:i[1]+3]
                    else:
                        s=s+string[i[0]-1:i[1]+1]
                be_string[name]=[s,i]
                if string[i[1]]=="%":
                    s=s+string[i[0]-1:i[1]+3]
                    be_string[name]=[s,i]
        
        for k,v in be_string.items():
            v_list=[v[0]]
            v[0]=add_bracket(v_list)[0]
            v[0]="C"+v[0]
    
        real_string={}
        for k,v in be_string.items():
            mol = Chem.MolFromSmiles(v[0])
            if mol:
                real_string[k]=v
    
        first=0
        last=len(string)
        outside_num=[]
        for k,v in real_string.items():
            outside_num.append(v[1][0])
            outside_num.append(v[1][1])

        outside_num.sort()
        if len(outside_num)>0:
            if outside_num[0]!=0:
                outside_str=string[0:outside_num[0]]+"(C)"
                i=1
                while i<len(outside_num)-1:
                    j=outside_num[i]
                    k=outside_num[i+1]
                    if string[j]!="%":
                        outside_str=outside_str+"(C)"+string[j:k+1]
                    elif string[j]=="%":
                        outside_str=outside_str+"(C)"+string[j+3:k+1]
                    i=i+2

            if outside_num[0]==0:
                outside_str=""
                i=1
                while i<len(outside_num)-1:
                    j=outside_num[i]
                    k=outside_num[i+1]
                    if string[j]!="%":
                        outside_str=outside_str+"(C)"+string[j:k+1]
                    elif string[j]=="%":
                        outside_str=outside_str+"(C)"+string[j+3:k+1]
                    i=i+2
        
            if outside_num[-1]!=last:
                n=outside_num[-1]
                m=string[n]
                if m!="%":
                    outside_str=outside_str+"(C)"+string[n:last]
                if m=="%":
                    outside_str=outside_str+"(C)"+string[n+3:last]
            for k,v in real_string.items():
                independent_string.append(v[0])
    
        if len(independent_string)==0:
            total_independent_string.append(string)
        else:
            mol=Chem.MolFromSmiles(outside_str)
            if mol:
                for i in independent_string:
                    total_independent_string.append(outside_str)
            for i in independent_string:
                        total_independent_string.append(i)
    
    return(total_independent_string)


# In[ ]:


def make_smi(smi):
    
    if len(smi)>1:
        if smi[0]=="(" and smi[-1]==")":
            smi=smi[1:-1]
    if "/" in smi:
        smi=smi.replace('/','')
    if "\\" in smi:
        smi=smi.replace('\\','')
    if "-" in smi:
        smi=smi.replace('-','')
    if smi[0]=="=":
        smi=smi[1:]
    return(smi)


# In[ ]:


def add_bracket(string):
    right_bracket=0
    left_bracket=0
    if string[0]==")":
        string=string[1:]
    if string[0]=="=":
        string="C"+string
    for j in string:
        if j =="(":
            right_bracket=right_bracket+1
        if j==")":
            left_bracket=left_bracket+1
    if right_bracket>left_bracket:
        n=right_bracket-left_bracket
        while n>0:
            string=string+")"
            n=n-1
    if right_bracket<left_bracket:
        m=left_bracket-right_bracket
        while m>0:
            string="("+string
            m=m-1
    if len(string)>2:
        if string[0]=="(" and string[1]==")":
            string=string[2:]
        if string[0]==")":
            string=string[1:]
    
    return(string)
        


# In[ ]:


def if_mol(smi_list):
    
    smi_list=[]
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            wrong_list.append(smi)
    return(smi_list)


# In[ ]:


def found_independent_ring_in_same_str1(independent_ring_string0):
    real_independent_ring=[]
    
    num=['1','2','3','4','5','6','7','8','9']
    for i in independent_ring_string0:
      
        j=0
        num_list=[]
        ring_list=[]
        cp_list=[]
        index1=0
        while j<len(i):
            k=i[j]
            if k in num:
                num_list.append(k)
                j=j+1
                if len(num_list)==2*len(set(num_list)):
                    num_list=[]
                   
                    if j <= (len(i)-2):
                        if i[j+1]=="=" and i[j+2]=="O":
                            cp_list.append([index1,j+3])
                        else:
                            cp_list.append([index1,j])
                    else:
                        cp_list.append([index1,j])
                    index1=j
            elif k == "%":
                number=str(i[j+1])+str(i[j+2])
                num_list.append(number)
                j=j+3
                if len(num_list)==2*len(set(num_list)):
                    num_list=[]
                    if j <= (len(i)-2):
                        if i[j+1]=="=" and i[j+2]=="O":
                            cp_list.append([index1,j+3])
                        else:
                            cp_list.append([index1,j])
                    else:
                        cp_list.append([index1,j])
                    index1=j
            else:
                j=j+1
        if cp_list !=[]:
            cp_list[-1][1]=len(i)
            for k in cp_list:
                f=k[0]
                l=k[1]
                s=i[f:l]
                ring_list.append(s)
        if len(ring_list)==0:
            real_independent_ring.append(i)
        if len(ring_list)!=0:
           
            for g in ring_list:
               
                real_independent_ring.append(g)
    return(real_independent_ring)


# In[ ]:


def found_location_in_DataFrame_double(DataFrame,key): 
    
    hang=DataFrame.shape[0]-1
    location=[]
    while hang >=0:
        lie=DataFrame.shape[1]-1
        while lie >=0:
            cp=DataFrame[lie][hang]
            if len(cp)>1:
                if cp==key:
                    location.append([hang,lie])
            lie=lie-1
        hang=hang-1
    return(location)


# In[ ]:


def structure_DataFrame(c_list,smallest_list,r_list,l_list):
    str_DataFrame=pd.DataFrame(index=np.arange(len(smallest_list)),columns=np.arange(len(c_list)))
    str_DataFrame[:]='n'
    
    i=0 
    while i < len(smallest_list):
        j=smallest_list[i]
        str_DataFrame[0][i]=j
        i=i+1
    
    
    list_2=[]
    list_1=smallest_list
    k=0
    
    while k<(len(c_list)-1):
        for u in list_1:
            if len(u)== 2:
                if u[1] !=c_list[-1][1]:
                    plus_u=int(c_list.index(u))+1
                    u_plus=c_list[plus_u]
                    while (u_plus[1]<u[1])and(u_plus[1]!=c_list[-1][1]):
                        plus_plus_u=c_list.index(u_plus)+1
                        u_plus=c_list[plus_plus_u]
                    if u_plus[1]<u[1]:
                        list_2.append("n")
                    if u_plus[1]>u[1]:
                        list_2.append(u_plus)
                if u[1] ==c_list[-1][1]:
                    list_2.append("n")
            elif len(u)==1:
                list_2.append("n")
        j=0
        for n in list_2:
            str_DataFrame[k+1][j]=n
            j=j+1
        k=k+1
        list_1=list_2
        list_2=[]
    return(str_DataFrame)


# In[ ]:


def get_cp_data(cp_list0,smallest_r0,str_df0,independent_cp0,bratch_cp0):
    cp_data0={}
    cp_name_list=[]
    independent_cp1=independent_cp0+bratch_cp0
    for cp in cp_list0:
        if cp not in smallest_r0:
            cp_name=str(cp[0])+"-"+str(cp[1])
            cp_name_list.append(cp_name)
            cp_data0[cp_name]=[cp]
         
    hang=len(smallest_r0)-1
    while hang >=0:
        lie=len(cp_list0)-1
        while lie >0:
            cp=str_df0[lie][hang]
            if len(cp) == 2:
                cp_name=str(cp[0])+"-"+str(cp[1])
                cp=str_df0[lie][hang]
                cp_min=str_df0[lie-1][hang]
                j=lie-1
                while (cp_min not in independent_cp1) and (j!=0):
                    j=j-1
                    cp_min=str_df0[j][hang]
                if cp_min in independent_cp1:
                    for k,v in cp_data0.items():
                        if k==cp_name:
                            if cp_min not in v:
                                v.append(cp_min)
            lie=lie-1
        hang=hang-1
    return(cp_data0)


# In[ ]:


def pairing(smiles,index_list,l_list,r_list):
 
    i=0
    cp_ed=[]
    c_list=[]
    while i < len(l_list):
        l=l_list[i]
        j=0
        try:
            while (r_list[j] < l) or (r_list[j] in cp_ed):
                j=j+1
        except IndexError:
            print(smiles)
            j=j-1
        r=r_list[j]
        cp_ed.append(r)
        cp=[l,r]
        c_list.append(cp)
        i=i+1
    
    first=0
    last=len(smiles)
    if first not in index_list:
        if last not in index_list:
            c_list.append([first,last])
    return(c_list)


# In[ ]:


def make_con(index_data,index_cp,br):
    index_data2={}
    error=[]
    error_k=[]
    for k,v in index_data.items():
        mol=Chem.MolFromSmiles(v[1])
        if mol:
            smi = Chem.MolToSmiles(mol)
            index_data2[k]=[v[0],smi]
        if not mol:
            error.append(v[0])
            error_k.append(k)
    for i in error:
        if i in index_cp:
            index_cp.remove(i)
    for i in error_k:
        if i in br:
            del br[i]
    return(index_data2,index_cp,br)
   


# In[ ]:


def found_end_point_neighbour(smiles,neighbor_data,index_data):
  
    left_end=[]
    right_end=[]
    l_or_r=[]
    for k,v in neighbor_data.items():
        if "br" not in k:
            m=index_data[k]
            if '[C]'in m[1]:
                if v['right_neighbor']=={}:
                    left_end.append(m)
                
                if v['left_neighbor']=={}:
                    right_end.append(m)
                
                else:
                    l_or_r.append(m)
                
    left_name="left_name"
    left_smiles="left_smiles"
    right_name="right_name"
    right_smiles="right_smiles"
    
    if len(left_end)>=1 and len(right_end)>=1:
        left=left_end[0][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=left_end[0][1]
        right=right_end[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=right_end[0][1]
        
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data

       
    elif len(left_end)==1 and len(l_or_r)==1:
        left=left_end[0][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=left_end[0][1]
        right=l_or_r[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=l_or_r[0][1]
       
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data
        
  
    elif len(right_end)==1 and len(l_or_r)==1:
        right=right_end[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=right_end[0][1]
        left=l_or_r[0][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=l_or_r[0][1]
       
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data
       
  
    elif len(l_or_r)==2:
        right=l_or_r[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=l_or_r[0][1]
        left=l_or_r[1][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=l_or_r[1][1]
        
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data
    
    
    elif len(l_or_r)==1:
        right=l_or_r[0][0]
        right_name=str(right[0])+"-"+str(right[1])
        right_smiles=l_or_r[0][1]
        left=l_or_r[0][0]
        left_name=str(left[0])+"-"+str(left[1])
        left_smiles=l_or_r[0][1]
        
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["right_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data
   
    elif len(right_end)==2:
        num1=right_end[0][0][1]
        num2=right_end[1][0][1]
        if num1>num2:
            right=right_end[0][0]
            right_name=str(right[0])+"-"+str(right[1])
            right_smiles=right_end[0][1]
            left=right_end[1][0]
            left_name=str(left[0])+"-"+str(left[1])
            left_smiles=right_end[1][1]
        if num1<num2:
            right=right_end[1][0]
            right_name=str(right[0])+"-"+str(right[1])
            right_smiles=right_end[1][1]
            left=right_end[0][0]
            left_name=str(left[0])+"-"+str(left[1])
            left_smiles=right_end[0][1]
        
        right_data={}
        left_data={}
        for k,v in neighbor_data.items():
            if k ==left_name:
                right_data[right_name]=right_smiles
                neighbor_data[k]["left_neighbor"]=right_data
            if k ==right_name:
                left_data[left_name]=left_smiles
                neighbor_data[k]["left_neighbor"]=left_data

    
    return(neighbor_data)


# In[ ]:


def bratch_amend(br):
    br2={}
    for k,v in br.items():
        v2=[]
        for i in v:
            if '(C)'in i:
                i=i.replace('(C)','')
            if 'c'in i:
                i=i.replace('c','C')
            v2.append(i)
        br2[k]=v2
    return(br2)


# In[ ]:


def found_neighbor(br,str_df,index_data,index_cp):
    neighbor_data={}
    for i in index_cp:
        location=found_location_in_DataFrame_double(str_df,i)
        right_neighbour=[]
        left_neighbour=[]
        for j in location:            
            hang=j[0]
            if j[1]<(str_df.shape[1]-1):
                right_lie=j[1]+1
                right=str_df[right_lie][hang]
                m=str_df.shape[1]-1
                while (right not in index_cp) and (right_lie!=m):
                    right_lie=right_lie+1
                    right=str_df[right_lie][hang]
                if right in index_cp:
                    right_name=str(right[0])+"-"+str(right[1])
                    right_neighbour.append(right_name)
            if j[1]>0:                                   
                left_lie=j[1]-1
                left=str_df[left_lie][hang]
                while (left not in index_cp) and (left_lie>0):
                    left_lie=left_lie-1
                    left=str_df[left_lie][hang]
                if left in index_cp:
                    left_name=str(left[0])+"-"+str(left[1])
                    left_neighbour.append(left_name)
        
        right_list=[]
        for r in right_neighbour:
            if r not in right_list:
                right_list.append(r)
      
        left_list=[]
        for l in left_neighbour:
            if l not in left_list:
                left_list.append(l)
        
        name=str(i[0])+"-"+str(i[1])
        neighbor={}
        right_neighbor={}
        left_neighbor={}
        self=index_data[name][1]

        for item in right_list:
            right_neighbor[item]=index_data[item][1]
        for item in left_list:
            left_neighbor[item]=index_data[item][1]
        if right_list==[] and left_list==[]:
            right_neighbor[name]=index_data[name][1]
            left_neighbor[name]=index_data[name][1]
        
        
        if len(br[name])!=0:
            count=len(br[name])
            i=0
            while i<count:
                br_name=name+"-br-"+str(i)
                left_neighbor[br_name]=br[name][i]
                i=i+1
        
        neighbor["right_neighbor"]=right_neighbor
        neighbor["left_neighbor"]=left_neighbor
        neighbor["self"]=self
        neighbor_data[name]=neighbor
        
        neighbor_data2=neighbor_data.copy()
   
    for k,v in br.items():
        if len(v)!=0:
            count=len(v)
            nei={}
            nei[k]=index_data[k][1]
            i=0
            while i<count:
                br_name=k+"-br-"+str(i)
                data={}
                data['self']=v[i]
                data['right_neighbor']=nei
                data['left_neighbor']={}
                neighbor_data[br_name]=data
                i=i+1
    return(neighbor_data)            


# In[ ]:


def found_location_in_DataFrame_single(DataFrame,key): 
  
    hang=DataFrame.shape[0]-1
    location=[]
    while hang >=0:
        lie=DataFrame.shape[1]-1
        while lie >0:
            cp=DataFrame[lie][hang]
            if cp==key:
                location.append([hang,lie])
            lie=lie-1
        hang=hang-1
    return(location)


# In[ ]:


def rigin_type_classify(cp_list,smiles,smallest_r,str_DataFrame):
    independent_cp0=[]
    dependent_cp0=[]
    bratch0={}
    bratch_cp0=[]
    i=0
    num=['1','2','3','4','5','6','7','8','9','10']
    while i<(len(cp_list)-1):
        j=0
        while j<len(smallest_r):
            h=str_DataFrame[i][j]
            h_out=str_DataFrame[i+1][j]
            out_judge=True
            if len(h_out)==2:
                string_out=smiles[h_out[0]:(h_out[1]+1)]
                
            j=j+1
            if len(h)==2:
                num_list=[]
                string=smiles[h[0]:(h[1]+1)]
                k=0
                while k < len(string):
                    e=string[k]
                    if e=="%":
                        u=str(string[k+1])+str(string[k+2])
                        num_list.append(u)
                        k=k+3
                    if e in num:
                        num_list.append(e)
                        k=k+1
                    else:
                        k=k+1
                diff=len(num_list)-len(set(num_list))
                if len(num_list)==0:
                    if len(string)>6:
                        if out_judge==True:
                            name=str(h[0])+"-"+str(h[1])
                            bratch0_data=[string,h]
                            bratch0[name]=bratch0_data
                            bratch_cp0.append(h)
                    if len(string)<=5:
                        dependent_cp0.append(h)
                elif len(num_list)==1:
                    dependent_cp0.append(h)
                elif len(num_list)>1:
                    judge=False
                    for n in num_list:
                        r=num_list.count(n)%2
                        if r==1:
                            judge=True
                    if judge==False:
                        independent_cp0.append(h)
                    if judge==True:
                        dependent_cp0.append(h)
                
        i=i+1
    
    bratch_arr=np.array(bratch_cp0)
    dependent_bratch=[]
    for i in bratch_arr:
        
        arr1=bratch_arr[bratch_arr[:,0]<i[0]]
        arr2=arr1[arr1[:,1]>i[1]]
        if len(arr2) !=0:
            dependent_bratch.append(i)
       
        
            
    dependent_bratch2=[]
    for i in dependent_bratch:
        j=i.tolist()
        if j not in dependent_bratch2:
            dependent_bratch2.append(j)
            
    for i in dependent_bratch2:
        name=str(i[0])+"-"+str(i[1])
        del bratch0[name]
    
    bratch=[]
    bratch_cp=[]
    for k,v in bratch0.items():
        bratch.append(v[0])
        bratch_cp.append(v[1])
    return(independent_cp0,dependent_cp0,bratch_cp,bratch)


# In[ ]:


def delete_free_radical_in_index_data(index_data):
    index_data2={}
    for k,v in index_data.items():
        if '[C]'in v[1]:
            v2=v[1].replace('[C]','C')
            index_data2[k]=[v[0],v2]
        else:
            index_data2[k]=v
    return(index_data2)


# In[ ]:


def bratch_in_string(s):
    num_list=['0','1','2','3','4','5','6','7','8','9','%']
    br_list=[]
    
    index1=0
    bratch=""
    i=0
    while i < len(s):
        n=s[i]
        if n in num_list:
            bratch=s[0:i]
            break
        i=i+1
    
    
    if len(bratch)>=6:
        if "[C]" not in bratch:
            s1=s[i-1:]
            br_list.append(bratch)
        else:
            s1=s
    else:
        s1=s
   
    j=len(s1)-1
    bratch2=''
    while j>0:
        k=s1[j]
        if k in num_list:
            bratch2=s1[j+1:]
            break
        j=j-1
    if len(bratch2)>=6:
        if "[C]" not in bratch2:
            s2=s1[:j+1]
            br_list.append(bratch2)
        else:
            s2=s1
    else:
        s2=s1
    return(s2,br_list)
    


# In[ ]:


def bratch_select(index_data0,index_data,str_df,index_cp,bratch_cp,smiles):
    cp_to_delete=[]
    cp_to_repair={}
    for i in index_cp:
        if i in bratch_cp:
            name=str(i[0])+"-"+str(i[1])
            location=found_location_in_DataFrame_double(str_df,i)
            string=index_data0[name][1]
            for j in location:
                plus=str_df[j[1]+1][j[0]]
                if plus in index_cp:
                    plus_name=str(plus[0])+"-"+str(plus[1])
                    plus_string=index_data0[plus_name][1]
                    inner_num=index_data0[plus_name][0]
                    if "CCCC" in plus_string:
                        inner_num.remove(i[0])
                        inner_num.remove(i[1])
                        cp_to_repair[plus_name]=inner_num
                        cp_to_delete.append(name)                        
    for key,v in cp_to_repair.items():
        v=inner_num
        if len(inner_num)>=1:
            inner_num.sort()
            slice_s=""
            i=0
            while i < (len(inner_num)-1):
                j=inner_num[i]
                k=inner_num[i+1]
                s1=smiles[j:k+1]
                slice_s=slice_s+s1
                if i <(len(inner_num)-2):
                    l=inner_num[i+2]
                    s2='C'
                    slice_s=slice_s+s2
                i=i+2
            v2=[int(key[0]),int(key[2])]
            index_data[key]=[v2,slice_s]
    for i in cp_to_delete:
        print(i)
        print(index_data[i])
        
    return(index_data)
    
            


# In[ ]:


def find_independent_str(smiles,smallest_r0,cp_data0,independent_cp0,dependent_cp0,bratch_cp0):
    independent_ring_string0=[]
    index_data={}
    index_data0={}
    index_cp=[]
    for i in smallest_r0:
        if i in independent_cp0:
            name=str(i[0])+"-"+str(i[1])
            p=smiles[i[0]:i[1]+1]
            if p!="([C])":
                independent_ring_string0.append(p)
                index_data[name]=[i,p]
                index_data0[name]=[i,p]
                index_cp.append(i)
    for i in bratch_cp0:
        name=str(i[0])+"-"+str(i[1])
        p=smiles[i[0]:i[1]+1]
        if p!="([C])":
            independent_ring_string0.append(p)
            index_data[name]=[i,p]
            index_data0[name]=[i,p]
            index_cp.append(i)
    for k,v in cp_data0.items():
        independent_num=[]
        inner_num=[]
        w=v[1:]
        first=v[0][0]
        last=v[0][1]
        name=str(first)+"-"+str(last)
        if v[0] in independent_cp0:
            inner_num=[first,last]
            for i in w:
                if i in independent_cp0:
                    inner_num.append(i[0])
                    inner_num.append(i[1])
                    independent_num.append(i[0])
                    independent_num.append(i[1])
                if i in bratch_cp0:
                    inner_num.append(i[0])
                    inner_num.append(i[1])
                    independent_num.append(i[0])
                    independent_num.append(i[1])
                if i in dependent_cp0:
                    if i not in smallest_r0:
                        inner_name=str(i[0])+"-"+str(i[1])
                        j=cp_data[inner_name]
                        for k in j[1:]:
                            if k in independent_cp0:
                                inner_num.append(k[0])
                                inner_num.append(k[1])
                                independent_num.append(k[0])
                                independent_num.append(k[1])
                            if k in bratch_cp0:
                                inner_num.append(k[0])
                                inner_num.append(k[1])
                                independent_num.append(k[0])
                                independent_num.append(k[1])
        if len(inner_num)>=1:
            inner_num.sort()
            slice_s=""
            i=0
            while i < (len(inner_num)-1):
                j=inner_num[i]
                k=inner_num[i+1]
                s1=smiles[j:k+1]
                slice_s=slice_s+s1
                if i <(len(inner_num)-2):
                    l=inner_num[i+2]
                    s2='C'
                    slice_s=slice_s+s2
                i=i+2
            index_data0[name]=[inner_num,slice_s]
            independent_ring_string0.append(slice_s)
            index_data[name]=[v[0],slice_s]
            index_cp.append(v[0])
    if len(independent_ring_string0)==0:
        independent_ring_string0.append(smiles)
    return(independent_ring_string0,index_data,index_cp,index_data0)


import warnings
warnings.filterwarnings('ignore')

import time
import streamlit as st

import pandas as pd
import numpy as np
import sklearn

import pickle

# 定義模型預測fucntion
def prediction(X, file_path, re_name):    
    # 防呆
    if re_name == "":
        st.write("請輸入配方編號")
        return
    else:        
        # 判斷數據是否有空值、空白欄位
        X_any = X.T.isnull().any() # 判斷row是否有任何空值        
        X_all = X.T.isnull().all() # 判斷row是否全部空值
        X_empty = (X == "").T.any() # 判斷row是否有空白欄位
        i1 = 1 # row的index起始值
        nan_ = 0 # 紀錄有缺值的row數
        empty = 0 # 紀錄有空白欄位的row數
        for any_row, all_row, empty_row in zip(X_any, X_all, X_empty):
            if empty_row: # 判斷row是否有空白欄位
                st.write("第", i1, "列數據為空白欄位，請修正")
                empty+=1
            else:
                if any_row: # 判斷是否是否有任何空值
                    if all_row: # 判斷row是否全部空值
                        X = X.drop(str(i1),axis=0)                    
                    else:
                        st.write("第", i1, "列數據有缺值，請補缺值")
                        nan_+=1
            i1+=1 # 更新index值        
        if empty == 0:
            if nan_ == 0:                
                pass
            else:
                return
        else:
            return

    # 根據機台編號、印刷長度換算總重
    if device_number == "A1":
        if length == "1000":
            weight = 30
        elif length == "2000":
            weight = 40
        elif length == "3000":
            weight = 50
        elif length == "4000":
            weight = 50
        elif length == "5000":
            weight = 60
        elif length == "6000":
            weight = 60
        elif length == "7000":
            weight = 70
        elif length == "8000":
            weight = 70
        elif length == "9000":
            weight = 80
        elif length == "10000":
            weight = 90                
    elif device_number == "A2":
        if length == "1000":
            weight = 30
        elif length == "2000":
            weight = 40
        elif length == "3000":
            weight = 50
        elif length == "4000":
            weight = 50
        elif length == "5000":
            weight = 60
        elif length == "6000":
            weight = 60
        elif length == "7000":
            weight = 65
        elif length == "8000":
            weight = 75
        elif length == "9000":
            weight = 80
        elif length == "10000":
            weight = 90
    elif device_number == "A3":
        if length == "1000":
            weight = 20
        elif length == "2000":
            weight = 35
        elif length == "3000":
            weight = 50
        elif length == "4000":
            weight = 65
        elif length == "5000":
            weight = 80
        elif length == "6000":
            weight = 95
        elif length == "7000":
            weight = 110
        elif length == "8000":
            weight = 125
        elif length == "9000":
            weight = 140
        elif length == "10000":
            weight = 155
    elif device_number == "A5":
        if length == "1000":
            weight = 25
        elif length == "2000":
            weight = 35
        elif length == "3000":
            weight = 45
        elif length == "4000":
            weight = 55
        elif length == "5000":
            weight = 65
        elif length == "6000":
            weight = 75
        elif length == "7000":
            weight = 85
        elif length == "8000":
            weight = 95
        elif length == "9000":
            weight = 105
        elif length == "10000":
            weight = 115
    elif device_number == "A6":
        if length == "1000":
            weight = 60
        elif length == "2000":
            weight = 65
        elif length == "3000":
            weight = 70
        elif length == "4000":
            weight = 75
        elif length == "5000":
            weight = 80
        elif length == "6000":
            weight = 85
        elif length == "7000":
            weight = 90
        elif length == "8000":
            weight = 95
        elif length == "9000":
            weight = 100
        elif length == "10000":
            weight = 105
    elif device_number == "A7":
        if length == "1000":
            weight = 30
        elif length == "2000":
            weight = 35
        elif length == "3000":
            weight = 40
        elif length == "4000":
            weight = 45
        elif length == "5000":
            weight = 50
        elif length == "6000":
            weight = 55
        elif length == "7000":
            weight = 60
        elif length == "8000":
            weight = 65
        elif length == "9000":
            weight = 65
        elif length == "10000":
            weight = 70
                
    ###
##    st.write("配墨總重:"+str(weight))       
    ###

    # 模型預測
    concat_col = pd.DataFrame(columns=out_col_list)
    for y_num, y_name in enumerate(out_col_list):        
        # 根據L數值選其對應功能檔案(標準化、機器學習模型)
        concat_row = np.array([])
        for row_X in range(X.shape[0]):            
            if float(X.iloc[row_X, [0]]) >= 65:
                model_path = "65u\\"
            else:                
                model_path = "65b\\"
            # 讀取scaler
            with open(file_path + model_path + "in.pickle", 'rb') as f:
                sc_X = pickle.load(f)
            with open(file_path + model_path + "out.pickle", 'rb') as f:
                sc_y = pickle.load(f)
            # 讀取模型
            with open(file_path + model_path + "RF_" + str(y_num) + ".pickle", "rb") as f:
                model = pickle.load(f)
            # 標準化轉換與預測
            X_sc = sc_X.transform(np.array(X.iloc[row_X,:]).reshape(1,-1)) # 1個row,3個column(L, a, b)
            y_pred_sc = model.predict(X_sc)            
            concat_row = np.append(concat_row, y_pred_sc)            
        concat_col[y_name] = concat_row    
    df_output = pd.DataFrame(sc_y.inverse_transform(concat_col.values),
                             columns=out_col_list)

    ###
##    st.write("預測結果")
##    st.write(df_output)
    ###    
    
    # 數據轉換
    df_output_total = [] # total欄位
    df_output_id = [] # ID欄位
    for row in range(df_output.shape[0]):
        # 全部總和為100
        df_output.iloc[row,:] = df_output.iloc[row,:]/df_output.iloc[row,:].sum()*100            
        # S、A配方固含量校正
        df_S = df_output.iloc[row,[4]]        
        df_A = df_output.iloc[row,[5]]        
        df_S_new = df_S - df_A.values*(9) #S校正(Snew = Sold+Aold*(1/0.9))
        df_output.iloc[row,[4]] = df_S_new
        df_A_new = df_A*10 #A校正(Anew=Aold*10)
        df_output.iloc[row,[5]] = df_A_new        
        # 根據機台、印刷長度進行總和換算
        df_output.iloc[row,:] = df_output.iloc[row,:]/df_output.iloc[row,:].sum()*weight
        # total欄位
        df_output_total.append(round(df_output.iloc[row,:].sum()))
        # ID欄位
        df_output_id.append(recipe_name+"-"+str(row+1)) 

    # 輸出結果(加total欄位、改ID欄位)
    df_output["total"] = df_output_total
    df_output[str(recipe_name)] = df_output_id
    df_output = df_output.set_index(str(recipe_name))

    # st.write("最終結果(配方100%、SA固含量校正、根據機台與印刷長度換算總重)")
    st.write("預測結果")
    st.write(df_output)    
    
    return(df_output)

# 定義下載Fuction
def convert_df(df):    
    return df.to_csv().encode('utf-8')

# 設定網頁標題(標籤頁、網頁最上方)
st.set_page_config(page_title="油墨配方製程推薦", layout="wide")
st.title("油墨配方製程推薦")

# 配方編號
recipe_name = st.text_input("請輸入配方編號",value="")

# 機台編號
device_number = st.selectbox("請輸入機台編號", ("A1", "A2", "A3", "A5", "A6", "A7"))

# 印刷長度
length = st.selectbox("請輸入印刷長度", ("1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000", "10000",))

# Lab值輸入表格
st.write('<p style="font-size:14px">請輸入L、a、b值</p>', unsafe_allow_html=True)
df_input = pd.DataFrame(columns = ["L", "a", "b"],
                        index = ["1", "2" , "3", "4", "5"]
                        )
edited_df = st.data_editor(df_input)

path = "file\\" # 儲存功能檔案的位置    

# 輸出欄位名稱
out_col_list = ["Y", "M", "C", "K", "S", "A"]


# 執行按鈕
if st.button("預測"):
    input_data = edited_df.copy()
    pred_re = prediction(input_data, path, recipe_name)
    pred_re = pred_re.drop(columns = "total")
    # 下載按鈕
    st.write("注意: 點選下載後，於頁面顯示的結果會消失")
    csv = convert_df(pred_re)
    st.download_button(label="下載結果(CSV)",
                       data=csv,
                       file_name='result.csv',
                       mime='text/csv',
                       )

# -*- coding:utf-8 -*- 

import os
import csv
import errno
import numpy as np
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io

def file_processing(file_path, encode=None):
    data = []

    with io.open(file_path, encoding=encode) as file:
        rows = csv.reader(file, delimiter=",")
        n_row = 0

        for row in rows:
            if n_row != 0:
                #column -> 0: code, 1: date
                for column in range(2, len(row)):
                    data[n_row - 1].append(float(row[column].strip()))

            data.append([])
            n_row += 1

    del data[-1]
    return np.array(data)

def normalize_data(data, t_data, scaler):
    minmaxscaler = scaler.fit(data)
    normalize_data = minmaxscaler.transform(data)
    t_data = minmaxscaler.transform(t_data)

    return normalize_data,t_data

def inverse_normalize_data(data, scaler):
    for i in range(len(data)):
        data[i] = scaler.inverse_transform(data[i])

    return data

def generate_output(output, model_name, class_list):
    class_list = class_list
    _output = []

    for i in range(len(output)):
        _output.append([])
        _output[i].append(class_list[i])
        for j in range(len(output[i]) - 1):
            if output[i][j+1] > output[i][j]:
                _output[i].append(1)
                _output[i].append(output[i][j+1])
            elif output[i][j+1] == output[i][j]:
                _output[i].append(0)
                _output[i].append(output[i][j+1])
            else:
                _output[i].append(-1)
                _output[i].append(output[i][j+1])

    file_path = 'outputs/output_{}.csv'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(file_path, 'w+') as file:
        w = csv.writer(file)

        w.writerow(['ETFid','Mon_ud','Mon_cprice','Tue_ud','Tue_cprice','Wed_ud','Wed_cprice','Thu_ud','Thu_cprice','Fri_ud','Fri_cprice'])
        w.writerows(_output)

def plot_model_architecture(model, model_name):
    file_path = 'images/model/{}.png'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plot_model(model, to_file=file_path, show_shapes=True)

def save_model(model, model_name):
    file_path = 'model/{}.h5'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    model.save(file_path)

def plot_predict(after_day,data, data_predict,test_predict, file_name):
    file_path = 'images/result/{}.png'.format(file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    tdata=test_predict[:, :, 3][0];
    fig = plt.figure(figsize=(30, 20))
    if after_day>6:#最多画6个
        after_day=6
    for i in range(after_day):
        ax1 = fig.add_subplot(231+i)
        ax1.plot(data[:, i, 3], color='black')
        pdata=data_predict[:, i, 3];
        ax1.plot(pdata, color='red')
        ax1.scatter(list(range(len(pdata),len(pdata)+len(tdata))),tdata,color='blue')
        ax1.title.set_text("Day "+str(i+1))#验证集87个滑动窗口，各自预测的第一天的结果
    plt.savefig(file_path)
    #plt.show()
def plot_predict(after_day,data,
                 data_predict,test_predict,
                 validate_predict_seq2seq,test_predict_seq2seq,
                 validate_predict_lstm_mtm,test_predict_lstm_mtm,
                 file_name):
    file_path = 'images/result/{}.png'.format(file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    tdata=test_predict[:, :, 3][0];
    fig = plt.figure(figsize=(30, 20))
    if after_day>6:#最多画6个
        after_day=6
    for i in range(after_day):
        ax1 = fig.add_subplot(231+i)
        pdata=data_predict[:, i, 3];
        pdata_seq2seq=validate_predict_seq2seq[:, i, 3];
        pdata_mtm=validate_predict_lstm_mtm[:, i, 3];
        ax1.plot(data[:, i, 3],label='real',color='#ff0000',linestyle='-',linewidth=1)
        ax1.plot(pdata,label='lstm',color='#00ff00',linestyle='-',linewidth=1)
        ax1.plot(pdata_seq2seq,label='seq2seq',color='#00ffff',linestyle='-',linewidth=1)
        ax1.plot(pdata_mtm,label='lstm_mtm',color='#000000',linestyle='-',linewidth=1)
        ax1.scatter(list(range(len(pdata),len(pdata)+len(tdata))),tdata,color='blue',label='predict')
        ax1.title.set_text("Day "+str(i+1))#验证集87个滑动窗口，各自预测的第一天的结果
        ax1.legend(loc=2,fontsize=20)
    plt.savefig(file_path)
    #plt.show()

#创建图片的目录，及图片文件
def getFilePath(file_name):
    file_path = 'images/result/{}.png'.format(file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return file_path;

#画图
def drawImg(all_days_mse,
        all_days_seq_mse,
        all_days_mtm_mse,type,days,file_name):
    # all_days_mape=np.array(all_days_mape).T
    # all_days_mse=np.array(all_days_mse).T
    # plt.plot(all_days_mape,color='r',linewidth=2,linestyle='-',label='mape')#color指定线条颜色，labeL标签内容
    # plt.plot(all_days_mse,color='g',linewidth=2,linestyle='--',label='mse')#linewidth指定线条粗细
    # plt.legend(loc=2)#标签展示位置，数字代表标签具位置
    # plt.xlabel('day')
    # plt.ylabel('误差')
    # plt.title('折线图标题')

    plt.rcParams['font.sans-serif']=['SimHei']  #使用指定的汉字字体类型（此处为黑体）
    plt.figure(figsize=(20,10)) #设置图表大小，长10，宽5
    # plt.plot(all_days_mape,label='lstm',color='#aa0000',linestyle='-',linewidth=1)
    # plt.plot(all_days_seq_mape,label='seq2seq',color='#00aa00',linestyle='-',linewidth=1)
    plt.plot(all_days_mse,label='lstm',color='#aa0000',linestyle='-',linewidth=1)
    plt.plot(all_days_seq_mse,label='seq2seq',color='#00aa00',linestyle='-',linewidth=1)
    plt.plot(all_days_mtm_mse,label='lstm_mtm',color='#0000aa',linestyle='-',linewidth=1)
    plt.legend(loc=2,fontsize=20)
    plt.title('各算法模型'+type,fontsize=20)#命名图表名称，设置字体大小
    # plt.xlabel('日',fontsize=10)#设置X轴名称及字体大小
    # plt.ylabel('误差',fontsize=10)#设置Y轴名称及字体大小
    plt.xticks(list(range(days)))
    file_path=getFilePath(file_name+"_"+type);
    plt.savefig(file_path)
    #plt.show()
def plot_mape(days,data,
              data_predict,
              validate_predict_seq2seq,
              validate_predict_mtm,
              file_name):
    fig = plt.figure(figsize=(15, 10))
    all_days_mape=[]
    all_days_seq_mape=[]
    all_days_mtm_mape=[]
    all_days_mse=[]
    all_days_seq_mse=[]
    all_days_mtm_mse=[]
    all_days_mae=[]
    all_days_seq_mae=[]
    all_days_mtm_mae=[]
    #准备画图的数据
    for i in range(0,days):
        day_mape=mape(data[:, i, 3],data_predict[:, 0, 3])
        seq_day_mape=mape(data[:, i, 3],validate_predict_seq2seq[:, 0, 3])
        mtm_day_mape=mape(data[:, i, 3],validate_predict_mtm[:, 0, 3])
        day_mse=mse(data[:, i, 3],data_predict[:, 0, 3])
        seq_day_mse=mse(data[:, i, 3],validate_predict_seq2seq[:, 0, 3])
        mtm_day_mse=mse(data[:, i, 3],validate_predict_mtm[:, 0, 3])
        day_mae=mae(data[:, i, 3],data_predict[:, 0, 3])
        seq_day_mae=mae(data[:, i, 3],validate_predict_seq2seq[:, 0, 3])
        mtm_day_mae=mae(data[:, i, 3],validate_predict_mtm[:, 0, 3])
        #mape指标
        all_days_mape.append(day_mape)
        all_days_seq_mape.append(seq_day_mape)
        all_days_mtm_mape.append(mtm_day_mape)
        #mse指标
        all_days_mse.append(day_mse)
        all_days_seq_mse.append(seq_day_mse)
        all_days_mtm_mse.append(mtm_day_mse)
        #mae指标
        all_days_mae.append(day_mae)
        all_days_seq_mae.append(seq_day_mae)
        all_days_mtm_mae.append(mtm_day_mae)
    #开始画图
    #mape指标
    drawImg(all_days_mape,
            all_days_seq_mape,
            all_days_mtm_mape,"mape指标",days,file_name)
    #mse指标
    drawImg(all_days_mse,
            all_days_seq_mse,
            all_days_mtm_mse,"mse指标",days,file_name)
    #mae指标
    drawImg(all_days_mae,
            all_days_seq_mae,
            all_days_mtm_mae,"mae指标",days,file_name)

def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape
def mse(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mse -- MSE 评价指标
    """

    n = len(y_true)
    mse = sum(np.square(y_true - y_pred))/n
    return mse
def mae(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    返回:
    mae -- MAE 评价指标
    """
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae
def plot_loss(history, file_name):
    file_path = 'images/loss/{}.png'.format(file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(file_path)
    # plt.show()

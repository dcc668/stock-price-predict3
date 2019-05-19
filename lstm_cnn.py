# -*- coding:utf-8 -*-

import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Conv1D
from keras.layers.core import Dense, Activation, Dropout, Lambda, RepeatVector
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from seq2seq import seq2seq_model
from lstm_mtm import lstm_mtm_model
from utils import *
import tushare as ts
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 根据time_step（20）天的数据预测after_day（10）天的数据,来训练模型
#以seq_length的长度（30天）滑动窗口，添加到训练集
def load_data(nor_data, time_step=20, after_day=1, validate_percent=0.8):
    seq_length = time_step + after_day
    result = []
    for index in range(len(nor_data) - seq_length + 1):#以seq_length的长度（30天）滑动窗口，添加到训练集
        result.append(nor_data[index: index + seq_length])

    result = np.array(result)
    print('total data: ', result.shape)

    train_size = int(len(result) * validate_percent)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :time_step]
    y_train = train[:, time_step:]
    x_validate = validate[:, :time_step]
    y_validate = validate[:, time_step:]

    return [x_train, y_train, x_validate, y_validate]


#feature_len 特征的个数（开盘价，收盘价，交易量...）
#after_day 预测的天数
#input_shape(time_step, feature_len),time_step:行,输入多少条数据,feature_len，每条数据有多少列（特征值）
def base_model(feature_len=1, after_day=1, input_shape=(20, 1)):
    # 序列模型是一个线性的层次堆栈
    model = Sequential()
    # 1维卷积层，用于输入一维输入信号进行领域滤波
    model.add(Conv1D(10, kernel_size=5, input_shape=input_shape,
                     activation='relu', padding='valid', strides=1))
    # 循环层
    model.add(LSTM(100, return_sequences=False, input_shape=input_shape))
    # Dropout层，防止过拟合
    model.add(Dropout(0.15))

    # one to many
    #  RepeatVector层将输入重复n次
    model.add(RepeatVector(after_day))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.15))

    # 包装器，把一层应用到每一个时间步上
    # Dense 全连接层
    model.add(TimeDistributed(
        Dense(100, activation='relu', kernel_initializer='uniform')))
    model.add(TimeDistributed(
        Dense(feature_len, activation='linear', kernel_initializer='uniform')))

    return model

def lstm_model(feature_len, after_day, input_shape,model_name,
                x_train, y_train, batch_size,_class,
                epochs, x_validate, y_validate
               ):
    model = base_model(feature_len, after_day, input_shape)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    #plot_model_architecture(model, model_name=model_name)

    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, validation_data=(x_validate, y_validate))
    model_class_name = model_name + '_00{}'.format(_class)
    save_model(model, model_name=model_class_name)

    print('-' * 100)
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train Score: %.8f MSE (%.8f RMSE)' %
          (train_score[0], math.sqrt(train_score[0])))

    validate_score = model.evaluate(x_validate, y_validate, verbose=0)

    ##################### 管道重定向########################################
    temp = sys.stdout
    Test_file = open('Test_to_RNN.csv', 'w')
    sys.stdout = Test_file
    print('Test Score: %.8f MSE (%.8f RMSE)' %
          (validate_score[0], math.sqrt(validate_score[0])))
    sys.stdout.close()
    sys.stdout = temp
    ##################### 管道打印########################################

    print('Test Score: %.8f MSE (%.8f RMSE)' %
          (validate_score[0], math.sqrt(validate_score[0])))
    train_predict = model.predict(x_train)
    validate_predict = model.predict(x_validate)
    test_predict = model.predict(x_test)
    return [train_predict,
            validate_predict,
            test_predict,history]


def plot_accuracy(history,file_name):
    # accuracy plot
    file_path = 'images/result/{}.png'.format(file_name)
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.savefig(file_path)


if __name__ == '__main__':
    class_list = ['600036']# 399001,300274,600016,600030,600036
    # class_list = input("请输入6位代码：") #输入股票代码
    # class_list=[class_list]
    # after_day = input("请输入预测天数：") #输入预测多少天后的价格
    # after_day = int(after_day)
    scaler = MinMaxScaler(feature_range=(0, 1))
    validate_percent = 0.9
    time_step = 30  #s使用time_step（20）天的数据，预测after_day（10）天的数据
    after_day = 10 #要预测多少天
    batch_size = 32
    epochs =10
    output = []

    model_name = sys.argv[0].replace(".py", "")
    names=model_name.split("/");
    model_name=names[len(names)-1]
    for index in range(len(class_list)):
        _class = class_list[index]
        print('******************************************* class 00{} *******************************************'.format(_class))

        # read data from csv, return data: (Samples, feature)
        # data = file_processing(
        #     'data/20181009_process/20181009_{}.csv'.format(_class))
        #===========改用tushare数据=====================start============
        ts_data=ts.get_k_data(_class,start='2011-01-01',end='2019-05-16')
        np_array_data=ts_data.values
        data=np_array_data[:, 1:7]
        print(data.shape)
        #===========改用tushare数据=====================end============
        # normalize data
        train_size=data.shape[0]-time_step;
        feature_len = data.shape[1] - 1  # 数据集总条数
        data = data[:, :5]
        train_data=data[:train_size];

        # test data
        x_test = data[-time_step:]
        nor_data,x_test = normalize_data(train_data,x_test,scaler)
        x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

        # get train and validate data
        x_train, y_train, x_validate, y_validate = load_data(
            nor_data, time_step=time_step, after_day=after_day, validate_percent=validate_percent)

        print('train data: ', x_train.shape, y_train.shape)
        print('validate data: ', x_validate.shape, y_validate.shape)

        # model complie
        #----------------------------lstm------start------------------------------------
        input_shape = (time_step, feature_len)
        train_predict,validate_predict,test_predict,history=lstm_model(feature_len, after_day, input_shape,model_name,
                   x_train, y_train, batch_size,_class,
                   epochs, x_validate, y_validate
                   )
        #----------------------------lstm------end------------------------------------
        #----------------------------seq2seq------start------------------------------------
        train_predict_seq2seq,validate_predict_seq2seq,test_predict_seq2seq=seq2seq_model(
            feature_len, after_day, input_shape,batch_size,epochs,_class,scaler,
            x_train, y_train, x_validate, y_validate,x_test)
        #---------------------------seq2seq---------end----------------------------------

        #----------------------------lstm_mtm------start------------------------------------
        train_predict_lstm_mtm,validate_predict_lstm_mtm,test_predict_lstm_mtm=lstm_mtm_model(
            feature_len, after_day, input_shape,batch_size,epochs,_class,scaler,
            x_train, y_train, x_validate, y_validate,x_test)
        #---------------------------lstm_mtm---------end----------------------------------
        # lstm回復預測資料值為原始數據的規模
        train_predict = inverse_normalize_data(train_predict, scaler)
        y_train = inverse_normalize_data(y_train, scaler)
        validate_predict = inverse_normalize_data(validate_predict, scaler)
        y_validate = inverse_normalize_data(y_validate, scaler)
        test_predict = inverse_normalize_data(test_predict, scaler)
        # seq2seq回復預測資料值為原始數據的規模
        train_predict_seq2seq = inverse_normalize_data(train_predict_seq2seq, scaler)
        validate_predict_seq2seq = inverse_normalize_data(validate_predict_seq2seq, scaler)
        test_predict_seq2seq = inverse_normalize_data(test_predict_seq2seq, scaler)
        # lstm_mtm回復預測資料值為原始數據的規模
        train_predict_lstm_mtm = inverse_normalize_data(train_predict_lstm_mtm, scaler)
        validate_predict_lstm_mtm = inverse_normalize_data(validate_predict_lstm_mtm, scaler)
        test_predict_lstm_mtm = inverse_normalize_data(test_predict_lstm_mtm, scaler)
        # 3 or 0: close 的位置, 0:5為五天
        ans = np.append(y_validate[-1, -1, 3], test_predict[-1, 0:after_day, 3])
        output.append(ans)
        # print("output: \n", output)

        # plot predict situation (save in images/result)
        file_name = 'result_' + '_00{}'.format(_class)
        #1.预测值与真实值 图像
        plot_predict(after_day,y_validate,
                     validate_predict, test_predict,
                     validate_predict_seq2seq,validate_predict_seq2seq,
                     validate_predict_lstm_mtm,test_predict_lstm_mtm,
                     file_name=file_name)
        #2.预测准确率与迭代次数的关系
        plot_accuracy(history,file_name+"_accuracy")

        file_name = 'result_' + '_11{}'.format(_class)
        plot_mape(after_day,y_validate,
                  validate_predict,
                  validate_predict_seq2seq,
                  validate_predict_lstm_mtm,
                  file_name=file_name)

        # plot loss (save in images/loss)
        file_name = 'loss_' + '_00{}'.format(_class)
        plot_loss(history, file_name)

    output = np.array(output)
    print(output)
    generate_output(output, model_name=model_name, class_list=class_list)

# -*- coding:utf-8 -*-

import sys
import datetime
import time

import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Conv1D
from keras.layers.core import Dense, Activation, Dropout, Lambda, RepeatVector
from keras.layers.recurrent import LSTM
# 回调函数，每个epoch保存模型到filepath
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

import pymysql
import sqlalchemy
from sqlalchemy.orm import sessionmaker


from utils import *

# 数据库model
from models import ENGINE, Stock

# 数据库配置
config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'fintech',
    'password': 'fintech',
    'db': 'fintech',
          'charset': 'utf8mb4',
          'cursorclass': pymysql.cursors.DictCursor,
}

Session = sessionmaker(bind=ENGINE)
session = Session()

# 加载数据


def load_data(data, time_step=20, after_day=1, validate_percent=0.8):
    seq_length = time_step + after_day
    result = []
    for index in range(len(data) - seq_length + 1):
        result.append(data[index: index + seq_length])

    result = np.array(result)
    # 获取数据据集总数
    print('total data: ', result.shape)

    train_size = int(len(result) * validate_percent)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :time_step]
    y_train = train[:, time_step:]
    x_validate = validate[:, :time_step]
    y_validate = validate[:, time_step:]

    # validate 验证
    return [x_train, y_train, x_validate, y_validate]


def base_model(feature_len=1, after_day=1, input_shape=(20, 1)):
    # 序列模型是一个线性的层次堆栈
    model = Sequential()
    # 1维卷积层，用于输入一维输入信号进行领域滤波
    model.add(Conv1D(after_day, kernel_size=input_shape[0], input_shape=input_shape,
                     activation='relu', padding='valid', strides=1))
    # 循环层
    model.add(LSTM(100, return_sequences=False, input_shape=input_shape))
    # Dropout层，防止过拟合
    model.add(Dropout(0.25))

    # one to many
    #  RepeatVector层将输入重复n次
    model.add(RepeatVector(after_day))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.25))

    # 包装器，把一层应用到每一个时间步上
    # Dense 全连接层
    model.add(TimeDistributed(
        Dense(100, activation='relu', kernel_initializer='uniform')))
    model.add(TimeDistributed(
        Dense(feature_len, activation='linear', kernel_initializer='uniform')))

    return model


if __name__ == '__main__':

    # 开始时间
    start = time.time()
    # 股票列表
    class_list = ['dailies']

    # 从数据库获取股票列表
    stocks = session.query(Stock).filter(Stock.id < 3)
    # 按列做minmax缩放
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集
    validate_percent = 0.9
    # 根据前20天数据预测未来5天
    time_step = 20
    after_day = 5
    """指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。"""
    batch_size = 64
    epochs = 10
    output = []

    model_name = sys.argv[0].replace(".py", "")

    # 从数据库dailies数据
    try:
        conn = pymysql.connect(**config)
    except pymysql.err.OperationalError as e:
        print('Error is '+str(e))
        sys.exit()

    for stock in stocks:

        print('******************************************* stock: {} *******************************************'.format(stock.name))

        sql = 'SELECT `popen`, `phigh`, `plow`, `pclose`, `volume` FROM `dailies` WHERE `stock_id`=%d ORDER BY `date` ASC' % stock.id
        data = pd.read_sql(sql, con=conn)
        # 打印输入数据
        print(data.values)

        feature_len = data.shape[1]

        # 归一化数据
        data = normalize_data(data, scaler, feature_len)

        # test data
        x_test = data[-time_step:]
        x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

        # get train and validate data
        x_train, y_train, x_validate, y_validate = load_data(
            data, time_step=time_step, after_day=after_day, validate_percent=validate_percent)
        # 训练集总数
        print('train data: ', x_train.shape, y_train.shape)
        # 验证集总数
        print('validate data: ', x_validate.shape, y_validate.shape)

        # model complie
        # 模型训练的BP模式设置
        input_shape = (time_step, feature_len)
        model = base_model(feature_len, after_day, input_shape)
        '''
        Model模型方法
        具体参考：https://keras-cn.readthedocs.io/en/latest/models/model/
        '''
        # compile 训练模式
        # optimizer：优化器
        # metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=[‘accuracy’]
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # 模型概括打印
        model.summary()
        plot_model_architecture(model, model_name=model_name)

        '''
            x_train:输入数据;
            y_train:标签;
            batch_size:整数，进行梯度下降时候每个batch包含的样本数;
            epochs:训练终止的epoch值;
            validation_data:形式为（X，y）的tuple，是指定的验证集。
            model.fit()函数教程：
            https://cnbeining.github.io/deep-learning-with-python-cn/4-advanced-multi-layer-perceptrons-and-keras/ch15-understand-model-behavior-during-training-by-plotting-history.html
        '''
        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=(x_validate, y_validate))
        # 打印历史数据
        print(history.history.keys())

        # 管道重定向
        temp = sys.stdout
        acc_file = open('cnn.csv', 'w')
        sys.stdout = acc_file
        print(history.history['acc'])   # 模型在训练集的准确率
        sys.stdout.close()
        sys.stdout = temp
        print(history.history['acc'])

        '''
            fit函数返回一个History的对象，
            其History.history属性记录了损失函数
            和其他指标的数值随epoch变化的情况，如果有验证集的话，
            也包含了验证集的这些指标变化情况。
        '''
        # model_class_name = model_name + '_00{}'.format(_class)
        model_class_name = model_name + '{}'.format(stock.code)
        # save_model(model, model_name=model_class_name)

        print('-' * 100)
        # 训练集 模型评估
        '''
            x：输入数据，与fit一样，是numpy array或numpy array的list
            y：标签，是numpy array或numpy array的list
            batch_size：整数，含义同fit的同名参数
            verbose：含义同fit的同名参数，但只能取0或1
            sample_weight：numpy array，含义同fit的同名参数
        '''
        train_score = model.evaluate(x_train, y_train, verbose=0)
        # 计算测试集均方误差MSE
        print('Train Score: %.8f MSE (%.8f RMSE)' %
              (train_score[0], math.sqrt(train_score[0])))
        # 验证机 模型评估
        validate_score = model.evaluate(x_validate, y_validate, verbose=0)
        # 计算验证集均方误差MSE平日你
        print('Test Score: %.8f MSE (%.8f RMSE)' %
              (validate_score[0], math.sqrt(validate_score[0])))
        # 模型预测
        train_predict = model.predict(x_train)
        validate_predict = model.predict(x_validate)
        test_predict = model.predict(x_test)
        # 恢复预测资料值为原始数据规模
        train_predict = inverse_normalize_data(train_predict, scaler)
        y_train = inverse_normalize_data(y_train, scaler)
        validate_predict = inverse_normalize_data(validate_predict, scaler)
        y_validate = inverse_normalize_data(y_validate, scaler)
        test_predict = inverse_normalize_data(test_predict, scaler)

        # 3 or 0: close 的位置, 0:5為五天
        ans = np.append(y_validate[-1, -1, 3], test_predict[-1, 0:5, 3])
        output.append(ans)
        # print("output: \n", output)

        # plot predict situation (save in images/result)
        # file_name = 'result_' + model_name + '_00{}'.format(_class)
        file_name = 'result_' + model_name + '{}'.format(stock.code)
        plot_predict(y_validate, validate_predict, file_name=file_name)

        # plot loss (save in images/loss)
        # file_name = 'loss_' + model_name + '_00{}'.format(_class)
        file_name = 'loss_' + model_name + '{}'.format(stock.code)
        plot_loss(history, file_name)

        # plot loss (save in images/predict)
        # file_name = 'predict_' + model_name + '_00{}'.format(_class)
        file_name = 'predict_' + model_name + '{}'.format(stock.code)
        plot_forecast(history, file_name)
    output = np.array(output)
    print('ok')
    # 结束时间
    end = time.time()
    # 运算时间
    print('%f seconds' % (end-start))
    print(output)
    generate_output(output, model_name=model_name, class_list=class_list)

    # 关闭数据库连接
    conn.close()

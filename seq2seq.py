#! -*- coding: utf-8 -*-

import numpy as np
import pymongo
from tqdm import tqdm
import os,json
#import uniout
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas as pd
import tensorflow as tf


min_count = 5
maxlen = 800
batch_size = 1024
epochs = 1     #原来是100 测试方便改为1
char_size = 128
db = pymongo.MongoClient().text.news2 # 连接数据库

if os.path.exists('seq2seq_config.json'):  # 配置文件
    chars,id2char,char2id = json.load(open('seq2seq_config.json'))
    id2char = {int(i):j for i,j in id2char.items()}
else:
    chars = {}
    for a in tqdm(db.find()):
        for w in a['content']: # 纯文本，不用分词
            chars[w] = chars.get(w,0) + 1
        for w in a['title']: # 纯文本，不用分词
            chars[w] = chars.get(w,0) + 1
    chars = {i:j for i,j in chars.items() if j >= min_count}
    # 0: mask
    # 1: unk
    # 2: start
    # 3: end
    id2char = {i+4:j for i,j in enumerate(chars)}
    char2id = {j:i for i,j in id2char.items()}
    json.dump([chars,id2char,char2id], open('seq2seq_config.json', 'w'))

def str2id(s, start_end=False):
    if start_end: # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen-2]]
        ids = [2] + ids + [3]
    else: # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    ml = max([len(i) for i in x])
    return [i + [0] * (ml-len(i)) for i in x]


def data_generator():
    X,Y = [],[]
    while True:
        for a in db.find():
            X.append(str2id(a['content']))
            Y.append(str2id(a['title'], start_end=True))
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X,Y], None
                X,Y = [],[]


# 搭建seq2seq网络模型 基于GPU的BiLSTM

x_in = Input(shape=(None,))
y_in = Input(shape=(None,))
x = x_in
y = y_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)


def to_one_hot(x): # 创建先验词典
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars)+4)
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x


class ScaleShift(Layer):   # 先验词典维度变换
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def build(self, input_shape):
        kernel_shape = (1,)*(len(input_shape)-1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')
    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs

#--------------------------------------------------------------------------------------
x_one_hot = Lambda(to_one_hot)([x, x_mask])
x_prior = ScaleShift()(x_one_hot) # 学习先验分布

embedding = Embedding(len(chars)+4, char_size)
x = embedding(x)
y = embedding(y)


# encoder，双层双向LSTM
x = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x)

# decoder，双层单向LSTM
y = CuDNNLSTM(char_size, return_sequences=True)(y)
y = CuDNNLSTM(char_size, return_sequences=True)(y)


class Interact(Layer):
    def __init__(self, **kwargs):
        super(Interact, self).__init__(**kwargs)
    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        out_dim = input_shape[1][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(in_dim, out_dim),
                                      initializer='glorot_normal')
    def call(self, inputs):
        q, v, v_mask = inputs
        k = v
        mv = K.max(v - (1. - v_mask) * 1e10, axis=1, keepdims=True) # maxpooling1d
        mv = mv + K.zeros_like(q[:,:,:1]) 
        
	# 注意力机制
        qw = K.dot(q, self.kernel)
        a = K.batch_dot(qw, k, [2, 2]) / 10.
        a -= (1. - K.permute_dimensions(v_mask, [0, 2, 1])) * 1e10
        a = K.softmax(a)
        o = K.batch_dot(a, v, [2, 1])
 
        return K.concatenate([o, q, mv], 2)
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1],
                input_shape[0][2]+input_shape[1][2]*2)
        


xy = Interact()([y, x, x_mask])
xy = Dense(512, activation='relu')(xy)
xy = Dense(len(chars)+4)(xy)
xy = Lambda(lambda x: (x[0]+x[1])/2)([xy, x_prior]) # 与先验词典向量融合计算
xy = Activation('softmax')(xy)



cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
loss = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])


model = Model([x_in, y_in], xy)
model.add_loss(loss)
model.compile(optimizer=Adam(1e-3))


def gen_title(s, topk=5):    # beam search

    xid = np.array([str2id(s)] * topk) 
    yid = np.array([[2]] * topk) 
    scores = [0] * topk 
    for i in range(50): # 控制生成长度
        proba = model.predict([xid, yid])[:, i, 3:] 
        log_proba = np.log(proba + 1e-6) # 取对数
        arg_topk = log_proba.argsort(axis=1)[:,-topk:] # 选出topk
        _yid = [] 
        _scores = [] 
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]+3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk): # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k]+3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:] # 选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == 3: # 找到<end>就返回
                return id2str(_yid[k])
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    # 找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)])



s1 = '1979年伊朗爆发革命，王朝被推翻，政教二元化的伊斯兰共和国成立，许多王公贵族、企业家、银行家纷纷外流，随后伊朗和伊拉克陷入长达8年的两伊战争。伊朗经济由此陷入崩溃，至1989年两伊战争告一段落，里亚尔兑美元汇率已缩水了一半。进入21世纪，内贾德上台，强推核计划，并与美欧全面对抗，结果遭至严厉国际制裁，令伊朗通胀加剧，至2013年内贾德卸任，美元对里亚尔的汇率已膨胀至其上任之初的400%。2015年伊朗核协定（JCPOA）的达成曾令伊朗经济一度趋于回暖，低迷已久的里亚尔也恍惚看到转机。但好景不长，2018年5月8日，特朗普宣布美国单方面退出JCPOA，随即以“让伊朗石油出口归零”为目标，对伊朗实施变本加厉的封锁、禁运。'
s2 = '5G在中国正式商用才半年，就遭遇了一场前所未有的大“练兵”。比如，远程医疗会诊需要清晰流畅的图像共享，远程医疗检查和手术更是容不得哪怕一秒的滞后和卡顿，5G大带宽、低时延、广连接的特性最适合打通这个“生命通道”。据统计，疫情发生以来3家电信央企累计开通5G基站13万个，全力保障战“疫”通信需要。在5G这条信息高速路上，全国各地优秀医疗资源为医院提供了强有力的远程支撑。'

s3 = input("输入文本：")
class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):

        print(gen_title(s1))
        print(gen_title(s2))
        #print(gen_title(s3))
        # 保存最优结果
        
        #if logs['loss'] <= self.lowest:
            #self.lowest = logs['loss']
        model.save_weights('./best_model.weights')
            #model.save('./best_model.h5')

###############################调试分界################################################



evaluator = Evaluate()

model.fit_generator(data_generator(),
                    steps_per_epoch=5,
                    epochs=epochs,
                    callbacks=[evaluator])

#model.fit_generator(data_generator(),
                   # steps_per_epoch=1000,
                   # epochs=epochs,
                  #  callbacks=[evaluator])
#为了测试修改epochs为10
###############################调试分界################################################

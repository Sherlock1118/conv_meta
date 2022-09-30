"""Deep Neural Network Estimator for Market, build with tf.layers. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pyspark.sql import Row, SparkSession, SQLContext, Window
import pyspark.sql.functions as F

import pyspark.sql.types as T
from aibrain_common.utils import oss_utils
from aibrain_job.utils import param_utils

import logging
import tensorflow as tf
from aibrain_common.conf import tf_context
from aibrain_common.data.dataset_builder import (ColumnSpec, DatasetBuilder)
from tensorflow.python.keras.layers import Flatten
import numpy as np
from keras import optimizers
from feature import *
import json
from model import deepFm, xdeepFm, dcn,DNN,DNN_scope
from aibrain_common.component import tools
from aibrain_common.utils.date_convert_utils import DateConvertUtils
date_converter = DateConvertUtils()
date_converter.set_biz_date("20220916")
date = date_converter.parse_data_date("${yyyymmdd}")
pre1day = date_converter.parse_data_date("${yyyymmdd - 1}")

tf.logging.set_verbosity(tf.logging.INFO)

# spark = SparkSession.builder.\
#         config('spark.executor.memory', '14g').\
#         config('spark.executor.cores', '6').\
#         config('spark.driver.memory','10g').\
#         config('spark.dynamicAllocation.minExecutors', '8').\
#         config('spark.dynamicAllocation.maxExecutors', '100').\
#         config('spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version', '2').\
#         config('spark.sql.execution.arrow.enabled', 'false').\
#         config('spark.driver.maxResultSize', '200g').\
#         config('spark.sql.shuffle.partitions', '2000').\
#         config('spark.default.parallelism', '800').\
#         config('spark.sql.crossJoin.enabled', 'true').\
#         config('spark.sql.autoBroadcastJoinThreshold','-1').\
#         config('spark.sql.broadcastTimeout','30000').\
#         appName('alpha recall dssm enbedding final yuliping09009').\
#         enableHiveSupport().getOrCreate()

# get tf config
config, _ = tf_context.get_tf_config(
    save_checkpoints_secs = 20*60,  # 每20分钟保存一次 checkpoints
    keep_checkpoint_max = 5      # 保留最新的10个checkpoints
)

logging.warning(config.model_dir)

usage_model = "DNN"

globle_columns = []
sparse_features = user_cate_cols + feed_cate_cols
dense_features = user_num_cols + feed_num_cols
feature_names = sparse_features + dense_features
us_emb = us_emb
vs_emb = vs_emb

def featuers_spec():
    """ Define feature column
    Args:
    """
    dense_features_spec = [ColumnSpec(column_name=x, dtype = 'float', is_label=False) for x in dense_features]

    sparse_features_spec = [ColumnSpec(column_name=x, dtype = 'int64', is_label=False) for x in sparse_features]

    label_feature_spec = [ColumnSpec(column_name='label_evehicle_mall_goodspicture', dtype='int64', is_label=True)]
    
    us_emb_feature_spec = [ColumnSpec(column_name=x, dtype = 'string', is_label=False) for x in us_emb]
    vs_emb_feature_spec = [ColumnSpec(column_name=x, dtype = 'string', is_label=False) for x in vs_emb]

    featuers_spec = dense_features_spec + sparse_features_spec + label_feature_spec + us_emb_feature_spec + vs_emb_feature_spec
#     featuers_spec = dense_features_spec + sparse_features_spec + label_feature_spec 
    
    return featuers_spec


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    """  Do concat
    Args:
    """
    sparse_dnn_input = Flatten()(tf.keras.layers.Concatenate()(sparse_embedding_list))
    dense_dnn_input = Flatten()(tf.keras.layers.Concatenate()(dense_value_list))
    return tf.keras.layers.Concatenate(axis=1)([sparse_dnn_input, dense_dnn_input])


def build_cross_layers(x0, num_layers):
    """  Cross NetWork
    Args:
    """
    x = x0
    for i in range(num_layers):
        x = cross_layer2(x0, x, 'cross_{}'.format(i))
    return x

def cross_layer2(x0, x, name):
    """  Cross NetWork
    Args:
    """
    with tf.variable_scope(name):
        input_dim = x0.get_shape().as_list()[1]
        # input_dim = 128
        w = tf.get_variable("weight", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
        xb = tf.tensordot(tf.reshape(x, [-1, 1, input_dim]), w, 1)
        return x0 * xb + b + x
    
def focal_loss(pred, y, alpha=0.25, gamma=2):
    zeros = tf.zeros_like(pred, dtype=pred.dtype)
    pos_p_sub = tf.where(y > zeros, y - pred, zeros) # positive sample 寻找正样本，并进行填充
    neg_p_sub = tf.where(y > zeros, zeros, pred) # negative sample 寻找负样本，并进行填充
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent)
    
def fc(inputs, units, name, mode, activation = "relu", is_bn = True, is_drop = True):
    """  全连接层
    Args:
    """
    if is_bn:
        hidden = tf.layers.dense(inputs=inputs, units=units, name= name, activation=None)
        batch_normed = tf.keras.layers.BatchNormalization()(hidden, training=True)
        output = tf.keras.activations.relu(batch_normed)
    else:
        if activation == "relu":     
            activation = tf.nn.relu
        else:
            activation = tf.nn.sigmoid
        
        ##Xavier normal initializer
        #output = tf.layers.dense(inputs=inputs, units=units, name=name,activation=tf.nn.relu, bias_initializer=tf.glorot_normal_initializer(), kernel_initializer=tf.glorot_normal_initializer())
        
        ##Kaiming初始化
        output = tf.layers.dense(inputs=inputs, units=units, name=name,activation=tf.nn.relu, bias_initializer=tf.initializers.he_uniform(), kernel_initializer=tf.initializers.he_uniform())
        #output = tf.layers.dense(inputs=inputs, units=units, name=name,activation=activation)
        if is_drop and mode == tf.estimator.ModeKeys.TRAIN:
            output = tf.nn.dropout(output, rate = 0.5)  
    return output


        
def meta(embeddings1,embeddings2,embeddings3):
    embeddings1 = tf.concat(embeddings1, 1)
    embeddings2 = tf.concat(embeddings2, 1)
#     embeddings3 = tf.concat(embeddings3, 1)
    emb = tf.concat([tf.concat([embeddings1, embeddings2], -1),embeddings3],1)

    ouput = tf.layers.dense(inputs=emb, units=64, name="meta_output")
    return ouput



def cmitn(features, labels, mode):
    """Build Deep & Cross NetworkField-wise Deep & Cross Netwo
    @inproceedings{wang2017deep,
      title={Deep \& cross network for ad click predictions},
      author={Wang, Ruoxi and Fu, Bin and Fu, Gang and Wang, Mingliang},
      booktitle={Proceedings of the ADKDD'17},
      pages={12},
      year={2017},
      organization={ACM}
    }
    Args:
    """
    user_denses = []
    user_embeddings = []
    x0_len = 0

    with open(f'/public/ludongyue580/process/evehicle/json_file/20220916_label_encoder_dict.json', 'r',encoding='utf8') as fp:
        le_dict = json.load(fp)
    label_rows = le_dict
    label_max_value_dict = {}
    for (k, v) in label_rows.items():
        max_value = 0
        for (kk, vv) in v.items():
            if vv >= max_value:
                max_value = vv
        label_max_value_dict[k] = max_value + 1
    print(label_max_value_dict)

    '''
        user塔 user_embedding是如何得到的
    '''
    
    for i, feat in enumerate(user_cate_cols):
        
        size = label_max_value_dict[feat]
        # emd_size = int(round(6 * (size ** 0.25)))
        emd_size = 128
        
        f = tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(feat, size), dimension=emd_size
            )
        
        x0_len += emd_size
        user_embeddings.append(tf.feature_column.input_layer(features, [f]))
        
        
    for fea in user_num_cols:
        f = tf.feature_column.numeric_column(fea)
        x0_len += 1
        user_denses.append(tf.feature_column.input_layer(features, [f]))

    
    
    '''
        item塔 item_embedding是如何得到的
    '''

    feed_denses = []
    feed_embeddings = []
    x0_len = 0
    
    for i, feat in enumerate(feed_cate_cols):
        
        size = label_max_value_dict[feat]
        # emd_size = int(round(6 * (size ** 0.25)))
        emd_size = 128
        
        f = tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(feat, size), dimension=emd_size
            )
        
        x0_len += emd_size
        feed_embeddings.append(tf.feature_column.input_layer(features, [f]))
        
        
    for fea in feed_num_cols:
        f = tf.feature_column.numeric_column(fea)
        x0_len += 1
        feed_denses.append(tf.feature_column.input_layer(features, [f]))
    
    with tf.variable_scope('user_model'):
        if usage_model == "DNN":
            # xdeepfm要保证emd_size是一样的
            user_net = DNN_scope(user_embeddings, user_denses)

    with tf.variable_scope('item_model'):
        if usage_model == "DNN":
            # xdeepfm要保证emd_size是一样的
            item_net = DNN_scope(feed_embeddings, feed_denses)
            

    '''
    源域向量怎么拿到
    '''
    us_embeddings = []
    for i, feat in enumerate(us_emb):
        us_embeddings.append(tf.feature_column.input_layer(features, [f]))
        
    vs_embeddings = []
    for i, feat in enumerate(vs_emb):
        vs_embeddings.append(tf.feature_column.input_layer(features, [f]))
        
    with tf.variable_scope('meta_model'):
        meta_ouput = meta(us_embeddings, vs_embeddings, item_net)
        

    
    logits = tf.reduce_sum(tf.multiply(meta_ouput, item_net), axis=1, keep_dims=True)
    y = tf.sigmoid(logits, name="y")

    threshold=0.5      #标签阈值设置
    one = tf.ones_like(y)            #生成与y大小一致的值全部为1的矩阵
    zero = tf.zeros_like(y)
    pred = tf.where(y < threshold, x=zero, y=one)  #数值小于0.5置0，大于0.5置1

    predictions = {
         "logits": y,
         "classes": pred
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
        'predict': pred,
        'user_embedding': user_net,
        'item_embedding': item_net
        }
        export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    loss1 = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels, logits))
    
    loss2 = tf.losses.mean_squared_error(user_net,meta_ouput)
    
    loss = loss1 + 0.05 * loss2
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def metric_fn(labels, predictions, logits):
        '''
         Add evaluation metrics (for EVAL mode)
        '''
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name='accuracy')
        auc = tf.metrics.auc(labels=labels, predictions=predictions["logits"], name='auc')
        recall = tf.metrics.recall(labels=labels, predictions=predictions["classes"], name='recall')
        recall_at_thresholds = tf.metrics.recall_at_thresholds(labels=labels, predictions=predictions["logits"], name='recall', thresholds = [i/10 for i in range(10)])
        precision = tf.metrics.precision(labels=labels, predictions=predictions["classes"],name='precision')
        return {
            'accuracy': accuracy,
            'auc': auc,
            'recall': recall,
            'precision': precision
        }

    eval_metric_ops = metric_fn(labels, predictions, y)

    print('----------------eval_metric_ops-----------------',flush=True)
    print('eval_metric_ops',eval_metric_ops)
    print('----------------eval_metric_opss-----------------',flush=True)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    spec = featuers_spec()

    train_dataset_builder = DatasetBuilder(input_table='turing_dev.cross_domain_recom_zuche_evehicle_sample', partitions=f'pt={date}', column_spec=spec)
    test_dataset_builder = DatasetBuilder(input_table='turing_dev.cross_domain_recom_zuche_evehicle_sample', partitions=f'pt={date}', column_spec=spec)


    classifier = tf.estimator.Estimator(
        model_fn=cmitn, model_dir=config.model_dir, config=config)
    
    #代表训练50步将会进行LOG输出，每个step开始/结束或者每个epoch开始/结束时需要执行某个操作
#     tensors_to_log={"probabilities": "y"，"logits":"logits"}
#     logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)

    train_input_fn = lambda: train_dataset_builder.input_fn(epoch=None)
    test_input_fn = lambda: test_dataset_builder.input_fn()

    label_name = 'label_evehicle_mall_goodspicture'
    def single_label_input_fn():
        features, labels = train_input_fn()
        print('-----------labels--------',labels)
        print('-----------labels name--------',labels[label_name])
        return features, labels[label_name]

    def single_label_input_test_fn():
        features, labels = test_input_fn()
        return features, labels[label_name]

    #TrainSpec确定训练的输入数据以及持续时间.可选的钩子(hook)在不同训练阶段运行.
    train_spec = tf.estimator.TrainSpec(
        single_label_input_fn, max_steps=20000
        #,hooks=[logging_hook]
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=single_label_input_test_fn,
        start_delay_secs=300,
        #,hooks=[logging_hook]
        #throttle_secs=900
    )
    results = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)[0]

    print('----------------results-----------------',flush=True)
    print('results',results)
    print('----------------results-----------------',flush=True)

    dict1 = {}
    dict1['business'] = 'local_life_m_homepage'
    dict1['accuracy'] =  format(results['accuracy'],'.5f') 
    dict1['auc'] = format(results['auc'],'.5f') 
    dict1['loss'] =  format(results['loss'],'.5f')  
    dict1['precision'] = format(results['precision'],'.5f')  
    dict1['recall'] = format(results['recall'],'.5f')  
    dict1['global_step'] = format(results['global_step'],'.5f') 
    dict1['pt'] = date

    import logging
    print("finish")
    logging.warning(config.model_dir)
    
    from estimator_utils1 import save_estimator_model
    save_estimator_model(classifier,train_dataset_builder)


    print('----------------dict1-----------------',flush=True)
    print('dict1',dict1)
    print('----------------dict1-----------------',flush=True)

    # df1 = spark.createDataFrame([dict1])

    # df1 = df1.withColumn('pt', F.lit(f'{date}'))
    param_utils.put_object_to_output_param('cdr_model_indicator_dict', dict1)
    

    return 0


if __name__ == "__main__":
    tf.app.run()





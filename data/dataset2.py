import numpy as np
import torch
try:
    from tqdm import tqdm
except:
    pass
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
import copy
import json
from data.feature import *
import pandas as pd
from aibrain_common.data.dataset_builder import (ColumnSpec, DatasetBuilder)
from aibrain_common.component import tools
from aibrain_common.utils.date_convert_utils import DateConvertUtils
date_converter = DateConvertUtils()
date_converter.set_biz_date("20220916")
date = date_converter.parse_data_date("${yyyymmdd}")
pre1day = date_converter.parse_data_date("${yyyymmdd - 1}")


class dataset_meta_conv(Dataset):
    """
    data set for offline train ,and  prepare for dataloader
    """
    def __init__(self, s_usr, s_item, t_usr, t_item, labels):
        super(dataset_meta_conv, self).__init__()
        self.s_user = np.array(s_usr)
        self.s_item = np.array(s_item)
        self.t_user = np.array(t_usr)
        self.t_item = np.array(t_item)
        self.labels = np.array(labels)
#         self.transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return  self.s_user.shape[0]
    def __getitem__(self, idx):
        s_user = self.s_user[idx]
        s_item = self.s_item[idx]
        t_user = self.t_user[idx]
        t_item = self.t_item[idx]
        label = self.labels[idx]
        return s_user.astype(np.float), s_item.astype(np.float), t_user.astype(np.float), t_item.astype(np.float), label.astype(np.int32)

class get_DataLoader():
    def __init__(self, args):
        self.args = args
        
        self.dataset = 0

        self.globle_columns = []
        self.t_user_cate_cols = user_cate_cols
        self.t_feed_cate_cols = feed_cate_cols
        self.t_user_num_cols = user_num_cols
        self.t_feed_num_cols = feed_num_cols
        self.sparse_features = user_cate_cols + feed_cate_cols
        self.dense_features = user_num_cols + feed_num_cols
        self.feature_names = self.sparse_features + self.dense_features
        self.us_emb = us_emb
        self.vs_emb = vs_emb
        self.batch_size = args.bridge_batch_size
        self.numworkers = args.numworkers
        self.input_json = args.input_json
#         self.dataset_conv_meta = dataset_meta_conv
        spec = self.featuers_spec()
        self.train_dataset_builder = DatasetBuilder(input_table=self.args.input_table_train,\
         partitions=f'pt={self.args.pt}', column_spec=spec).to_pandas()
        self.test_dataset_builder = DatasetBuilder(input_table=self.args.input_table_test, \
            partitions=f'pt={self.args.pt}', column_spec=spec).to_pandas()
        
    
    def cat_embeding_prepare(self, feature_df):
        
        with open(self.input_json, 'r',encoding='utf8') as fp:
            le_dict = json.load(fp)
        label_rows = le_dict
        label_max_value_dict = {}
        for (k, v) in label_rows.items():
            max_value = 0
            for (kk, vv) in v.items():
                if k in feature_df.columns.tolist():
                    feature_df[feature_df[k]==kk] = vv
                if vv >= max_value:  
                    max_value = vv
            label_max_value_dict[k] = max_value + 1
        for col in feature_df.columns.tolist():
            size_ =  label_max_value_dict[col]
            embedding_layer = nn.Embedding(size_, 10)
            embed_col = embedding_layer(torch.tensor(feature_df[col], dtype=torch.long)).view(feature_df.shape[0], 10)
            feature_df.drop(columns=[col])
            feature_df = pd.concat([feature_df, pd.DataFrame(embed_col.detach().numpy())], axis=1)

        return feature_df
    def source_emb_prepare(self, embed_need_prepare):
        embed_need_prepare = pd.DataFrame(embed_need_prepare)
        index_name_orig = embed_need_prepare.columns.values.tolist()
        for name_orig in index_name_orig:
            embed_temp = copy.deepcopy(embed_need_prepare)
            embed_temp_ = embed_temp[embed_temp[name_orig].notna()].reset_index(drop=True)
            len_str = len(embed_temp_[name_orig][0].split(','))
            null_fit_str = ','.join(['-1.0' for _ in range(len_str)])
            values = {name_orig:null_fit_str}
            embed_need_prepare.fillna(value=values)
            embed_need_prepare[name_orig].replace(to_replace={None:null_fit_str}, inplace=True)
            for i in range(len_str):
                embed_need_prepare[name_orig+str(i)] = embed_need_prepare[name_orig].apply(lambda x:x.split(',')[i])
            embed_need_prepare.drop(columns=name_orig, inplace=True)
        return embed_need_prepare
    def get_each_dataset(self, mode=None):
        if mode == 'support':
            train_dataset_builder_ = self.train_dataset_builder.sample(n=10000, replace=True, random_state=18).reset_index(drop=True)
            s_usr_embed_df = train_dataset_builder_[self.us_emb]
            s_usr_embed_df = self.source_emb_prepare(s_usr_embed_df)
            s_item_embed_df = train_dataset_builder_[self.vs_emb]
            s_item_embed_df = self.source_emb_prepare(s_item_embed_df)
            t_usr_cat_df = train_dataset_builder_[self.t_user_cate_cols]
            t_usr_num_df = train_dataset_builder_[self.t_user_num_cols]
            t_item_cat_df = train_dataset_builder_[self.t_feed_cate_cols]
            t_item_num_df = train_dataset_builder_[self.t_feed_num_cols]
            label_df = train_dataset_builder_['label_evehicle_mall_goodspicture'].rename('labels', inplace=True)
            t_usr_cat_df_prep = self.cat_embeding_prepare(t_usr_cat_df)
            t_item_cat_df_prep = self.cat_embeding_prepare(t_item_cat_df)
            t_usr_df = pd.concat([t_usr_cat_df_prep, t_usr_num_df], axis=1)
            t_item_df = pd.concat([t_item_cat_df_prep, t_item_num_df], axis=1)
            dataset_conv_meta = dataset_meta_conv(s_usr_embed_df, s_item_embed_df, t_usr_df, t_item_df, label_df)
            del train_dataset_builder_, s_usr_embed_df, s_item_embed_df, t_usr_cat_df, t_usr_num_df, t_item_cat_df, t_item_num_df, label_df, t_usr_cat_df_prep, t_usr_df, t_item_df
            return dataset_conv_meta
        elif mode == 'query':
            test_dataset_builder_ = self.test_dataset_builder.sample(n=10000, replace=True, random_state=18).reset_index(drop=True)
            s_usr_embed_df = test_dataset_builder_[self.us_emb]
            s_usr_embed_df = self.source_emb_prepare(s_usr_embed_df)
            s_item_embed_df = test_dataset_builder_[self.vs_emb]
            s_item_embed_df = self.source_emb_prepare(s_item_embed_df)
            t_usr_cat_df = test_dataset_builder_[self.t_user_cate_cols]
            t_usr_num_df = test_dataset_builder_[self.t_user_num_cols]
            t_item_cat_df = test_dataset_builder_[self.t_feed_cate_cols]
            t_item_num_df = test_dataset_builder_[self.t_feed_num_cols]
            label_df = test_dataset_builder_['label_evehicle_mall_goodspicture'].rename('labels', inplace=True)
            t_usr_cat_df_prep = self.cat_embeding_prepare(t_usr_cat_df)
            t_item_cat_df_prep = self.cat_embeding_prepare(t_item_cat_df)
            t_usr_df = pd.concat([t_usr_cat_df_prep, t_usr_num_df], axis=1)
            t_item_df = pd.concat([t_item_cat_df_prep, t_item_num_df], axis=1)
            dataset_conv_meta = dataset_meta_conv(s_usr_embed_df, s_item_embed_df, t_usr_df, t_item_df, label_df)
            del test_dataset_builder_, s_usr_embed_df, s_item_embed_df, t_usr_cat_df, t_usr_num_df, t_item_cat_df, t_item_num_df, label_df, t_usr_cat_df_prep, t_usr_df, t_item_df
            return dataset_conv_meta
        else:
            return None
    def featuers_spec(self):
        """ Define feature column
        Args:
        """
        dense_features_spec = [ColumnSpec(column_name=x, dtype = 'float', is_label=False) for x in self.dense_features]

        sparse_features_spec = [ColumnSpec(column_name=x, dtype = 'int64', is_label=False) for x in self.sparse_features]

        label_feature_spec = [ColumnSpec(column_name='label_evehicle_mall_goodspicture', dtype='int64', is_label=True)]
        
        us_emb_feature_spec = [ColumnSpec(column_name=x, dtype = 'string', is_label=False) for x in self.us_emb]
        vs_emb_feature_spec = [ColumnSpec(column_name=x, dtype = 'string', is_label=False) for x in self.vs_emb]

        featuers_spec = dense_features_spec + sparse_features_spec + label_feature_spec + us_emb_feature_spec + vs_emb_feature_spec
    #     featuers_spec = dense_features_spec + sparse_features_spec + label_feature_spec 
    
        return featuers_spec
    def get_dataBulider(self):
        dataset = 0
        return dataset
    def get_dataLoader(self):
        train_all_dataset = self.get_each_dataset(mode='support')
        support_dataloader = torch.utils.data.DataLoader(train_all_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.numworkers,
                                    pin_memory=False,
                                    drop_last=True)
        del train_all_dataset
        test_all_dataset = self.get_each_dataset(mode='query')
        query_dataloader = torch.utils.data.DataLoader(test_all_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.numworkers,
                                    pin_memory=False,
                                    drop_last=True)
        del test_all_dataset
        return support_dataloader, query_dataloader
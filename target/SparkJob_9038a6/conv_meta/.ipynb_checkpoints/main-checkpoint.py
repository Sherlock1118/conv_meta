import argparse
import torch
import os
import numpy as np
from data import dataset2
#from model import transfer,tranfer_pre1
# from model import meta as transfer
from model import meta_update as transfer
#from model import multi_transfer as transfer

def get_parse():
    parser = argparse.ArgumentParser(description='MF and TR(transfer) parameters in our SML.')

    parser.add_argument('--data_name', default="yelp",  #
                        help='dataset name: yelp or news (i.e Adressa)')
    parser.add_argument('--data_path', default='/home/sml/dataset/',  #
                        help='dataset path')

    '''
    **********************************************************************************************
    SML parameters:
    outer loop number, stop condition for SML.
    **********************************************************************************************
    '''
    parser.add_argument('--multi_num', type=int, default=20,  # outer loop
                        help='Number of epochs to train MF and Transfer,outer loop for SML.')
    
    parser.add_argument('--s_item_dimention', type=int, default=320,  # outer loop
                        help='Number of epochs to train MF and Transfer,outer loop for SML.')
    parser.add_argument('--t_usr_dimention', type=int, default=815,  # outer loop
                        help='Number of epochs to train MF and Transfer,outer loop for SML.')
    parser.add_argument('--t_item_dimention', type=int, default=212,  # outer loop
                        help='Number of epochs to train MF and Transfer,outer loop for SML.')
#     parser.add_argument('--input_table_train', type=str, default='turing_dev.cross_domain_recom_zuche_evehicle_sample_train',  # outer loop
#                         help='Number of epochs to train MF and Transfer,outer loop for SML.')
#     parser.add_argument('--input_table_test', type=str, default='turing_dev.cross_domain_recom_zuche_evehicle_sample_test',  # outer loop
#                         help='Number of epochs to train MF and Transfer,outer loop for SML.')
    parser.add_argument('--input_table_train', type=str, default='turing_dev.cross_domain_recom_zuche_evehicle_sample_train_tmp',  # outer loop
                        help='Number of epochs to train MF and Transfer,outer loop for SML.')
    parser.add_argument('--input_table_test', type=str, default='turing_dev.cross_domain_recom_zuche_evehicle_sample_test_tmp',  # outer loop
                        help='Number of epochs to train MF and Transfer,outer loop for SML.')
    parser.add_argument('--input_json', type=str, default='/home/jovyan/conv_meta/conv_meta/data/20220916_label_encoder_dict.json',  # outer loop
                        help='Number of epochs to train MF and Transfer,outer loop for SML.')
    parser.add_argument('--pt', type=str, default='20220916',  # outer loop
                        help='Number of epochs to train MF and Transfer,outer loop for SML.')

    '''
    **********************************************************************************************
    SML parameters:
    parameters for step 1 of SML, i.e, parameters for MF training. 
    In the final version, only some parameters are used.
    **********************************************************************************************
    '''
    parser.add_argument('--bridge_lr', type=float, default=0.01,
                        help='Learning rate for MF learning, step 1 of SML.')
    parser.add_argument('--bridge_epochs', type=int, default=1,  # 1 or 2
                        help='Number of epochs to train MF, step 1 of SML.')
    parser.add_argument('--l2', type=float, default=1e-6,    # 1e-6
                        help='L2 norm for loss')
    
    parser.add_argument('--bridge_batch_size', type=int, default=1024,
                        help='batch size for MF training, step in SML.')
    parser.add_argument('--bridge_input_dim', type=int, default=512,
                        help='batch size for MF training, step in SML.')
    parser.add_argument('--bridge_output_dim', type=int, default=512,
                        help='batch size for MF training, step in SML.')
    parser.add_argument('--laten', type=int, default=512,    # dim of embedding
                        help='dim of laten factor.')
    parser.add_argument('--pre_model', default="/home/sml/save_model/sml/yelp/BCE_init.pkl",
                        help='pre train model path.')        # pretrain MF model path
    parser.add_argument('--bridge_sample', default="all",        # MF sample type, "all" we used
                        help='MF sample type, all or alone')


    parser.add_argument('--Load_W_hat', default=False,       # see code to understand, default is we used
                        help='load MF w_hat after transfer learning!!')
    parser.add_argument('--clip_grad', default=False,        # not used in the final version
                        help='if clip_grad !!')
    parser.add_argument('--need_adaptive', default=False,    # not used in the final version
                        help='if need adaptive norm !!')
    parser.add_argument('--maxnorm_grad', type=float,default=3.0, # not used in the final version
                        help='max grad on norm !!')
    parser.add_argument('--loss1_', type=float,default=0.05, # not used in the final version
                        help='max grad on norm !!')
    parser.add_argument('--loss2_', type=float,default=1.0, # not used in the final version
                        help='max grad on norm !!')

    """
    ******************************************************************************************
    SML parameters:
    parameters for step 2 of SML, i.e, parameters for transfer training. 
    In the final version, only some parameters are used.
    ******************************************************************************************
    """

    parser.add_argument('--TR_lr', type=float, default=0.001, # 1e-2
                        help='transfer Learning rate.')
    parser.add_argument('--TR_l2', type=float, default=0.0001, # 1e-4
                        help='transfer reg.')
    parser.add_argument('--TR_epochs', type=int, default=1,    # 1 or 2
                        help='Number of epochs to train transfer.')
    parser.add_argument('--TR_batch_size', type=int, default=256,
                        help='batch size for transfer.')
    parser.add_argument('--TR_sample_type', default="alone",
                        help='how to sample negative sample of TR train.[all , alone].all: is to select all,alone: select only this stage')
    parser.add_argument('--TR_with_MF_bias', type=bool, default=False,
                        help='if input MF bias into MF model.')
    parser.add_argument('--TR_stop_', type=bool, default=False,
                        help='if stop TR train when online test stage. not same to SML-S')
    parser.add_argument('--transfer_type', default="conv_com",   # transfer type,  conv_com is same to our paper
                        help='transfer type "conv_com"')


    '''
    ******************************************************************************************
    other parameters, some are not used in our final version!!
    
    ******************************************************************************************
    '''
    parser.add_argument('--seed', type=int, default=2000,
                        help='random seed')
    parser.add_argument('--numworkers', type=int, default=4,
                        help='numberworker ,used for loader data ?.default 4')
    parser.add_argument('--cuda', type=str, default='1',
                        help='which GPU be used?.default 1')
    parser.add_argument('--topK', type=int, default=20,  #
                        help='topK ? not usful for final test,only useful for training process !')

    '''
    not used in this version, but may help to improve the model performance !!
    '''
    parser.add_argument('--pass_num', type=int, default=1,         # not used in final version, but may be helpful
                        help='the pass number to rerun the offline training for transfer(may be help improving), default: 1, not used in final version')
    parser.add_argument('--norm', type=bool, default=False,        # not used !!
                        help='An normalized tech. An extra tech that we developed may be help to improve performance, we not used in final version!!')
    parser.add_argument('--Lambda_lr', type=float, default=0.01,   # not used!!
                        help='init lr for adaptive learning lr, Not used!!!')
    parser.add_argument('--min_l2', type=float, default=0.0001,     # not used!!
                        help='Not used!!')
    parser.add_argument('--set_t_as_tt', type=bool, default=False,  # be default is ok
                        help='Not used!.')
    parser.add_argument('--tqdm', type=bool, default=False,         # not used
                        help='tqdm.')
    parser.add_argument('--need_writer', type=bool, default=False,
                        help='need summary write? ')
    parser.add_argument('--test_in_TR_Train', type=bool, default=False, # not used!!
                        help='if need test when in trabsfer training,not used!')

    return parser

if __name__=="__main__":
    parser = get_parse()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    TR_l2 = [0.0001]  # not used
    multi_num = [10] # not used
    MF_l2 = [1e-6]   # not used
    recall_all = []
    ndcg_all = []
    for p in MF_l2:
        for m_num in multi_num:       
            print("###(multi num,l2)",m_num,p)
            # args.MF_epochs = 1
            # args.l2 = p
            # args.multi_num = m_num
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed+1)
            np.random.seed(args.seed+2)
            # cudnn随机种子固定
            torch.backends.cudnn.deterministic = True
            # cudnn.benchmark = True

            print("*********************  parameters information ****************************************")
            print(args)
            print("stop:", args.TR_stop_)
            if args.Load_W_hat:
              print("load w_hat:",args.Load_W_hat)
            print("TR sample type",args.TR_sample_type)
            # print("MF sample type:",args.MF_sample)
            print("MF: lr:{},l2:{},batch_size:{},laten:{},epoch:{}".format(args.bridge_lr, args.l2, args.bridge_batch_size, args.laten,
                                                                     args.bridge_epochs))
            print("TR: lr:{},l2:{},batch_size:{},epoch:{}".format(args.TR_lr, args.TR_l2, args.TR_batch_size, args.TR_epochs))
            print("top k:",args.topK)
            print("*********************  parameters information ****************************************\n")

#             data_path = args.data_path                      # /home/wcx/onlineMF_yelp/dataset/
#             data_name = args.data_name                      #  yelp

#             file_list = [str(i) for i in range(0, 40)]
#             test_list = [str(j) for j in range(30,40)]
#             validation_list = [str(j) for j in range(10,20)]


#             DataSets = dataset2.transfer_data(args,path=data_path, datasetname=data_name, file_path_list=file_list, test_list=test_list,
#                                      validation_list=None, online_train_time=round(10),online_test_time=round(30))
            #MetaModel = transfer.multi_train(args, DataSets, DataSets.user_number, DataSets.item_number, args.laten)
            MetaModel = transfer.meta_train(args, args.bridge_input_dim, args.bridge_output_dim, args.laten)
            MetaModel.run(args)

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@TR:L2:",p)
            print("##")
            print("##")
            print("\n")
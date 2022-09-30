# import model.MF as MF 
from formatter import NullWriter
from logging.handlers import BufferingHandler
import torch
import torch.nn as nn
from data.dataset2 import dataset_meta_conv
# from data.dataset import offlineDataset_withsample as SampleDaset
# from data.dataset2 import trainDataset_withPreSample as PreSampleDatast
# from data.dataset2 import testDataset
import torch.nn.functional as F
import torch.utils.data
#from model.train_model import train_one_epoch
from evalution.evaluation2 import test_model
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np
import time
from model.meta_bridge_transfer import ConvTransfer_com, bridgeModel
from data.dataset2 import get_DataLoader



class Gelu(nn.Module):
    def __init__(self):
        super(Gelu, self).__init__()
    def forward(self,x):
        return x*torch.sigmoid(1.702*x)

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hid_dim,dropout):
        super(MLP, self).__init__()
        model_list = nn.ModuleList()
        input_dim = in_dim
        for i in range(len(hid_dim)):
            model_list.append(nn.Linear(input_dim,hid_dim[i]))
            model_list.append(nn.Tanh())
            input_dim = hid_dim[i]
        model_list.append(nn.Dropout(p=dropout))
        model_list.append(nn.Linear(input_dim,out_dim))
        self.model_ = nn.Sequential(*model_list)
    def forward(self, x):
        return self.model_(x)

class bridgeModel(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(bridgeModel, self).__init__()
        self.layer = nn.Linear(input_dim,out_dim)
        
    def reset_parameters(self):
        self.layer.reset_parameters()
    def set_parameters(self,bridge_weight):
        self.layer.weight.data.copy_(bridge_weight[:,0:-1])
    def forword(self, x):
        return self.layer(x)

"""
meta Transfer model !!
"""
class meta_train():
    '''
    meta Transfer model !!!
    '''
    def __init__(self,args,input_dim_bridge, out_dim_bridge, laten_dim):
        '''
        :param args:
        :param datasets:
        :param user_num: user number
        :param item_num: item number
        :param laten_dim: embedding dim
        '''
        # bridge model
#         self.bridge_base = bridgeModel(input_dim_bridge, out_dim_bridge)
        self.bridge_base = nn.DataParallel(bridgeModel(input_dim_bridge, out_dim_bridge)).cuda()
        self.with_MF_bias = args.TR_with_MF_bias
        self.test_in_TR_train = args.test_in_TR_Train
        self.TR_train_sampleTYpe = args.TR_sample_type
        self.get_DataLoader = get_DataLoader(args)
        self.need_writer = args.need_writer
        #todo



        if args.need_writer:
            path = "m-num"+str(args.multi_num)+"-Bridge-lr"+str(args.bridge_lr)+"-l2-"+str(args.l2)+"e-"+str(args.bridge_epochs)+"--TR-lr"+str(args.TR_lr)+"-l2-"+str(args.TR_l2)+"-e-"+str(args.TR_epochs)+str(args.TR_sample_type)+"user-norm"+str(args.norm)
            self.writer = SummaryWriter(comment=path)

        self.bridge_weight = self.bridge_base.layer.weight.data
        if self.with_MF_bias: # input the bridge bais into transfer!
            self.last_bridge_weight = self.bridge_base.layer.weight.data
            self.bridge_weight_hat = self.bridge_base.layer.weight.data
            self.last_bridge_weight_hat = self.bridge_base.layer.weight.data
            
            input_dim_bridge = laten_dim + 1
        else:                # don't input the bridge bais into transfer!
            input_dim_bridge = laten_dim

            self.last_bridge_weight = torch.zeros_like(self.bridge_base.layer.weight.data)
            self.item_embd = torch.zeros_like(torch.empty(args.bridge_batch_size, 1, input_dim_bridge))
            self.bridge_weight_hat = copy.deepcopy(self.bridge_base.layer.weight.data)
            self.last_bridge_weight_hat = copy.deepcopy(self.bridge_weight_hat)
            

        ''' creat meta transfer model'''
        self.transfer = nn.DataParallel(ConvTransfer_com(args, input_dim_bridge, input_dim_bridge,  args.s_item_dimention, args.t_usr_dimention, args.t_item_dimention)).cuda()
#         self.transfer = ConvTransfer_com(args, input_dim_bridge, input_dim_bridge,  args.s_item_dimention, args.t_usr_dimention, args.t_item_dimention)
#         self.dataset = datasets

        '''optimizer for two parts  MFbase and transfer!'''
        self.bridge_optimizer = torch.optim.Adam(self.bridge_base.parameters(), lr=args.bridge_lr, weight_decay=0)
        self.transfer_optimizer = torch.optim.Adam(self.transfer.parameters(), lr=args.TR_lr, weight_decay=args.TR_l2) #1

        '''save model performance'''
        self.recall = []
        self.ndcg = []
        self.test_num = []



    def get_next_data(self,stage_id):
        '''
        get model next data. For more information ,please read the method of corresponding dataset class
        :param stage_id: stage idx
        :return: set_t,set_tt,now_test
        '''
        set_t,set_tt,now_test,val = self.dataset.next_train(stage_id)
        return set_t,set_tt,now_test,val


    def bridge_train_onestage(self,args,surpport_set,stage_id,val=None): # this right for main
        
    
        
        self.transfer.eval()
        if val is not None:
            val = torch.utils.data.DataLoader(val,
                                                   batch_size=1024,
                                                   num_workers=args.numworkers,
                                                   pin_memory=False
                                                   )
            #todo
            recall, ndcg = test_model(self.bridge_base, val, topK=args.topK)
            print("before train bridge test:recall:{:.4f} ndcg:{:.4f}".format(recall, ndcg))

        print("******bridge (inner) training ******")
        #dataset准备，待更新代码

        surp_set_dataloader = surpport_set
            
        for epoch in range(args.bridge_epochs):
            self.bridge_base.train()
            self.transfer.eval()

            loss_all = 0
            for batch_id,(source_user_embd, source_item_embd, t_user, t_item, labels) in enumerate(surp_set_dataloader):
                self.bridge_base.zero_grad()
                self.transfer.zero_grad()
#                 source_user_embd_cuda = source_user_embd.float()
#                 source_item_embd_cuda = source_item_embd.float()
#                 target_user_cuda = t_user.float()
#                 target_item_cuda = t_item.float()
#                 labels = labels.long()
                source_user_embd_cuda = source_user_embd.cuda().float()
                source_item_embd_cuda = source_item_embd.cuda().float()
                target_user_cuda = t_user.cuda().float()
                target_item_cuda = t_item.cuda().float()
                labels = labels.cuda().long()
                self.item_embd = source_item_embd_cuda
                last_weight_bridge = self.last_bridge_weight
                bridge_weight_hat = self.bridge_base.layer.weight.data


                loss_batch = self.transfer.run_bridge(args, copy.deepcopy(self.bridge_base), last_weight_bridge,
                                                  bridge_weight_hat,
                                                  source_user_embd_cuda,
                                                  source_item_embd_cuda,
                                                  target_user_cuda,
                                                  target_item_cuda,
                                                  labels)


                l2loss = 0.5 * torch.sum(last_weight_bridge**2 + bridge_weight_hat**2)

                loss_batch = loss_batch + args.l2 * l2loss

                loss_all += loss_batch.data
                loss_batch.backward()

                self.bridge_optimizer.step()
                #print(self.MFbase.user_laten.weight.grad)

            loss_all = loss_all/(batch_id + 1)
            loss_all = loss_all.item() / args.MF_batch_size

            if val is not None:
                recall, ndcg = test_model(self.MFbase, val, topK=args.topK)
                print("bridge-stage:",stage_id,"epoch:",epoch,"loss:{:.5f}".format(loss_all))
                # ,"recall:{:.4f}".format(recall),"ndcg:{:.4f}".format(ndcg)

            else:
                print("bridge-stage:",stage_id,"epoch:",epoch,"loss:",loss_all)
                if self.need_writer:
                    self.writer.add_scalar("Loss/bridge-loss",self.MF_itr)


    def transfer_train_onestage(self,args,query_set,stage_id,compute_performance=False,val=None):
        """
        for better stop , we can sample a set_tt_test beased on set_tt, similar to online test traing set!!!!
        noticed: (1) because only user/item in set_tt(including negative items) will supervised!!  so ,we only need
                 put these weight into transfer ,in training stage.
                 (2) when, get weight_t used for next, stage, that all weight need be computed ! how about the new
                 user/item weights??
        :param args: some hyper_paraments
        :param set_tt: next time set
        :param stage_id: equal to time_stage
        :return:
        """
        print("********* this is Transfer model training stage ***********")
        self.bridge_base.eval()  # don't compute MFbase gradient
        ''' what's sample type Dataset wused! '''
        if val is not None:
            now_test = testDataset(val)
            now_test = torch.utils.data.DataLoader(now_test,
                                                    batch_size=1024,
                                                    num_workers=args.numworkers,
                                                    pin_memory=False
                                                    )

            recall, ndcg = test_model(self.bridge_base, now_test, topK=args.topK)
            print("before train transfer test:recall:{:.4f} ndcg:{:.4f}".format(recall,ndcg))
            if self.need_writer:
                self.writer.add_scalar("Acc/tr-TR-recall@" + str(args.topK), recall, self.TR_itr)
                self.writer.add_scalar("Acc/tr-TR-ndcg@" + str(args.topK), ndcg, self.TR_itr)
                self.TR_itr += 1

        s_time = time.time()
        for epoch in range(args.TR_epochs):
            self.transfer.train()
            loss_all = 0
            for batch_id,(source_user_embd, source_item_embd, t_user, t_item, labels) in enumerate(query_set):
                self.transfer.zero_grad()
#                 source_user_embd_cuda = source_user_embd.float()
#                 source_item_embd_cuda = source_item_embd.float()
#                 target_user_cuda = t_user.float()
#                 target_item_cuda = t_item.float()
#                 labels = labels.long()
                source_user_embd_cuda = source_user_embd.cuda().float()
                source_item_embd_cuda = source_item_embd.cuda().float()
                target_user_cuda = t_user.cuda().float()
                target_item_cuda = t_item.cuda().float()
                labels = labels.cuda().long()
                self.item_embd = source_item_embd_cuda
                bridge_weight_hat = self.bridge_weight_hat


                loss_batch = self.transfer.run_bridge(args, copy.deepcopy(self.bridge_base), last_weight_bridge,
                                                  bridge_weight_hat,
                                                  source_user_embd_cuda,
                                                  source_item_embd_cuda,
                                                  target_user_cuda,
                                                  target_item_cuda,
                                                  labels,
                                                  mode='transfer')

                loss_all += loss_batch.data
                loss_batch.backward()
                # if clip:
                #     #torch.nn.utils.clip_grad_value_(self.transfer.parameters(),2)
                #     max_norm = args.maxnorm_grad
                #     torch.nn.utils.clip_grad_norm_(self.transfer.parameters(), max_norm,norm_type=2)
                self.transfer_optimizer.step()
            loss_all = loss_all/(batch_id+1)

            print("one epcohs TR time cost:",time.time()-s_time)
            """
            if need test, we can add code in here to test the real performance in here
            we use now test transfer performance
            """
            if self.need_writer:
                self.writer.add_scalar("Loss/TR-loss",loss_all.item() / args.TR_batch_size,self.TR_itr)
            if compute_performance:
                self.updata()
                recall, ndcg = test_model(self.MFbase, now_test, topK=args.topK)
                print("stage:{}, epcoh:{},loss:{:.4f},*****val result  reacll:{:.4f}  ndcg:{:.4f}".format(stage_id,epoch,loss_all.item() / args.TR_batch_size,recall, ndcg))
                #self.load_MFbase_weight(self.user_weight_hat,self.item_weight_hat)  # recover the weight to train transfer
                if self.need_writer:
                    itrs = stage_id * args.TR_epochs + epoch
                    self.writer.add_scalar("Acc/tr-TR-recall@"+str(args.topK), recall,self.TR_itr)
                    self.writer.add_scalar("Acc/tr-TR-ndcg@"+str(args.topK), ndcg, self.TR_itr)
            else:
                print("stage:", stage_id, "epoch:", epoch, "transfer train loss:", loss_all.item() / args.TR_batch_size)
        print("stage ",stage_id," transfer trained finished!!!!")

    def train_one_stage3(self,args,stage_id):
        """
        run meta transfer for one stage (period) !
        :param args: hyper-parameters
        :param stage_id: stage idx
        :return: bool  True: sucess  False: without data available.
        """

        ''' save bridge_base  last weight'''
        #bridge权重Wt存为Wt-1
        self.save_bridge_weight(save_as='last')              


        #todo overlapping随机分组，多次训练
        dataset = 0
        # train phase
        for phase in range(args.multi_num): # outer loop
            support_set, query_set = self.get_DataLoader.get_dataLoader()
            #内层更新
            self.bridge_train_onestage(args, support_set, stage_id,val=None)
            self.bridge_base.eval()
            ''' save bridge_base weight'''
            # Wt_hat-->Wt-1_hat, Wt-->Wt_hat
            self.save_bridge_weight(save_as='hat') # save as W_hat
            # 更新Wt
            self.updata(self.item_embd)

            if self.need_writer:
                self.writer.add_scalars("Scale/bridge_weight",{"weight_hat":torch.norm(self.bridge_weight_hat),
                                                                "weight_last":torch.norm(self.last_bridge_weight)},stage_id)
                
            '''train transfer'''
            #外层更新
            self.transfer_train_onestage(args,query_set,stage_id,val=None)
        '''updata model'''
        #更新Wt
        self.updata(self.item_embd)
        

        # # test phase
        # s_time = time.time()
        # # todo
        # support_set, query_set = self.get_DataLoader.get_dataLoader()
        # # todo
        # # print metrics
        # print("only traning time cost:",time.time()-s_time)
        # recall, ndcg = test_model(self.bridge_base, now_test, topK=20)
        # print("test result --------- reacll:{:.4f}  ndcg:{:.4f}".format(recall, ndcg))
        # self.recall.append(recall)
        # self.ndcg.append(ndcg.cpu().numpy())

        return True
        


    def updata(self, item_embd):
        '''
        we will update MFbase model weights
        :return: None
        '''
        self.bridge_base.eval()
        self.transfer.eval()
        new_bridge_weight = self.transfer(self.last_bridge_weight,self.bridge_weight_hat,item_embd)

        self.load_bridge_weight(new_bridge_weight)       


    def save_bridge_weight(self, save_as="last"):
        '''
        save MFbase weights,usually it's used when the stage strating and when MFbase have trained over.
        if the save type is wrong.will raise type error
        :param save_as: "last":save as weight_last  "hat":save as weight_hat
        :return: None
        '''
        if self.with_MF_bias:
            if save_as == "hat":
                a = torch.cat([self.MFbase.user_laten.weight.data, self.MFbase.user_bais.weight.data], dim=-1)
                self.user_weight_hat.copy_(a)
                b = torch.cat([self.MFbase.item_laten.weight.data, self.MFbase.item_bais.weight.data], dim=-1)
                self.item_weight_hat.copy_(b)
        else:
            if save_as == "last":
                self.last_bridge_weight.copy_(self.bridge_base.layer.weight.data)

            elif save_as == "hat":
                self.last_bridge_weight_hat.copy_(self.bridge_weight_hat.data)
                self.bridge_weight_hat.copy_(self.bridge_base.layer.weight.data)
            else:
                raise TypeError("save MFbase weight type is wrong")

    def load_bridge_weight(self, bridge_weight):
        '''
        set MFbase model weights as the input.
        :param user_weight: MF user weights
        :param item_weight: MF item weight
        :return: None
        '''
        if self.with_MF_bias:
            self.bridge_base.layer.weight.data.copy_(bridge_weight[:, 0:-1])
        else:
            self.bridge_base.layer.weight.data.copy_(bridge_weight)

    def run(self,args):
        '''
        run the full model, will print the model performance
        :param args: hyper-parameters
        :return: None
        '''

        pass_num = args.pass_num

        for pass_id in range(pass_num):
            stage_id = 0
#             self.dataset.reinit()
                #np.save(str(stage_id)+"_weight.npy", self.transfer.user_transfer.weight.data.cpu().numpy())
            self.train_one_stage3(args, stage_id)

            # todo 打印metrics
            print(str(pass_id)+"--trained over!!!!!")
#             test_num = np.array(self.test_num)
#             N3 = round(test_num.shape[0]*1/3)
#             val_num=test_num[0:N3]
#             test_num=test_num[N3:-1]
#             recall = np.array(self.recall)
#             ndcg = np.array(self.ndcg)

#             print(test_num)
#             print(recall)
#             print(ndcg)
#             print("include stage 0 of test:")
#             val_num=val_num*1.0/val_num.sum()
#             test_num = test_num*1.0 / test_num.sum()
#             #test_num = test_num.reshape(-1,1)

#             print("val average recall@20:",(recall[0:N3]*val_num).sum())
#             print("val average ndcg@20:", (ndcg[0:N3]*val_num).sum())
#             print("test average recall@20:", (recall[N3:-1] * test_num).sum())
#             print("test average ndcg@20:", (ndcg[N3:-1] * test_num).sum())



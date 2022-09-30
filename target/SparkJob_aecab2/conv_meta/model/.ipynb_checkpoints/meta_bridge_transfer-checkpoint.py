import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import classification_report

def Gelu(x):
    return x*torch.sigmoid(1.702*x)

def BCEloss(item_score,negitem_score):
    pos_loss = -torch.mean(torch.log(torch.sigmoid(item_score)+1e-15))
    neg_loss = -torch.mean(torch.log(1-torch.sigmoid(negitem_score)+1e-15))
    loss = pos_loss + neg_loss
    return loss

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
    #todo
    def test(self, topK=20):
        metrics = 0
        return metrics

class one_transfer(nn.Module):
    '''
    one transfer that contain two cnn layers
    '''

    def __init__(self,input_dim,out_dim, kernel=2):
        super(one_transfer, self).__init__()
        self.hidden_dim = input_dim
        self.out_channel = 10
        self.conv1 = nn.Conv2d(1,self.out_channel,(kernel,1),stride=1)

        self.out_channel2 = 5
        self.conv2 = nn.Conv2d(self.out_channel,self.out_channel2,(1,1),stride=1)


        self.fc1 = nn.Linear(input_dim*self.out_channel2,512) # 128
        self.fc2 = nn.Linear(512,out_dim)

        print("kernel:",kernel)
    def forward(self,x):
        x = self.conv1(x)
        #x = x.view(-1,self.hidden_dim*self.out_channel)
        x = Gelu(x)

        x = self.conv2(x)
        x = x.view(-1, self.hidden_dim*self.out_channel2)
        x = Gelu(x)
        x = self.fc1(x)
        x = Gelu(x)
        x = self.fc2(x)
        return x




class ConvTransfer_com(nn.Module):
    def __init__(self, args, input_dim, out_dim, s_item_dimention, t_usr_dimention, t_item_dimention):
        super(ConvTransfer_com, self).__init__()
        self.args = args
        self.meta_transfer = one_transfer(input_dim, out_dim, kernel=4)
        self.fc_source_item = nn.Linear(args.bridge_batch_size*s_item_dimention, input_dim*out_dim)
        self.fc_t_usr = nn.Linear(t_usr_dimention, out_dim)
        self.fc_t_item = nn.Linear(t_item_dimention, out_dim)
        self.mse_loss = nn.MSELoss()
        # self.item_transfer = one_transfer(in_dim,out_dim,kernel=3)
    def forward(self, x_t, x_hat, s_item):
        s_item = self.fc_source_item(s_item.view(-1))
        print('x_t.view(-1).shape: ', x_t.view(-1).shape)
        print('x_hat.view(-1).shape', x_hat.view(-1).shape)
        print('s_item.shape', s_item.shape)
        x_com = torch.mul(torch.mul(x_t.view(-1), x_hat.view(-1).data.detach()), s_item.data.detach())
        x_t_norm = (x_t**2).sum(dim=-1).sqrt()
        x_com = x_com / x_t_norm.unsqueeze(-1)
        x_com.requires_grad = False
        x = torch.cat((s_item, x_t, x_hat, x_com),dim=-1)
        x = x.view(-1,1,4,x_t.shape[-1])
        x = self.meta_transfer(x)
        return x
    #todo
    def cal_loss_emb(self, s_usr_embed_transfer, t_user):

        loss = self.mse_loss(s_usr_embed_transfer, t_user)
        return loss
    #todo
    def cal_loss_final(self, s_usr_embed_transfer, t_item, labels, mode):
        pred = torch.sum(torch.mul(s_usr_embed_transfer, t_item), dim=1)
        logits = nn.Sigmoid(pred).squeeze(dim=-1)
        loss = nn.BCELoss(logits, labels.squeeze(dim=-1))
        pred_label = logits.argmax(1)
        if mode == 'transfer':
            classification_report(labels.data.cpu().numpy(), pred_label.data.cpu().numpy(),\
                target_names=['0', '1'], digits=4, labels=[0,1])
        return loss

    def run_bridge(self, args, bridge_model, last_bridge_weight, bridge_weight_hat, 
                                                source_user_embd_cuda,
                                                source_item_embd_cuda,
                                                target_user_cuda,
                                                target_item_cuda,
                                                labels, mode=None,adpative=False,BCE=True):
        bridge_weight_new = self.forward(last_bridge_weight, bridge_weight_hat, source_item_embd_cuda)
        target_user_cuda = self.fc_t_usr(target_user_cuda)
        target_user_cuda = Gelu(target_user_cuda)
        target_item_cuda = self.fc_t_item(target_item_cuda)
        target_item_cuda = Gelu(target_item_cuda)
        bridge_model.layer.weight.data.copy_(bridge_weight_new)
        source_usr_embed_transfer = bridge_model(source_user_embd_cuda)
        loss1 = self.cal_loss_emb(source_usr_embed_transfer, target_user_cuda)
        loss2 = self.cal_loss_final(source_usr_embed_transfer, target_item_cuda, labels, mode)
        loss = args.loss1_ * loss1 + args.loss2_ * loss2
        
        return loss


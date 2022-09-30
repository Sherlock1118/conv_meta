import torch
import torch.nn.functional as F
import torch.nn as nn


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
        x = self.layer(x)
        return x


class one_transfer(nn.Module):
    '''
    one transfer that contain two cnn layers
    '''
    def __init__(self,input_dim,out_dim, kernel=2):
        super(one_transfer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = input_dim
        self.out_channel = 10
        self.conv1 = nn.Conv2d(1,self.out_channel,(kernel,1),stride=1)
        self.out_channel2 = 5
        self.conv2 = nn.Conv2d(self.out_channel,self.out_channel2,(1,1),stride=1)
        self.fc1 = nn.Linear(input_dim*out_dim*self.out_channel2,input_dim*out_dim*2) # 128
        self.fc2 = nn.Linear(input_dim*out_dim*2,input_dim*out_dim)
        print("kernel:",kernel)
    def forward(self,x):
        x = self.conv1(x)
        #x = x.view(-1,self.hidden_dim*self.out_channel)
        x = Gelu(x)
        x = self.conv2(x)
        x = x.view(-1, self.input_dim*self.out_dim*self.out_channel2)
        x = Gelu(x)
        x = self.fc1(x)
        x = Gelu(x)
        x = self.fc2(x)
        return x.view(self.input_dim, self.out_dim)

class atten_source_emb_fc(nn.Module):
    def __init__(self, args, s_item_dimention, t_item_dimention, input_dim, out_dim):
        super(atten_source_emb_fc, self).__init__()
        self.args = args
        print('s_item_dimention', s_item_dimention)
        self.multihead_attn = nn.MultiheadAttention(s_item_dimention, args.num_heads)
        self.fc_source_item = nn.Linear(s_item_dimention, input_dim*out_dim)
        self.fc_titem2sitem = nn.Linear(t_item_dimention, s_item_dimention)
    
    def forward(self, s_item, t_item):
        t_item = self.fc_titem2sitem(t_item.reshape((self.args.bridge_batch_size, 1, -1)))
        attn_output, attn_output_weights = self.multihead_attn(t_item.reshape((self.args.bridge_batch_size, 1, -1)).transpose(0, 1),\
                                                               s_item.reshape((self.args.bridge_batch_size, 1, -1)).transpose(0, 1),\
                                                               s_item.reshape((self.args.bridge_batch_size, 1, -1)).transpose(0, 1), need_weights=True)
        attn_output = torch.sum(attn_output, dim=1).view(1, -1)
        return attn_output


class ConvTransfer_com(nn.Module):
    def __init__(self, args, input_dim, out_dim, s_item_dimention, t_usr_dimention, t_item_dimention):
        super(ConvTransfer_com, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.meta_transfer = one_transfer(input_dim, out_dim, kernel=4)
        self.atten_emb = atten_source_emb_fc(args, s_item_dimention, t_item_dimention, input_dim, out_dim)
        self.fc_s_item = nn.Linear(s_item_dimention, input_dim*out_dim)
        self.fc_t_usr = nn.Linear(t_usr_dimention, out_dim)
        self.fc_t_item = nn.Linear(t_item_dimention, out_dim)
        self.mse_loss = nn.MSELoss()
        self.layer_this = nn.Linear(input_dim,out_dim)
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
        # self.item_transfer = one_transfer(in_dim,out_dim,kernel=3)
    def forward(self, x_t, x_hat, s_item, t_item):
        s_item = self.atten_emb(s_item, t_item)
        s_item = self.fc_s_item(s_item)
        x_com = torch.mul(torch.mul(x_t.view(-1), x_hat.view(-1).data.detach()), s_item.view(-1).data.detach())
        x_t_norm = (x_t.view(-1)**2).sqrt()
        x_com = x_com / x_t_norm.view(-1)
        x_com.requires_grad = False
        x = torch.cat((s_item.view(1, -1), x_t.view(1, -1), x_hat.view(1, -1), x_com.view(1, -1)),dim=0)
        x = x.view(-1,1,4,x.shape[-1])
        x = self.meta_transfer(x)
        return x
    #todo
    def cal_loss_emb(self, s_usr_embed_transfer, t_user):

        loss = self.mse_loss(s_usr_embed_transfer, t_user)
        return loss
    #todo
    def cal_loss_final(self, s_usr_embed_transfer, t_item, labels, mode):
        pred = torch.sum(torch.mul(s_usr_embed_transfer, t_item), dim=1)
        logits = self.sigmoid(pred).squeeze(dim=-1)
        loss = self.bce(logits, labels.float().squeeze(dim=-1))
        zero_logits = torch.zeros_like(logits)
        one_logits = torch.ones_like(logits)
        pred_label = torch.where(logits >= 0.5, one_logits, zero_logits).long()
#         pred_label = logits.argmax(1)
        if mode == 'transfer':
            return (loss, labels, pred_label)
        else:
            return loss

    def run_bridge(self, args, bridge_model, last_bridge_weight, bridge_weight_hat, 
                                                source_user_embd_cuda,
                                                source_item_embd_cuda,
                                                target_user_cuda,
                                                target_item_cuda,
                                                labels, mode=None,adpative=False,BCE=True):
        bridge_weight_new = self.forward(last_bridge_weight, bridge_weight_hat, source_item_embd_cuda, target_item_cuda)
        target_user_cuda = self.fc_t_usr(target_user_cuda)
        target_user_cuda = Gelu(target_user_cuda)
        target_item_cuda = self.fc_t_item(target_item_cuda)
        target_item_cuda = Gelu(target_item_cuda)
        self.layer_this.weight.data.copy_(bridge_weight_new)
        source_usr_embed_transfer = self.layer_this(source_user_embd_cuda)
        loss1 = self.cal_loss_emb(source_usr_embed_transfer, target_user_cuda)
        loss2 = self.cal_loss_final(source_usr_embed_transfer, target_item_cuda, labels, mode)
        if mode == 'transfer':
            loss = args.loss1_ * loss1 + args.loss2_ * loss2[0]
            return (loss, loss2[1], loss2[2])
        else:
            loss = args.loss1_ * loss1 + args.loss2_ * loss2
            return loss


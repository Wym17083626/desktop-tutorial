#32, 128, 768

import torch
from torch import nn
import torch.nn.functional as F


class tree(nn.Module):
    def __init__(self,input_d,depth,n_label):
        super(tree,self).__init__()
        self.depth=depth
        self.n_node=2**depth-1
        self.fc=nn.Linear(input_d,self.n_node)
        # self.drop=nn.Dropout(0.5)
        self.activation=nn.Sigmoid()
        # self.softmax=nn.Softmax()
        self.n_leaf=2**depth
        self.pi=(torch.nn.Parameter(data=torch.rand((self.n_leaf,n_label)), requires_grad=True))
        # self.register_parameter(name='bias', param=torch.nn.Parameter(torch.randn(3)))
        # self.pi=torch.rand((self.n_leaf,n_label)).cuda()
        # self.pi.requires_grad = True
        self.n_label=n_label
    def forward(self,x):
        decision=self.fc(x)
        decision=self.activation(decision)
        decision = torch.unsqueeze(decision,dim=3)
        decision_comp = 1-decision
        decision = torch.cat((decision,decision_comp),dim=3) # -> [batch_size,n_leaf,2]
        begin=0
        end=1
        batch_size = x.size()[0]
        n_token=x.size()[1]
        routing_probability=torch.ones((batch_size,n_token,1,1)).cuda() 
        
        for n_layer in range(self.depth):
            routing_probability=routing_probability.view(batch_size,n_token,-1,1).repeat(1,1,1,2)
            current_layer_p= decision[:, :,begin:end,:]
            routing_probability=routing_probability*current_layer_p
            begin=end
            end=2*end+1
        routing_probability = routing_probability.view(batch_size,n_token,self.n_leaf)
        prediciton=torch.matmul(routing_probability,F.softmax(self.pi,dim=1))
        # if update_pi:
        #     cls_onehot = torch.eye(self.n_label)
        #     target=target.view(-1,1)
        #     target=cls_onehot[target]
        #     pi = self.pi.data # [n_leaf,n_class]
        #     prob=prediciton.data #batch, token, n_class
        #     mu=routing_probability.data #batch, token, n_leaf

        #     _target = target.unsqueeze(2)  # [batch_size,1,n_class]
        #     _pi = pi.unsqueeze(1)  # [1,n_leaf,n_class]
        #     _mu = mu.unsqueeze(2)  # [batch_size,n_leaf,1]
        #     _prob = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)  # [batch_size,1,n_class]

        #     _new_pi = torch.mul(torch.mul(_target, _pi), _mu) / _prob
        # print(self.pi)
        return prediciton



# input=torch.rand((1, 128, 768))
# print(input.shape)
# a=tree(15,3,5)
# a(input)

class Conditional_forest(nn.Module):
    def __init__(self,n_trees,depth,n_label,input_d):
        super(Conditional_forest,self).__init__()
        self.n_trees = n_trees
        self.n_label=n_label
        self.trees=nn.ModuleList([tree(input_d=input_d,depth=depth,n_label=n_label) for _ in range(self.n_trees)])
        
    # def conditional_forward(self,x,last_p):
    #     print(x)
    #     print(last_p)
    #     #x:[batch,n_tokens,d_feature]
    #     #last_p:[batch,n_tokens,n_label]
    #     batch_size,n_tokens,d_feature=x.shape
    #     n_label=last_p.shape[2]
    #     prediction=torch.zeros((batch_size,n_tokens,n_label))
        # print("==========")
        # for i in range(self.n_trees):
        #     a=self.trees[i](x)
        #     print(a)
        #     b=last_p[:,:,i]
            
        #     b = torch.unsqueeze(b,dim=2)
        #     print(b)
        #     c=a*b
        #     print(c)
        #     prediction+=self.trees[i](x)*torch.unsqueeze(last_p[:,:,i],dim=2)
        # print("===========")
        # print(prediction)
        # return prediction
    def forward(self,x,conditional=False,dependence_length=-1,smooth_phi=0):
        # print(self.trees[0].pi.data)
        batch_size,n_tokens,d_feature=x.shape
        if conditional==False:
            prediction=torch.zeros((batch_size,n_tokens,self.n_label)).cuda() 
            for i in range(self.n_trees):
                # a=self.trees[i](x)
                # print(a)
                # b=last_p[:,:,i]
                
                # b = torch.unsqueeze(b,dim=2)
                # print(b)
                # c=a*b
                # print(c)
                prediction+=self.trees[i](x)*1/self.n_label
        else:
            if dependence_length==-1:
                tree_prediction=torch.zeros((self.n_trees,batch_size,n_tokens,self.n_label)).cuda() 
                for i in range(self.n_trees):
                    tree_prediction[i]+=self.trees[i](x)
                # print(tree_prediction)

                prediction=torch.zeros((batch_size,n_tokens,self.n_label)).cuda()
                tmp_prediction=torch.ones((batch_size,self.n_label)).cuda()/self.n_label
                for tokens in range(n_tokens):
                    tmp_prediction=tmp_prediction.clone().detach()
                    # token_prediction=tree_prediction[:,:,tokens,:]
                    new_tmp_prediction=torch.zeros((batch_size,self.n_label)).cuda() 
                    for i in range(self.n_trees):
                        new_tmp_prediction+=tree_prediction[i,:,tokens,:]*torch.unsqueeze(tmp_prediction[:,i],dim=1)
                    prediction[:,tokens,:]+=new_tmp_prediction
                    # if smooth_phi==0:
                    #     tmp_prediction=new_tmp_prediction
                    # if smooth_phi!=0:
                    tmp_prediction=F.softmax(torch.exp(new_tmp_prediction),dim=1)
            else:

                tree_prediction=torch.zeros((self.n_trees,batch_size,n_tokens,self.n_label)).cuda() 
                for i in range(self.n_trees):
                    tree_prediction[i]+=self.trees[i](x)
                prediction=tree_prediction.sum(dim=0)/self.n_trees
                # print(prediction.size)
                for round in range(1,dependence_length):
                    padding_p=torch.ones((batch_size,1,self.n_label)).cuda() /self.n_label
                    prediction=torch.cat((padding_p,prediction[:,:-1,:]),1)
                    new_prediction=torch.zeros((batch_size,n_tokens,self.n_label)).cuda() 
                    for i in range(self.n_trees):
                        # a=self.trees[i](x)
                        # print(a)
                        # b=last_p[:,:,i]
                        
                        # b = torch.unsqueeze(b,dim=2)
                        # print(b)
                        # c=a*b
                        # print(c)
                        new_prediction+=tree_prediction[i]*torch.unsqueeze(prediction[:,:,i],dim=2)
                    prediction=new_prediction
            # return prediction

                # prediction=prediction[:,]


            # for round in range(dependence_length):
            #     padding_p=torch.zeros((batch_size,1,self.n_label))/self.n_label
            #     prediction=torch.cat((padding_p,prediction[:,1:,:,:]),1)
        #         conditional_forward(self,x,last_p)


        #     last_p=torch.ones((batch_size,1,self.n_label))
        #     prediction=torch.zeros((batch_size,n_tokens,self.n_label))
        #     for i in range(n_tokens):
        #         a=self.trees[i](x)
        #     print(a)
        #     b=last_p[:,:,i]
            
        #     b = torch.unsqueeze(b,dim=2)
        #     print(b)
        #     c=a*b
        #     print(c)
        #     prediction+=self.trees[i](x)*torch.unsqueeze(last_p[:,:,i],dim=2)



        # print(x)
        # print(last_p)
        # #x:[batch,n_tokens,d_feature]
        # #last_p:[batch,n_tokens,n_label]
        # batch_size,n_tokens,d_feature=x.shape
        # n_label=last_p.shape[2]
        # prediction=torch.zeros((batch_size,n_tokens,n_label))
        # print("==========")
        # for i in range(self.n_trees):
        #     a=self.trees[i](x)
        #     print(a)
        #     b=last_p[:,:,i]
            
        #     b = torch.unsqueeze(b,dim=2)
        #     print(b)
        #     c=a*b
        #     print(c)
        #     prediction+=self.trees[i](x)*torch.unsqueeze(last_p[:,:,i],dim=2)
        # print("===========")
        # print(prediction)
        return prediction
# batch_size=1
# n_label=n_trees=10
# # n_trees=3
# tokens=7
# depth=3
# d_feature=8

# input=torch.rand((batch_size, tokens, d_feature))
# a=Conditional_forest(n_trees,depth,n_label,d_feature)
# p=F.softmax(torch.rand((batch_size, tokens, n_label)),dim=2)
# pp=a(input,conditional=False).sum()
# print(pp)
# pp.backward()
        
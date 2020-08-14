import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn
from torch.autograd import Variable
from torch import nn
import pickle as pkl
import os
import random
import string


cpt=1
class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)



    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        global cpt
        
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            '''
            if (self.num_batches_tracked is not None  and self.num_batches_tracked == 40):
                self.num_batches_tracked = 0
            '''
            if self.num_batches_tracked is not None:
                #print("================== TRACKED ===============", self.num_batches_tracked)
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            ''''
            if((int(self.num_batches_tracked) >= 40) and (int(self.num_batches_tracked) % 40 == 0)):
                print("I MA HERE---------------------------", self.num_batches_tracked)
                self.reset_running_stats()
            
            if(self.num_features == 100):
                
                if((int(self.num_batches_tracked) == 1601)):
                    #print("I MA HERE---------------------------", self.num_batches_tracked)
                    self.reset_running_stats()
                
                f = open("/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics/%d" % cpt, "a")
                f.write("TRAIN")
                f.write(str(self.running_mean))
                
                
                f.write("BEFORE")
                f.write(str(self.num_batches_tracked))

                
                f.write("-------------------------------------------------------------------------")
            
                f.close()
                
                random_string = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
                current_directory = os.path.dirname(os.path.realpath(__file__))

                final_directory = "/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics"

                file_path = os.path.join(final_directory, 'tensor-{:04d}-{:s}.pt'.format(cpt, random_string)) 
                torch.save(self.running_mean.data, file_path)
            '''
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
            '''
            if(self.num_features == 100):
                
                if((int(self.num_batches_tracked) == 1601)):
                    #print("I MA HERE---------------------------", self.num_batches_tracked)
                    self.reset_running_stats()
                
                f = open("/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics/%d" % cpt, "a")
                f.write("TRAIN")
                f.write(str(self.running_mean))
                
                
                f.write("AFTERR")
                f.write(str(self.num_batches_tracked))

                
                f.write("-------------------------------------------------------------------------")
            
                f.close()
                
                random_string = ''.join(random.choice(string.ascii_lowercase) for i in range(16))
                current_directory = os.path.dirname(os.path.realpath(__file__))

                final_directory = "/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics"

                file_path = os.path.join(final_directory, 'tensor-{:04d}-{:s}.pt'.format(cpt, random_string)) 
                torch.save(self.running_mean.data, file_path)
            '''
            cpt = cpt+1

            

        else:
            '''
            if(self.num_features == 100):
                print("=============================================Inside Test",self.running_mean)
            
                f = open("/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics/%d" % cpt, "a")
                f.write("TEST")
                f.write(str(self.running_mean))
                
                
                f.write("COUNTER")
                f.write(str(cpt))

                
                f.write("-------------------------------------------------------------------------")
            
                f.close()
            '''
            #self.reset_running_stats()
            mean = self.running_mean
            var = self.running_var
            

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        
        if((int(self.num_batches_tracked) >= 40) and (int(self.num_batches_tracked) % 40 == 0)):
            return input, self.running_mean, self.running_var

        else:
            return input, None, None




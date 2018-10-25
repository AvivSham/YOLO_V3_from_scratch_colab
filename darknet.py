
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import *

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parse_cfg(config_file):
  """
  Takes a config file and returns a list of blocks.
  Each one of the blocks describes a block in the neural network(dictionary in list).
  """
  
  file = open(config_file,'r') # read and store the config file
  lines = file.read().split('\n') # read all lines
  lines = [x for x in lines if (len(x) > 0 and 
                                x[0] != '#')] # read non empty lines & witout comments
  
  lines = [x.rstrip().lstrip() for x in lines] # get rid of unnecessary whitespaces
  
  
  block = {}
  blocks = []

  for line in lines:
      if line[0] == "[":               # This marks the start of a new block
          if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
              blocks.append(block)     # add it the blocks list
              block = {}               # initialize the block
          block["type"] = line[1:-1].rstrip()     
      else:
          key,value = line.split("=") 
          block[key.rstrip()] = value.lstrip()
  blocks.append(block)

  return blocks

def create_module(blocks):
  net_info = blocks[0]
  module_list = nn.ModuleList()
  prev_filters = 3
  output_filters = []
  
  for index, x in enumerate(blocks[1:]):
    module = nn.Sequential()
    
    # Check the block type, if conv
    if (x["type"] == "convolutional"):
      
      # Get the parameters of the layer
      activation = x["activation"]
      try:
        batch_norm = int(x["batch_normalize"])
        bias = False     
      except:
        batch_norm = 0
        bias = True
        
      filters = int(x["filters"])
      padding = int(x["pad"])
      kernel_size = int(x["size"])
      stride = int(x["stride"])
      
      if padding:
        pad = (kernel_size-1) // 2
      else:
        pad = 0
        
      # Adding the Conv layer
      conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
      module.add_module("conv_{0}".format(index), conv)
      
      # Checking and adding Batch_norm
      if batch_norm:
        bn = nn.BatchNorm2d(filters)
        module.add_module("batch_norm{0}".format(index),bn)
        
      # Adding Actiovation function
      # For YOLO implementation its either LeakyReLU of Linear
      if activation == 'leaky':
        activation_func = nn.LeakyReLU(0.1, inplace = True)
        module.add_module("Leaky_{0}".format(index),activation_func)
    
      # If the layer is Upsampling
    elif (x["type"] == "upsample"):
        stride = int(x["stride"])
        upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        module.add_module("Upsample_{}".format(index), upsample)
    
      # If it is a route layer
    elif (x["type"] == "route"):
        x["layers"] = x["layers"].split(',')
        start = int(x["layers"][0])
        try:
          end = int(x["layers"][1])
        except:
          end = 0
          
        if start > 0:
          start = start - index
        
        if end > 0:
          end = end - index
        
        route = EmptyLayer()
        module.add_module("route_{}".format(index), route)
        if end < 0:
          #If concatenating maps
          filters = output_filters[index + start] + output_filters[index + end]
        else:
          filters = output_filters[index + start]
      
    elif (x["type"] == "shortcut"):
        shortcut = EmptyLayer()
        module.add_module("shortcut_{}".format(index), shortcut)
        
    elif (x["type"] == "yolo"):
        mask = x["mask"].split(",")
        mask = [int(x) for x in mask]

        anchors = x["anchors"].split(",")
        anchors = [int(a) for a in anchors]
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
        anchors = [anchors[i] for i in mask]
        
        detection = DetectionLayer(anchors)
        module.add_module("Detection_{}".format(index), detection)
   
    module_list.append(module)
    prev_filters = filters
    output_filters.append(filters)

  return (net_info, module_list)

class Darknet(nn.Module):
  '''
  We subclassed the nn.module class and named our class 'Darknet'
  The initialization is with blocks, net_info, module_list
  '''
  
  def __init__(self, cfgfile):
      super(Darknet, self).__init__()
      self.blocks = parse_cfg(cfgfile)
      self.net_info, self.module_list = create_module(self.blocks)

  def forward(self, x, CUDA):
      modules = self.blocks[1:]
      outputs = {}  #Store the outputs of the route layer

      write = 0
      for i, module in enumerate(modules):
        module_type = (module["type"])
        if module_type == "convolutional" or module_type == "upsample":
          x = self.module_list[i](x)
        elif module_type == "route":
          layers = module["layers"]
          layers = [int(a) for a in layers]

          if(layers[0]) > 0:
            layers[0] = layers[0] - i
          if len(layers) == 1:
            x = outputs[i + (layers[0])]
          else:
            if(layers[1]) > 0:
              layers[1] = layers[1] - i

            map1 = outputs[i + layers[0]]
            map2 = outputs[i + layers[1]]

            x = torch.cat((map1,map2), 1)

        elif module_type == "shortcut":
          from_ = int(module["from"])
          x = outputs[i-1] + outputs [i+from_]

        elif module_type == 'yolo':
          anchors = self.module_list[i][0].anchors
          inp_dim = int(self.net_info["height"])

          #num of classes
          num_classes = int(module["classes"])

          #Transform
          x = x.data
          x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
          if not write:
            detections = x
            write = 1

          else:
            detections = torch.cat((detections, x), 1)

        outputs[i] = x

      return detections
  
  def load_weights(self, weightfile):
    
    # Open the weights file for reading
    
    fp = open(weightfile, 'rb');
    
    # read the headers
    header = np.fromfile(fp, dtype = np.int32, count = 5)
    self.header = torch.from_numpy(header)
    self.seen = self.header[3]
    
    # the rest of bits represent the weights
    weights = np.fromfile(fp, dtype = np.float32)
    
    # load the weights to the model
    ptr = 0
    for i in range(len(self.module_list)):
      module_type = self.blocks[i+1]["type"]
      if module_type == "convolutional":
        model = self.module_list[i]
        # Due to the way that the weights file is written we need to
        # know if the conv layers has batch norm or not
        try:
          batch_norm = int(self.blocks[i+1]["batch_normalize"])
        except:
          batch_norm = 0

        conv = model[0]

        if batch_norm:
          bn = model[1]

          num_bn_biases = bn.bias.numel()

          bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
          ptr += num_bn_biases

          bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
          ptr += num_bn_biases

          bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
          ptr += num_bn_biases

          bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
          ptr += num_bn_biases

          #cast the loaded weights into dims of model weights.
          bn_biases = bn_biases.view_as(bn.bias.data)
          bn_weights = bn_weights.view_as(bn.weight.data)
          bn_running_mean = bn_running_mean.view_as(bn.running_mean)
          bn_running_var = bn_running_var.view_as(bn.running_var)

          # copy the data to model
          bn.bias.data.copy_(bn_biases)
          bn.weight.data.copy_(bn_weights)
          bn.running_mean.data.copy_(bn_running_mean)
          bn.running_var.data.copy_(bn_running_var)

        # if there is not batch_norm just load the weights to the model
        else:
          num_biases = conv.bias.numel()

          # load the weights
          conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
          ptr += num_biases

          # reshape the loaded weights according to the dims of the model weights
          conv_biases = conv_biases.view_as(conv.bias.data)

          # copy the data
          conv.bias.data.copy_(conv_biases)

        #load the weights for the Convolutional layers
        num_weights = conv.weight.numel()

        #Do the same as above for weights
        conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
        ptr = ptr + num_weights

        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

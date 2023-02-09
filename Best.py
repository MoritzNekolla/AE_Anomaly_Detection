#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import optim
import time
import torch.nn.functional as  F
import sys
sys.path.insert(0, 'py_scripts/') 
from clearml import Dataset

from Image_Sampler import Sampler



from clearml import Task, Logger



MODEL_NAME = "clearML"
PATH = "models/" + MODEL_NAME
# IMG_TRAIN = "/disk/vanishing_data/is789/anomaly_samples/train_set/"
# IMG_TEST = "/disk/vanishing_data/is789/anomaly_samples/40test/"
TRAIN_ID = "bab5282c27d84032963519640975b252"
# TRAIN_ID = "834165cf80dd430b9374de0c3414c23a" #Simon train cityscape train: cb2f41e2945247ccb906b19aab083875
# TEST_ID = "cd75e39b0aa641fc9b7e6d6c76656627"
# TRAIN_ID = "8ce5cdd31e8e499db2e07fc70b6136d5"


### ClearML section
task = Task.init(project_name="bogdoll/Anomaly_detection_Moritz", task_name="final_AE", output_uri="s3://tks-zx.fzi.de:9000/clearml")
task.set_base_docker(
            "nvcr.io/nvidia/pytorch:21.10-py3",
            docker_setup_bash_script="apt-get update && apt-get install -y python3-opencv",
            docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all"  # --ipc=host",   
            )
task.execute_remotely('rtx3090', clone=False, exit_process=True) 


###lr anpassen auch die early stops
# epoch = 2500
# BATCH_SIZE = 32
# imgSize=512
# zDim=512
# learning_rate = 1e-05 #1e-04 
# REDUCE_THRESHOLD = [0.6,0.8]
# layers=[32, 64, 128, 265, 512]

parameters = {
    "epoch" : 1000,
    "batch_size" : 7,
    "imgSize": 256,
    "zDim": 1024,
    "learning_rate" : 1e-05,
#     "layers" : [64, 128, 256, 256, 512, 512, 940],
    "layers" : [64, 120, 240, 480, 512],
    "layers_out" : [512,256,128,64],
    "reduce_threshold" : [0.6,0.8]
}
test_size = 1000

guard = int(parameters["epoch"] / 10)
if guard == 0: guard = 1


print("Loading data...")
train_root = Dataset.get(dataset_id=TRAIN_ID).get_local_copy()
train_data = Sampler.load_Images(train_root).astype("float32") / 255

train_data, test_data = torch.utils.data.random_split(train_data, [len(train_data) - test_size, test_size])
train_data = np.array(train_data)
test_data = np.array(test_data)
print(train_data.shape)
print(test_data.shape)

parameters["train_data"] = train_data.shape
parameters["test_data"] = test_data.shape


#start ClearML logging
task.connect(parameters)
logger = task.get_logger()



## transpose images

def transpose_img(data):
    x = 0
    new_data = []
    for img in data:
        img = np.transpose(img, (2,1,0))
        new_data.append(img)
        
    return np.array(new_data)

train_data = torch.as_tensor(transpose_img(train_data))
test_data = torch.as_tensor(transpose_img(test_data))



# # NUR für Simon data
# def resize(data):
#     resizer = transforms.Resize((256,256))
#     tmp = []
#     for img in data:
#         kk = resizer(img)
#         kk = np.array(kk)
#         tmp.append(kk)
        
#     return tmp

# train_data = torch.as_tensor(resize(train_data))
# test_data = torch.as_tensor(resize(test_data))


# In[ ]:


img = train_data[14]
img = img.numpy()
img = np.transpose(img, (2,1,0))
plt.imshow(img)


# In[ ]:


# ## apply noise
# noise_set = []

# for img in test_data:
#     pic = img.numpy().copy()
#     pic[:,250:310,200:350] = 0.0
#     noise_set.append(pic)

# noise_set = np.array(noise_set)
# noise_set = torch.as_tensor(noise_set)
# noise_set.shape


# In[ ]:


# ## show noisy image

# img = noise_set[11]
# # img = img.squeeze()
# img = img.numpy()
# img = np.transpose(img, (2,1,0))
# plt.imshow(img)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:



class VAE(nn.Module):
    
    def __init__(self, imgChannels=3, imgSize=parameters["imgSize"], zDim=parameters["zDim"]):
        super(VAE, self).__init__()
        
        
        stride=[1,2,2,2,2]
        out_stride=[2,2,2,2,1]
#         in_stride=[1,2,2,2,2]
#         out_stride=[1,2,2,2,1]
        in_padding=[3,0,0,0,0]
        in_trans_padding=[0,0,1,0,0]
        out_padding=[0,0,1,0,0]
        kernel=[7,3,3,3,3]
        kernel_out=[3,3,4,4,1]
#         layers=[128, 128, 128, 256, 256]
        layers=parameters["layers"]
        layers_out = parameters["layers_out"]
#         layers=[32, 64, 64, 128, 128]
#         layers=[64, 128, 128, 128, 256]
        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(in_channels=imgChannels, out_channels=layers[0], kernel_size=kernel[0], stride=stride[0], padding=in_padding[0])
        self.encBn1 = nn.BatchNorm2d(layers[0])
        self.encConv2 = nn.Conv2d(in_channels=layers[0], out_channels=layers[1], kernel_size=kernel[1], stride=stride[1], padding=in_padding[1])
        self.encBn2 = nn.BatchNorm2d(layers[1])
        self.encConv3 = nn.Conv2d(in_channels=layers[1], out_channels=layers[2], kernel_size=kernel[2], stride=stride[2], padding=in_padding[2])
        self.encBn3 = nn.BatchNorm2d(layers[2])
        self.encConv4 = nn.Conv2d(in_channels=layers[2], out_channels=layers[3], kernel_size=kernel[3], stride=stride[3], padding=in_padding[3])
        self.encBn4 = nn.BatchNorm2d(layers[3])
        self.encConv5 = nn.Conv2d(in_channels=layers[3], out_channels=layers[4], kernel_size=kernel[4], stride=stride[4], padding=in_padding[4])
        self.encBn5 = nn.BatchNorm2d(layers[4])
#         self.encConv6 = nn.Conv2d(in_channels=layers[4], out_channels=layers[5], kernel_size=kernel[5], stride=stride[5], padding=in_padding[5])
#         self.encBn6 = nn.BatchNorm2d(layers[5])
#         self.encConv7 = nn.Conv2d(in_channels=layers[5], out_channels=layers[6], kernel_size=kernel[6], stride=stride[6], padding=in_padding[6])
#         self.encBn7 = nn.BatchNorm2d(layers[6])
        
        encoderDims = self.calcEncoderDims(len(layers), imgSize, kernel, in_padding, stride)
        featureDim = layers[-1] * encoderDims[-1] * encoderDims[-1]
        self.encFC1 = nn.Linear(featureDim, zDim)

# #         Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decBn1 = nn.BatchNorm1d(featureDim)
        self.decConv1 = nn.ConvTranspose2d(in_channels=layers[4], out_channels=layers_out[0], kernel_size=kernel_out[0], stride=out_stride[0], padding=in_trans_padding[0], output_padding=out_padding[0])
        self.decBn2 = nn.BatchNorm2d(layers_out[0])
        self.decConv2 = nn.ConvTranspose2d(in_channels=layers_out[0], out_channels=layers_out[1], kernel_size=kernel_out[1], stride=out_stride[1], padding=in_trans_padding[1], output_padding=out_padding[1])
        self.decBn3 = nn.BatchNorm2d(layers_out[1])
        self.decConv3 = nn.ConvTranspose2d(in_channels=layers_out[1], out_channels=layers_out[2], kernel_size=kernel_out[2], stride=out_stride[2], padding=in_trans_padding[2], output_padding=out_padding[2])
        self.decBn4 = nn.BatchNorm2d(layers_out[2])
        self.decConv4 = nn.ConvTranspose2d(in_channels=layers_out[2], out_channels=layers_out[3], kernel_size=kernel_out[3], stride=out_stride[3], padding=in_trans_padding[3], output_padding=out_padding[3])
        self.decBn5 = nn.BatchNorm2d(layers_out[3])
        self.decConv5 = nn.ConvTranspose2d(in_channels=layers_out[3], out_channels=imgChannels, kernel_size=kernel_out[4], stride=out_stride[4], padding=in_trans_padding[4], output_padding=out_padding[4])
#         self.decBn6 = nn.BatchNorm2d(layers[1])
#         self.decConv6 = nn.ConvTranspose2d(in_channels=layers[1], out_channels=layers[0], kernel_size=kernel[1], stride=stride[1], padding=in_trans_padding[5], output_padding=out_padding[5])
#         self.decBn7 = nn.BatchNorm2d(layers[0])
#         self.decConv7 = nn.ConvTranspose2d(in_channels=layers[0], out_channels=imgChannels, kernel_size=kernel[0], stride=stride[0], padding=in_trans_padding[6], output_padding=out_padding[6])
        
        self.final_encoder_dim = None
        
        decoderDims = self.calcDecoderDims(len(layers), encoderDims[-1], kernel_out, in_trans_padding, out_padding, out_stride)
        self.printModel(layers, layers_out, encoderDims, decoderDims, imgSize, imgChannels)

    def calcEncoderDims(self, layer_size, imageSize, kernel, in_padding, stride):
        newDims = [imageSize]
        for x in range(layer_size):
#             tmpSize = int((newDims[-1]-kernel[x]+2*in_padding[x])/stride[x])+1
            tmpSize = int(((newDims[-1] + 2*in_padding[x]-(kernel[x]-1)-1)/stride[x])+1)
            newDims.append(tmpSize)
        newDims.pop(0)
        return newDims
    
    def calcDecoderDims(self, layer_size, imageSize, kernel, in_trans_padding, out_padding, stride, d=1):
        newDims = [imageSize]
        for x in range(layer_size):            
            tmpSize = (newDims[-1] - 1)*stride[x] - 2*in_trans_padding[x] + d*(kernel[x] - 1) + out_padding[x] + 1
            newDims.append(tmpSize)
#         newDims.pop(0)
        return newDims
    
    
    def printModel(self, layers, layers_out, encDims, decDims, imageSize, imgChannels):
        print("=============")
        print("Image Flow:")
        print("Encoder:")
        print(f"{imageSize}x{imageSize}x{imgChannels} (Input Image)")
        for x in range(len(layers)):
            print(f"{encDims[x]}x{encDims[x]}x{layers[x]}")
        
        print("Decoder:")
        for x in range(len(layers_out)):
            if x == 0:
                print(f"{decDims[x]}x{decDims[x]}x{layers[-1]}")
            print(f"{decDims[x]}x{decDims[x]}x{layers_out[x]}")
        print(f"{decDims[-1]}x{decDims[-1]}x{imgChannels} (Output Image)")
        print("=============")
            
        
    def encoder(self, x):
#         a = 
# #         b = self.res_conv1(x)
#         print(a.size())
#         x = x.resize_(2,32,510,510)
#         print(x.size())
        x1 = F.leaky_relu(self.encConv1(x))
        x1 = self.encBn1(x1)
#         x = self.res_conv1(x).resize_(parameters["batch_size"],512,512)
        x2 = F.leaky_relu(self.encConv2(x1))
        x2 = self.encBn2(x2)
        x3 = F.leaky_relu(self.encConv3(x2))
        x3 = self.encBn3(x3)
        x4 = F.leaky_relu(self.encConv4(x3))
        x4 = self.encBn4(x4)
        x5 = F.leaky_relu(self.encConv5(x4))
        x5 = self.encBn5(x5)
#         x6 = F.relu(self.encConv6(x5))
#         x6 = self.encBn6(x6)
#         x7 = F.relu(self.encConv7(x6))
#         x7 = self.encBn7(x7)
        self.final_encoder_dim = np.array([x5.size(1), x5.size(2), x5.size(3)])
        flatten = np.prod(self.final_encoder_dim)

        x7 = x5.view(-1, flatten)
        z = self.encFC1(x7)
        
        return z
#         return x5


    def decoder(self, z):

        d1 = F.leaky_relu(self.decFC1(z))
        d1 = self.decBn1(d1)
        d1 = d1.view(-1, self.final_encoder_dim[0], self.final_encoder_dim[1], self.final_encoder_dim[2])
        d2 = F.leaky_relu(self.decConv1(d1))
        d2 = self.decBn2(d2)
        d3 = F.leaky_relu(self.decConv2(d2))
        d3 = self.decBn3(d3)
        d4 = F.leaky_relu(self.decConv3(d3))
        d4 = self.decBn4(d4)
        d5 = F.leaky_relu(self.decConv4(d4))
        d5 = self.decBn5(d5)
#         d6 = F.relu(self.decConv5(d5))
#         d6 = self.decBn6(d6)
#         d7 = F.relu(self.decConv6(d6))
#         d7 = self.decBn7(d7)
        d8 = torch.sigmoid(self.decConv5(d5))
        return d8

    def forward(self, x):

        z = self.encoder(x)

        out = self.decoder(z)
        return out
    
#     def residual(self, x, out_channels, stride=2, kernel=1, padding=1):
#         conv = nn.Conv2d(in_channels=imgChannels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)


# In[ ]:


# model = VAE()
# model.to(device)
# model.load_state_dict(torch.load(PATH))


# In[ ]:



def loss_fn(x, recon_x):
#     Recon_loss = F.mse_loss(recon_x.view(-1, 1024), x.view(-1, 1024), reduction = "sum")
#     Recon_loss = F.mse_loss(recon_x.view(-1, 1024), x.view(-1, 1024)) * 32 * 32
#     BC_loss = F.cross_entropy(recon_x.view(-1, 1024), x.view(-1, 1024))
#     KLD_loss = 1 + log_var - mu.pow(2) - log_var.exp()
#     KLD_loss = torch.sum(KLD_loss)
#     KLD_loss *= -0.5
#     return torch.mean(Recon_loss + KLD_loss)
#     Recon_loss = F.mse_loss(recon_x.view(-1, 2500), x.view(-1, 2500), reduction = "sum") * 32 * 32 *3
#     Recon_loss = F.binary_cross_entropy(recon_x.view(-1, imgSize*imgSize), x.view(-1, imgSize*imgSize), reduction = "sum") * imgSize * imgSize *3
    imgSize = parameters["imgSize"]
    Recon_loss = F.mse_loss(recon_x.view(-1, imgSize*imgSize), x.view(-1, imgSize*imgSize), reduction = "sum")
    return Recon_loss, Recon_loss
#     return Recon_loss_adapted, Recon_loss


# In[ ]:


model = VAE()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = parameters["learning_rate"])

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {pytorch_total_params}")


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torch.utils.data import DataLoader

dataloaders = {}
dataloaders["train"] = DataLoader(dataset=train_data,
                                          batch_size=parameters["batch_size"],
                                          shuffle=True,
                                          drop_last=True)

dataloaders["test"] = DataLoader(dataset=test_data,
                                          batch_size=5,
                                          shuffle=True,
                                          drop_last=True)

# dataloaders["noise"] = DataLoader(dataset=noise_set,
#                                           batch_size=5,
#                                           shuffle=True,
#                                           drop_last=True)

images = next(iter(dataloaders["test"]))
plt.imshow(np.transpose(images[0], (2,1,0)))


# In[ ]:


def evalOnSet(data):
    avg_MSE = 0
    for img in data:
        true_img = img.numpy()
        img = np.array([true_img])
        img = torch.as_tensor(img)
        img = img.to(device)
        out, _, _ = model(img)
        out = out[0].detach().cpu().numpy()
        errorMatrix = np.absolute(true_img - out)
        errorAvg = np.sum(errorMatrix) / (errorMatrix.shape[0] * errorMatrix.shape[1] * errorMatrix.shape[2])
        avg_MSE += errorAvg
    return avg_MSE / len(data)


# In[ ]:


import random

def make_prediction(dataSet, index):
    model.eval()
    with torch.no_grad():
        imgs = torch.as_tensor(np.array([dataSet[index].numpy()]))
        imgs = imgs.to(device)
    #         img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
        true_img = imgs[0].cpu().numpy()
        true_img = np.transpose(true_img, (2,1,0))

        out = model(imgs)
    #         outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
        out = out[0].cpu().numpy()
        out = np.transpose(out, (2,1,0))
        errorMatrix = np.absolute(true_img - out)
        errorAvg = np.sum(errorMatrix) / (errorMatrix.shape[0] * errorMatrix.shape[1] * errorMatrix.shape[2])
        errorAvg = int(errorAvg * 100000)/ 100000.0
        
        #plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
        ax1.set_title("Original")
        ax1.imshow(true_img)
        ax2.set_title(f"Reconstruction | MAE: {errorAvg}")
        ax2.imshow(out)
        return fig
    
def save_prediction(dataset, index, epoch, group, title):
    fig = make_prediction(dataset, index)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    logger.report_image(group, title, iteration=epoch, image=data)
    fig.clear()


# In[1]:


def performance_snapshot(epoch=None):
    save_prediction(train_data, 0, epoch=epoch, group="Snapshot_Train_set", title="001")
    save_prediction(train_data, 1, epoch=epoch, group="Snapshot_Train_set", title="002")
    save_prediction(train_data, 2, epoch=epoch, group="Snapshot_Train_set", title="003")
    save_prediction(train_data, 3, epoch=epoch, group="Snapshot_Train_set", title="004")
    
    save_prediction(test_data, 0, epoch=epoch, group="Snapshot_Validation_set", title="001")
    save_prediction(test_data, 1, epoch=epoch, group="Snapshot_Validation_set", title="002")
    save_prediction(test_data, 2, epoch=epoch, group="Snapshot_Validation_set", title="003")
    save_prediction(test_data, 3, epoch=epoch, group="Snapshot_Validation_set", title="004")


# In[ ]:


train_losses = []
val_losses = []
train_MAE = []
val_MAE = []

reducer = []
for cutter in parameters["reduce_threshold"]:
    reducer.append(int(parameters["epoch"] * cutter))

print("Training started!")

for e in range(1, parameters["epoch"]+1):
    for reduce in reducer:    
        if e == reduce:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 10
            print("Changed learningrate")

    train_loss = 0.0
    train_mse = 0.0
    for x in dataloaders["train"]:
        x = x.to(device)
        x_recon = model(x)

        optimizer.zero_grad()
        loss, mse = loss_fn(x, x_recon)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mse += mse.item()
    
    val_loss = 0.0
    val_mse = 0.0
    for x in dataloaders["test"]:
        x = x.to(device)
        x_recon = model(x)

        optimizer.zero_grad()
        loss, mse = loss_fn(x, x_recon)

        val_loss += loss.item()
        val_mse += mse.item()
        
    ## logging
    train_loss /= len(dataloaders["train"].dataset)
    val_loss /= len(dataloaders["test"].dataset)
    train_mse /= len(dataloaders["train"].dataset)
    val_mse /= len(dataloaders["test"].dataset)
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
    logger.report_scalar(
        "loss", "train", iteration=e, value=train_loss)
    logger.report_scalar(
        "loss", "validation", iteration=e, value=val_loss)
    logger.report_scalar(
        "MSE", "train", iteration=e, value=train_mse)
    logger.report_scalar(
        "MSE", "validation", iteration=e, value=val_mse)
    if e % guard == 0 and not e == 0:
        model.eval()
        performance_snapshot(epoch=e)
        torch.save(model.state_dict(), "model.pt")
        model.train()

    print(f"Epoch {e} | Loss: {train_loss} | V_Loss: {val_loss} | MSE: {train_mse} | V_MSE: {val_mse}")


# ## Evaluation Section

# In[ ]:


# save final model
torch.save(model.state_dict(), "model.pt")


# In[ ]:


model.eval()


# In[ ]:


# end_time = time.time()
# time_elapsed = ((end_time - start_time) / 60.0) / 60.0
# time_elapsed = int(time_elapsed * 100000)/ 100000.0

# avg_mse = evalOnSet(test_data)
# print(avg_mse)

# file = open(PATH + "summary.txt", "w")
# file.write("Train loss: " + str(train_losses[-1]) + " Val loss: " + str(val_losses[-1]))
# file.write("\nEpochs: " + str(epoch))
# file.write("\nBatchSize: " + str(BATCH_SIZE))
# file.write("\nzDim: " + str(zDim))
# file.write("\nAvg mse on test_data: " + str(avg_mse))
# file.write("\nTime elapsed: " + str(time_elapsed) + " hours")


# In[ ]:


# plt.figure(figsize=(10,5))
# plt.title("Training and Validation Loss")
# plt.plot(val_losses[15:],label="val")
# plt.plot(train_losses[15:],label="train")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig(PATH + "loss_plot.svg")
# plt.savefig(PATH + "loss_plot.png")
# plt.show()


# In[ ]:


def printReconError(img_in, img_out, threshold=None):
    img_in = img_in.dot([0.07, 0.72, 0.21])
    img_out = img_out.dot([0.07, 0.72, 0.21])
    errorMatrix = np.absolute(img_in - img_out)
    if not threshold == None:
        errorMatrix[errorMatrix < threshold] = 0.0
    errorAvg = np.sum(errorMatrix) / (errorMatrix.shape[0] * errorMatrix.shape[1])
    print(f"MAE: {errorAvg}")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,15))
    
    ax1.set_title("Original")
    ax1.imshow(img_in, cmap="gray")
    ax2.set_title("Recreation")
    ax2.imshow(img_out, cmap="gray")
    ax3.set_title("ErrorMap")
    ax3.imshow(errorMatrix, cmap="gray")
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    logger.report_image("zAnomaly", "forecast", image=data)


# In[ ]:


logger.report_text('log some text', print_console=False)
logger.flush()


# In[ ]:


import random

# model.eval()
with torch.no_grad():
    for imgs in random.sample(list(dataloaders["train"]), 1):
        imgs = imgs.to(device)
#         plt.subplot(121)
        img = imgs[0].cpu().numpy()
        img = np.transpose(img, (2,1,0))
#         plt.imshow(img, cmap="gray")
        
        out = model(imgs)
#         plt.subplot(122)
        out = out[0].cpu().numpy()
        out = np.transpose(out, (2,1,0))
#         plt.imshow(out, cmap="gray")
        
        printReconError(img, out, 0.0)
        break


# In[1]:


save_prediction(train_data, 0, epoch=None, group="#Train_set", title="001")
save_prediction(train_data, 1, epoch=None, group="#Train_set", title="002")
save_prediction(train_data, 2, epoch=None, group="#Train_set", title="003")
save_prediction(train_data, 3, epoch=None, group="#Train_set", title="004")

save_prediction(test_data, 0, epoch=None, group="#Validation_set", title="001")
save_prediction(test_data, 1, epoch=None, group="#Validation_set", title="002")
save_prediction(test_data, 2, epoch=None, group="#Validation_set", title="003")
save_prediction(test_data, 3, epoch=None, group="#Validation_set", title="004")


# In[ ]:





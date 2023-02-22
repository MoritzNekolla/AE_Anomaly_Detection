import torch
import torch.nn as nn
import torch.nn.functional as F

from env_carla import IM_HEIGHT
from env_carla import IM_WIDTH
from env_carla import N_ACTIONS

### Mode: Observation + Anomaly RL Agent!
class DQN_DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)      # 3x3 kernel that moves 2 pixels per iteration
        self.bn1 = nn.BatchNorm2d(16)                               # Normalize input features so they are on the same scale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(32)

        self.anomalyConv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)      # 3x3 kernel that moves 2 pixels per iteration
        self.anomalyBn1 = nn.BatchNorm2d(16)                               # Normalize input features so they are on the same scale
        self.anomalyConv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.anomalyBn2 = nn.BatchNorm2d(32)
        self.anomalyConv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.anomalyBn3 = nn.BatchNorm2d(32)
        self.anomalyConv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.anomalyBn4 = nn.BatchNorm2d(32)
        self.anomalyConv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.anomalyBn5 = nn.BatchNorm2d(32)
        self.anomalyConv6 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.anomalyBn6 = nn.BatchNorm2d(32)

        self.mapConv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)      # 3x3 kernel that moves 2 pixels per iteration
        self.mapBn1 = nn.BatchNorm2d(16)                               # Normalize input features so they are on the same scale
        self.mapConv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.mapBn2 = nn.BatchNorm2d(32)
        self.mapConv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.mapBn3 = nn.BatchNorm2d(32)
        self.mapConv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.mapBn4 = nn.BatchNorm2d(32)
        self.mapConv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.mapBn5 = nn.BatchNorm2d(32)
        self.mapConv6 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.mapBn6 = nn.BatchNorm2d(32)

        #Compute number of linear input connections after conv2d layers (https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173)
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw1 = conv2d_size_out(IM_WIDTH)
        convh1 = conv2d_size_out(IM_HEIGHT)

        convw2 = conv2d_size_out(convw1)
        convh2 = conv2d_size_out(convh1)

        convw3 = conv2d_size_out(convw2)
        convh3 = conv2d_size_out(convh2)

        convw4 = conv2d_size_out(convw3)
        convh4 = conv2d_size_out(convh3)

        convw5 = conv2d_size_out(convw4)
        convh5 = conv2d_size_out(convh4)

        convw6 = conv2d_size_out(convw5)
        convh6 = conv2d_size_out(convh5)

        linear_input_size = convw6 * convh6 * 32                    # width * height * channels
        print(linear_input_size)

        self.concat = nn.Linear(linear_input_size * 3, linear_input_size)


        self.head = nn.Linear(linear_input_size, N_ACTIONS)

    def forward(self, t):
        '''Called with either one element to determine next action, or a batch during optimization'''
        t = torch.tensor_split(t, 3, dim=1)
        observation = torch.squeeze(t[0], dim=1)
        anomaly = torch.squeeze(t[1], dim=1)
        miniMap = torch.squeeze(t[2], dim=1)

        # print(observation.size())

########### Observation pipeline ###############
        observation = self.conv1(observation)
        observation = self.bn1(observation)
        observation = F.relu(observation)           # Acivaion funcion

        observation = F.relu(self.bn2(self.conv2(observation)))
        observation = F.relu(self.bn3(self.conv3(observation)))
        observation = F.relu(self.bn4(self.conv4(observation)))
        observation = F.relu(self.bn5(self.conv5(observation)))
        observation = F.relu(self.bn6(self.conv6(observation)))

        observation = observation.view((observation.size()[0], observation.size()[1] * observation.size()[2] * observation.size()[3]))

########### Anomaly pipeline ###############
        anomaly = self.anomalyConv1(anomaly)
        anomaly = self.anomalyBn1(anomaly)
        anomaly = F.relu(anomaly)           # Acivaion funcion

        anomaly = F.relu(self.anomalyBn2(self.anomalyConv2(anomaly)))
        anomaly = F.relu(self.anomalyBn3(self.anomalyConv3(anomaly)))
        anomaly = F.relu(self.anomalyBn4(self.anomalyConv4(anomaly)))
        anomaly = F.relu(self.anomalyBn5(self.anomalyConv5(anomaly)))
        anomaly = F.relu(self.anomalyBn6(self.anomalyConv6(anomaly)))

        anomaly = anomaly.view((anomaly.size()[0], anomaly.size()[1] * anomaly.size()[2] * anomaly.size()[3]))

########### minimap pipeline ###############
        miniMap = self.mapConv1(miniMap)
        miniMap = self.mapBn1(miniMap)
        miniMap = F.relu(miniMap)           # Acivaion funcion

        miniMap = F.relu(self.mapBn2(self.mapConv2(miniMap)))
        miniMap = F.relu(self.mapBn3(self.mapConv3(miniMap)))
        miniMap = F.relu(self.mapBn4(self.mapConv4(miniMap)))
        miniMap = F.relu(self.mapBn5(self.mapConv5(miniMap)))
        miniMap = F.relu(self.mapBn6(self.mapConv6(miniMap)))

        miniMap = miniMap.view((miniMap.size()[0], miniMap.size()[1] * miniMap.size()[2] * miniMap.size()[3]))

########### After ###############
        concatination = torch.cat((observation, anomaly, miniMap), dim=1)
        linear1 = F.relu(self.concat(concatination))

        output = self.head(linear1)
        output = output.view(output.size(0), -1)
        # print(output.size())
        return output



class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(9, 16, kernel_size=3, stride=2)      # 3x3 kernel that moves 2 pixels per iteration
        self.bn1 = nn.BatchNorm2d(16)                               # Normalize input features so they are on the same scale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(32)

        self.mapConv1 = nn.Conv2d(9, 16, kernel_size=3, stride=2)      # 3x3 kernel that moves 2 pixels per iteration
        self.mapBn1 = nn.BatchNorm2d(16)                               # Normalize input features so they are on the same scale
        self.mapConv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.mapBn2 = nn.BatchNorm2d(32)
        self.mapConv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.mapBn3 = nn.BatchNorm2d(32)
        self.mapConv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.mapBn4 = nn.BatchNorm2d(32)
        self.mapConv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.mapBn5 = nn.BatchNorm2d(32)
        self.mapConv6 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.mapBn6 = nn.BatchNorm2d(32)

        #Compute number of linear input connections after conv2d layers (https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173)
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw1 = conv2d_size_out(IM_WIDTH)
        convh1 = conv2d_size_out(IM_HEIGHT)

        convw2 = conv2d_size_out(convw1)
        convh2 = conv2d_size_out(convh1)

        convw3 = conv2d_size_out(convw2)
        convh3 = conv2d_size_out(convh2)

        convw4 = conv2d_size_out(convw3)
        convh4 = conv2d_size_out(convh3)

        convw5 = conv2d_size_out(convw4)
        convh5 = conv2d_size_out(convh4)

        convw6 = conv2d_size_out(convw5)
        convh6 = conv2d_size_out(convh5)

        linear_input_size = convw6 * convh6 * 32                    # width * height * channels
        print(f"Latent-size: {linear_input_size}")

        self.concat = nn.Linear(linear_input_size * 2, linear_input_size)


        self.head = nn.Linear(linear_input_size, N_ACTIONS)

    def forward(self, t):
        '''Called with either one element to determine next action, or a batch during optimization'''
        t = torch.tensor_split(t, 2, dim=1)
        observation = torch.squeeze(t[0], dim=1)
        miniMap = torch.squeeze(t[1], dim=1)

        # print(observation.size())

########### Observation pipeline ###############
        observation = self.conv1(observation)
        observation = self.bn1(observation)
        observation = F.relu(observation)           # Acivaion funcion

        observation = F.relu(self.bn2(self.conv2(observation)))
        observation = F.relu(self.bn3(self.conv3(observation)))
        observation = F.relu(self.bn4(self.conv4(observation)))
        observation = F.relu(self.bn5(self.conv5(observation)))
        observation = F.relu(self.bn6(self.conv6(observation)))

        observation = observation.view((observation.size()[0], observation.size()[1] * observation.size()[2] * observation.size()[3]))

########### minimap pipeline ###############
        miniMap = self.mapConv1(miniMap)
        miniMap = self.mapBn1(miniMap)
        miniMap = F.relu(miniMap)           # Activation funcion

        miniMap = F.relu(self.mapBn2(self.mapConv2(miniMap)))
        miniMap = F.relu(self.mapBn3(self.mapConv3(miniMap)))
        miniMap = F.relu(self.mapBn4(self.mapConv4(miniMap)))
        miniMap = F.relu(self.mapBn5(self.mapConv5(miniMap)))
        miniMap = F.relu(self.mapBn6(self.mapConv6(miniMap)))

        miniMap = miniMap.view((miniMap.size()[0], miniMap.size()[1] * miniMap.size()[2] * miniMap.size()[3]))

########### After ###############
        concatination = torch.cat((observation, miniMap), dim=1)
        linear1 = F.relu(self.concat(concatination))

        output = self.head(linear1)
        output = output.view(output.size(0), -1)
        # print(output.size())
        return output
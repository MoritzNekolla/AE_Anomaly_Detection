#!/usr/bin/env python
# coding: utf-8


from operator import truediv
import re
from turtle import distance
import cv2
import time
import torch
import numpy as np
import argparse
import math

import os
import sys
from tempfile import gettempdir

from torch.utils.tensorboard import SummaryWriter

from clearml import Task

from AE_model import AutoEncoder
from model_eval import Evaluater

# from env_carla import Environment
from ScenarioPlanner import ScenarioPlanner
from scenario_env import ScenarioEnvironment


from training import Training

from training import N_EPISODES
from training import TARGET_UPDATE
from training import EPS_START

# The learned Q value rates (state,action) pairs
# A CNN with a state input can rate possible actions, just as a classifier would

PREVIEW = False
VIDEO_EVERY = 1_000
PATH_MODEL = "model.pt"
PATH_SCENARIOS = "/disk/vanishing_data/is789/scenario_samples/Set_2022-12-14_00:59/"
IM_HEIGHT = 256
IM_WIDTH = 256

EGO_X = 246
EGO_Y = 128

evaluater = None

def main(withAE, concatAE):
    init_clearML(withAE, concatAE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu": print("!!! device is CPU !!!")

    if withAE:
        ae_model = AutoEncoder()
        evaluater = Evaluater(ae_model, device, PATH_MODEL)

    if withAE and not concatAE:
        DISTANCE_MATRIX = init_distance_matrices(EGO_X,EGO_Y)
        print(DISTANCE_MATRIX)

    writer = SummaryWriter()
#     env = Environment(host="tks-fly.fzi.de", port=2000)
    #env = ScenarioEnvironment(world="Town01_Opt", host="localhost", port=2100, s_width=256, s_height=256, cam_height=4.5, cam_rotation=-90, cam_zoom=130, random_spawn=False)
    settings = ScenarioPlanner.load_settings(PATH_SCENARIOS)
    env = ScenarioEnvironment(world=settings.world, host='localhost', port=2100, s_width=256, s_height=256, cam_height=4.5, cam_rotation=-90, cam_zoom=130)
    env.init_ego(car_type=settings.car_type)

    trainer = Training(writer, device, concatAE=concatAE)
    scenario_index = 0

    epsilon = EPS_START
    reward_best = -1000
    reward_per_episode_list = []
    duration_per_episode_list = []
    travel_dist_per_episode_list = []
    frames_per_episode_list = []
    spawn_point = None
    end_point = None

    for i in range(N_EPISODES):

        reward_per_episode = 0
        start = time.time()
        n_frame = 1

        env.reset(settings=settings.scenario_set[f"scenario_{scenario_index}"])
        # env.spawn_anomaly_alongRoad(max_numb=20)
        spawn_point = env.get_Vehicle_positionVec()

        obs_current = env.get_observation()
        obs_current = obs_current[0] #no segemntation

        if concatAE:
            heatMap = evaluater.getHeatMap(obs_current)
            heatMap = np.transpose(heatMap, (2,1,0))

        
        obs_current = np.transpose(obs_current, (2,1,0))
        if concatAE: obs_current = np.array([obs_current, heatMap])
        obs_current = np.array([obs_current])
        # print(obs_current.shape)
        obs_current = torch.as_tensor(obs_current)

        chw_list = []

        while True:
            if PREVIEW:
                cv2.imshow("", env.observation)
                cv2.waitKey(1)

#             chw = obs_current.squeeze(0)  # Remove batch information from BCHW
            chw_list.append(obs_current)

            # Perform action on observation and buildup replay memory

            if i % VIDEO_EVERY == 0:
                action = trainer.select_action(obs_current, 0)
            else:
                action = trainer.select_action(obs_current, epsilon)
            obs_next, reward, done, _ = env.step(action)
            obs_next = obs_next[0] #no segemntation

            if concatAE:
                heatMap = evaluater.getHeatMap(obs_next)
                heatMap = np.transpose(heatMap, (2,1,0))

            if withAE and not concatAE:
                detectionMap = evaluater.getDetectionMap(obs_next)
                reward = calcualte_enriched_reward(reward, detectionMap, DISTANCE_MATRIX)
                # print(reward)
                

            reward_per_episode += reward
            reward_torch = torch.tensor([reward], device=device)  # For compatibility with PyTorch

            if (
                time.time() - start
            ) > 30:  # Since the agent can simply stand now, the episode should terminate after 30 seconds
                done = True

            if done:
                reward_per_episode_list.append(reward_per_episode)
                end_point = env.get_Vehicle_positionVec()
                obs_next = None
            else:
                # if withAE:
                #     # heatmap = evaluater.getHeatMap(obs_next)
                #     detectionMap = evaluater.getheatMap(obs_next)
                #     obs_next = np.hstack((obs_next, detectionMap))

                obs_next = np.transpose(obs_next, (2,1,0))
                if concatAE: obs_next = np.array([obs_next, heatMap])
                obs_next = np.array([obs_next])
                obs_next = torch.as_tensor(obs_next)
            
            # Python tuples () https://www.w3schools.com/python/python_tuples.asp
            trainer.replay_memory.push(obs_current, action, obs_next, reward_torch, done)
            
            obs_current = obs_next

            # Optimization on policy model (I believe this could run in parallel to the data collection task)
            trainer.optimize(i)

            if done:
                end = time.time()
                duration = end - start
                duration_per_episode_list.append(duration)
                travel_dist = math.dist(spawn_point, end_point)
                travel_dist_per_episode_list.append(travel_dist)
                frames_per_episode_list.append(n_frame)

                reward_scalars = {
                    'Reward': reward_per_episode,
                    'avg_reward': np.average(reward_per_episode_list)
                }
                dist_scalars = {
                    'distance': travel_dist,
                    'avg_distance': np.average(travel_dist_per_episode_list)
                }
                duration_scalars = {
                    'duration': duration,
                    'avg_duration': np.average(duration_per_episode_list)
                }
                frame_scalars = {
                    'frames': n_frame,
                    'avg_frames': np.average(frames_per_episode_list)
                }
                writer.add_scalars("Reward", reward_scalars, i)
                writer.add_scalars("Distance", dist_scalars, i)
                writer.add_scalars("Duration", duration_scalars, i)
                writer.add_scalars("Frame", frame_scalars, i)

                if reward_per_episode > reward_best:
                    reward_best = reward_per_episode
                    name = "DQN Champ: "
                    save_video(chw_list, reward_best, i, writer, withAE, concatAE, name)
                    # tchw_list = torch.stack(chw_list)  # Adds "list" like entry --> TCHW
                    # tchw_list = torch.squeeze(tchw_list)
                    # name = "DQN Champ: " + str(reward_per_episode)
                    # writer.add_video(
                    #     tag=name, vid_tensor=tchw_list.unsqueeze(0), global_step=i
                    # )  # Unsqueeze adds batch --> BTCHW
                    torch.save(trainer.policy_net.state_dict(), os.path.join(gettempdir(), "dqn_" + str(i) + ".pt"))
                break

            n_frame += 1

        # Save video of episode to ClearML https://github.com/pytorch/pytorch/issues/33226
        if i % VIDEO_EVERY == 0:
            name = "DQN Agent: "
            save_video(chw_list, reward_best, i, writer, withAE, concatAE, name)

        # Update the target network, copying all weights and biases in DQN
        if i % TARGET_UPDATE == 0:
            trainer.target_net.load_state_dict(trainer.policy_net.state_dict())

        # Decay epsilon
        writer.add_scalar("Exploration-Exploitation/epsilon", epsilon, i)
        epsilon = trainer.decay_epsilon(epsilon)

        scenario_index += 1
        if scenario_index >= settings.size:
            scenario_index = 0
            print("Reseting scenario counter!")
        print(f"Episode: {i}")

    writer.flush()

# generate distance map from the center of the car to all other pixels in the space (works only in BEV)
# remains static for the whole training since the car in the image is never moving
def init_distance_matrices(pos_x, pos_y):
    size = IM_WIDTH
    ring_count = size # we want to double the size. each ring adds a size of 2 
    distance_matrix = np.zeros((1,1))
    for x in range(ring_count):
        distance_matrix = add_ring(distance_matrix, x + 1)


    max_distance = max(distance_matrix.flatten())

    distance_matrix = distance_matrix[size - pos_y: size - pos_y + size, size - pos_x: size - pos_x + size]
    distance_matrix

    max_distance_matrix = np.zeros((size, size)) + max_distance

    distance_matrix = distance_matrix / max_distance_matrix
    distance_matrix

    return distance_matrix

def calcualte_enriched_reward(reward, detectionMap, distanceMap):
    if reward == -1 : return -1 # collision or timeout

    rewardMap = detectionMap * distanceMap # element wise
    total_reward = np.sum(rewardMap)
    penalty = total_reward / (rewardMap.shape[0] * rewardMap.shape[1] - 1) # minus one, because the origion of the car should not be taken into count and is always zero
    reward = 1 - penalty# * 0.1
    reward = np.float32(reward)

    return reward



# given matrix a, adds a ring to it of the given value:
# a   --->    b-b-b
#             b-a-b
#             b-b-b
def add_ring(matrix, value):
    b = np.zeros(tuple(s+2 for s in matrix.shape), matrix.dtype) + value
    b[tuple(slice(1,-1) for s in matrix.shape)] = matrix
    return b


def save_video(chw_list, reward_best, step, writer, withVAE, concatAE, name):
    aug_list = []

    if concatAE:
        for stacked_img in chw_list:
            stacked_img = torch.squeeze(stacked_img)
            stacked_img = torch.tensor_split(stacked_img, 2, dim=0)
            observation = torch.squeeze(stacked_img[0])
            detectionMap = torch.squeeze(stacked_img[1]) # shape 3,w,h
            seperator = torch.zeros((3,2,256))
            seperator[0,:,1] = 1.
            aug_img = torch.hstack((observation, seperator, detectionMap))
            aug_list.append(aug_img)

    elif withVAE:
        for img in chw_list:
            img = img.numpy()
            img = np.squeeze(img)
            img = np.transpose(img, (2,1,0)) # shape: w,h,3
            detectionMap = evaluater.getColoredDetectionMap(img)
            detectionMap = color_pixel(detectionMap)
            seperator = np.zeros((256,2,3))
            seperator[:,:,0] = 1.
            aug_img = np.hstack((img, seperator, detectionMap))
            aug_img = np.transpose(aug_img, (2,1,0)) # shape: 3,w,h
            aug_img = torch.as_tensor(np.array([aug_img]))
            aug_list.append(aug_img)

    tchw_list = aug_list
    if not withAE and not concatAE: tchw_list = chw_list # when running in normal (no AE) mode
    
    tchw_list = torch.stack(tchw_list)  # Adds "list" like entry --> TCHW
    tchw_list = torch.squeeze(tchw_list)
    tchw_list = tchw_list.unsqueeze(0)
    # print(tchw_list.size())
    name = name + str(reward_best)
    writer.add_video(
        tag=name, vid_tensor=tchw_list, global_step=step
    )  # Unsqueeze adds batch --> BTCHW

def color_pixel(img):
    img[EGO_X+1, EGO_Y+1, 0] = 1.
    img[EGO_X+1, EGO_Y+1, 1] = 0.
    img[EGO_X+1, EGO_Y+1, 2] = 1.

    img[EGO_X+1, EGO_Y, 0] = 1.
    img[EGO_X+1, EGO_Y, 1] = 0.
    img[EGO_X+1, EGO_Y, 2] = 1.

    img[EGO_X, EGO_Y+1, 0] = 1.
    img[EGO_X, EGO_Y+1, 1] = 0.
    img[EGO_X, EGO_Y+1, 2] = 1.

    img[EGO_X-1, EGO_Y-1, 0] = 1.
    img[EGO_X-1, EGO_Y-1, 1] = 0.
    img[EGO_X-1, EGO_Y-1, 2] = 1.

    img[EGO_X-1, EGO_Y, 0] = 1.
    img[EGO_X-1, EGO_Y, 1] = 0.
    img[EGO_X-1, EGO_Y, 2] = 1.

    img[EGO_X, EGO_Y-1, 0] = 1.
    img[EGO_X, EGO_Y-1, 1] = 0.
    img[EGO_X, EGO_Y-1, 2] = 1.

    img[EGO_X+1, EGO_Y-1, 0] = 1.
    img[EGO_X+1, EGO_Y-1, 1] = 0.
    img[EGO_X+1, EGO_Y-1, 2] = 1.

    img[EGO_X-1, EGO_Y+1, 0] = 1.
    img[EGO_X-1, EGO_Y+1, 1] = 0.
    img[EGO_X-1, EGO_Y+1, 2] = 1.

    img[EGO_X, EGO_Y, 0] = 1.
    img[EGO_X, EGO_Y, 1] = 0.
    img[EGO_X, EGO_Y, 2] = 1.

    return img

def init_clearML(withAE, concatAE):
    name = "RL-"
    if concatAE: name = name + "Obs+Anomaly"
    elif withAE: name = name + "RichReward"
    else: name = name + "Baseline"

    task = Task.init(project_name="bogdoll/Anomaly_detection_Moritz", task_name=name, output_uri="s3://tks-zx.fzi.de:9000/clearml")
    task.set_base_docker(
            "nvcr.io/nvidia/pytorch:21.10-py3", 
            docker_setup_bash_script="apt-get update && apt-get install -y python3-opencv",
            docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all"  # --ipc=host",   
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)

    args = parser.parse_args()
    mode = args.mode

    if mode == "0":
        withAE = False
        concatAE = False
        print(f"~~~~~~~~~~~~~~\n### Mode: Baseline RL Agent! \n~~~~~~~~~~~~~~")
    elif mode == "1":
        withAE = True
        concatAE = False
        print(f"~~~~~~~~~~~~~~\n### Mode: Enriched Reward RL Agent \n~~~~~~~~~~~~~~")
    elif mode == "2":
        withAE = True
        concatAE = True
        print(f"~~~~~~~~~~~~~~\n### Mode: Observation + Anomaly RL Agent! \n~~~~~~~~~~~~~~")
    
    else:
        print("!!! Wrong mode flag. (0 = Baseline | 1 = Enriched Reward | 2 = Observation + Anomaly)")


    main(withAE, concatAE)
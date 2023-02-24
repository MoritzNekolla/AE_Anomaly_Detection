#!/usr/bin/env python
# coding: utf-8


from operator import truediv
import re
import cv2
import time
import torch
import numpy as np
import argparse
import math
from matplotlib import pyplot as plt

import os
import sys
from tempfile import gettempdir

from torch.utils.tensorboard import SummaryWriter

from clearml import Task, Dataset

from AE_model import AutoEncoder
from model_eval import Evaluater

# from env_carla import Environment
from ScenarioPlanner import ScenarioPlanner
from scenario_env import ScenarioEnvironment
from scenario_env import FIXED_DELTA_SECONDS


from training import Training

from training import N_EPISODES
from training import TARGET_UPDATE
from training import EPS_START

# The learned Q value rates (state,action) pairs
# A CNN with a state input can rate possible actions, just as a classifier would
HOST = "tks-iris.fzi.de"
# HOST = "localhost"
# HOST = "ids-ford.fzi.de"

PORT_LIST = [2200,2300,2400,2500]

PREVIEW = False
VIDEO_EVERY = 1_000
PATH_MODEL = "model.pt"
CLEARML_PATH_MODEL = "f93abef8342244859741408739724d18"
PATH_SCENARIOS = "/disk/vanishing_data/is789/scenario_samples/Set_2023-02-23_01:46/"
CLEARML_PATH_SCENARIOS = "8cf70bba1df24410bd0ded8ce45d05a1"
IM_HEIGHT = 256
IM_WIDTH = 256

EGO_X = 246
EGO_Y = 128


def main(withAE, concatAE, clearmlOn):
    port_list = PORT_LIST
    current_port = port_list.pop(0)
    day_count = time.time()
    task = init_clearML(withAE, concatAE, clearmlOn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {HOST} with port: {current_port}")
    if device == "cpu": print("!!! device is CPU !!!")

    evaluater = None
    if withAE:
        ae_model = AutoEncoder()

        if clearmlOn:
            path = Dataset.get(dataset_id=CLEARML_PATH_MODEL).get_local_copy()
            ae_model.to(device)
            ae_model.load_state_dict(torch.load(path + "/model.pt"))
            evaluater = Evaluater(ae_model, device)
        else:
            ae_model.to(device)
            ae_model.load_state_dict(torch.load(PATH_MODEL))
            evaluater = Evaluater(ae_model, device)


    if withAE and not concatAE:
        DISTANCE_MATRIX = init_distance_matrices(EGO_X,EGO_Y)
        print(DISTANCE_MATRIX)

    writer = SummaryWriter()
    # env = ScenarioEnvironment(host="tks-fly.fzi.de", port=2000)
    #env = ScenarioEnvironment(world="Town01_Opt", host="localhost", port=2100, s_width=256, s_height=256, cam_height=4.5, cam_rotation=-90, cam_zoom=130, random_spawn=False)
    if clearmlOn:
        path = Dataset.get(dataset_id=CLEARML_PATH_SCENARIOS).get_local_copy()
        settings = ScenarioPlanner.load_settings(path)
    else:
        settings = ScenarioPlanner.load_settings(PATH_SCENARIOS)

    env = ScenarioEnvironment(world=settings.world, host=HOST, port=current_port, s_width=256, s_height=256, cam_height=4.5, cam_rotation=-90, cam_zoom=130)
    env.init_ego(car_type=settings.car_type)

    trainer = Training(writer, device, concatAE=concatAE)
    scenario_index = 0

    epsilon = EPS_START
    reward_best = -1000
    reward_per_episode_list = []
    duration_per_episode_list = []
    travel_dist_per_episode_list = []
    frames_per_episode_list = []
    runtime_forwardpass_list = []
    spawn_point = None
    end_point = None

    for i in range(N_EPISODES):
        # if time.time() - day_count > 180: # 86.400 = 24 h
        #     print("Switching to new server...")
        #     day_count = time.time()
        #     env.deleteActors()
        #     del env
        #     env, current_port, port_list = conncet_to_carla(settings, port_list, current_port)
        #     env.init_ego(car_type=settings.car_type)
        # scenario_index = 4
        print(f"Episode: {i} | Scenario_index: {scenario_index}")
        reward_per_episode = 0
        n_frame = 1

        spawn_worked = False
        counter = 0
        while spawn_worked == False:
            scenario = settings.scenario_set[f"scenario_{scenario_index}"] # select new scenario
            _, spawn_worked = env.reset(settings=scenario)
            counter +=1
            if counter > 3:
                counter = 0
                print("Constantly failing to spawn vehicle... skipping scenario!!!")
                scenario_index += 1
                if scenario_index >= settings.size:
                    scenario_index = 0
                    print("Reseting scenario counter!")
                print(f"Now trying to run scenario {scenario_index}")
                

        start = time.time()
        # env.spawn_anomaly_alongRoad(max_numb=20)
        spawn_point = np.array([scenario.agent.spawn_point.location.x, scenario.agent.spawn_point.location.y, scenario.agent.spawn_point.location.z])
        goal_point = np.array([scenario.goal_point.location.x, scenario.goal_point.location.y, scenario.goal_point.location.z])

        obs_current = env.get_observation()
        minimap = env.createMiniMap()
        obs_current = obs_current[0] #no segemntation

        if concatAE:
            heatMap = evaluater.getHeatMap(obs_current)
            heatMap = np.transpose(heatMap, (2,0,1))

        
        obs_current = np.transpose(obs_current, (2,0,1))
        minimap = np.transpose(minimap, (2,0,1))
        # add heat and minimap
        if concatAE: obs_current = np.array([obs_current, heatMap, minimap])
        else : obs_current = np.array([obs_current, minimap])

        # # batch shape
        # obs_current = np.array([obs_current])
        # obs_current = torch.as_tensor(obs_current)

        chw_list = []
        r_fwd_list = []

        obs_current = torch.unsqueeze(torch.as_tensor(obs_current), 0)
        # t_0 = obs_current
        # t_1 = obs_current

        while True:
            fwdPass = time.time()
            # for video
            chw_list.append(obs_current)

            #Temporal stacking
            # obs_temporal = torch.cat((t_0, t_1, obs_current), dim=2)
            # print(obs_temporal.size())

            # batch shape
            # obs_current = torch.unsqueeze(obs_current, 0)
            # obs_current = np.array([obs_current])
            # obs_current = torch.as_tensor(obs_current)

            # Perform action on observation and buildup replay memory

            if i % VIDEO_EVERY == 0:
                # action = trainer.select_action(obs_temporal, 0)
                action = trainer.select_action(obs_current, 0)
            else:
                # action = trainer.select_action(obs_temporal, epsilon)
                action = trainer.select_action(obs_current, epsilon)

            env.tick_world()
            obs_next, reward, done, crashed, succeed = env.step(action)
            obs_next = obs_next[0] #no segemntation
            minimap = env.createMiniMap()

            if concatAE:
                heatMap = evaluater.getHeatMap(obs_next)
                heatMap = np.transpose(heatMap, (2,0,1))

            if withAE and not concatAE:
                detectionMap = evaluater.getDetectionMap(obs_next)
                rich_reward = calcualte_enriched_reward(reward, detectionMap, DISTANCE_MATRIX)
                reward = reward + rich_reward
                reward = float(reward)
                # print(np.dtype(reward))
                # print(reward)
                

            reward_per_episode += reward
            reward_torch = torch.tensor([reward], device=device)  # For compatibility with PyTorch

            if env.isTimeExpired():  # Since the agent can simply stand now, the episode should terminate after a given time
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

                obs_next = np.transpose(obs_next, (2,0,1))
                minimap = np.transpose(minimap, (2,0,1))
                # add minimap
                if concatAE: obs_next = np.array([obs_next, heatMap, minimap])
                else: obs_next = np.array([obs_next, minimap])
                # batch shape
                obs_next = torch.unsqueeze(torch.as_tensor(obs_next), 0)

                # Temporal stacking
                # obs_next_temporal = torch.cat((t_1, obs_current, obs_next), dim=2)
            
            # Python tuples () https://www.w3schools.com/python/python_tuples.asp
            # trainer.replay_memory.push(obs_temporal, action, obs_next_temporal, reward_torch, done)
            trainer.replay_memory.push(obs_current, action, obs_next, reward_torch, done)
            
            # t_0 = t_1
            # t_1 = obs_current
            obs_current = obs_next

            # Optimization on policy model (I believe this could run in parallel to the data collection task)
            trainer.optimize(i)

            # calc runtime
            runtime_end = time.time() - fwdPass
            r_fwd_list.append(runtime_end)

            if done:
                numReachPoints = env.getReachedPoints()
                end = time.time()
                duration = end - start
                duration_per_episode_list.append(duration)
                travel_dist = math.dist(spawn_point, end_point)
                travel_dist_per_episode_list.append(travel_dist)
                frames_per_episode_list.append(n_frame)
                runtime_forwardpass_list.append(np.average(r_fwd_list))

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
                crash_scalars = {
                    'crashed': crashed
                }
                succeed_scalars = {
                    'succed': succeed
                }
                reached_points = {
                    '# points': numReachPoints
                }
                runtime_scalars = {
                    'avg': np.average(runtime_forwardpass_list),
                    'runtime': np.average(r_fwd_list)
                }
                writer.add_scalars("Reward", reward_scalars, i)
                writer.add_scalars("Distance", dist_scalars, i)
                writer.add_scalars("Duration", duration_scalars, i)
                writer.add_scalars("Frame", frame_scalars, i)
                writer.add_scalars("Crashed", crash_scalars, i)
                writer.add_scalars("Succeed", succeed_scalars, i)
                writer.add_scalars("Reached_waypoints", reached_points, i)
                writer.add_scalars("Runtime_per_forward_pass", runtime_scalars, i)

                if reward_per_episode > reward_best:
                    reward_best = reward_per_episode
                    name = f"Best | Scenario_{scenario_index}: "
                    save_video(chw_list, reward_best, i, writer, withAE, concatAE, name, evaluater)
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
            save_video(chw_list, reward_best, i, writer, withAE, concatAE, name, evaluater)

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
    reward_result = 1 - penalty# * 0.1
    reward_result = np.float32(reward_result)

    return reward_result



# given matrix a, adds a ring to it of the given value:
# a   --->    b-b-b
#             b-a-b
#             b-b-b
def add_ring(matrix, value):
    b = np.zeros(tuple(s+2 for s in matrix.shape), matrix.dtype) + value
    b[tuple(slice(1,-1) for s in matrix.shape)] = matrix
    return b


def save_video(chw_list, reward_best, step, writer, withVAE, concatAE, name, evaluater):
    aug_list = []

    if concatAE:
        for stacked_img in chw_list:
            stacked_img = torch.squeeze(stacked_img)
            stacked_img = torch.tensor_split(stacked_img, 3, dim=0)
            observation = torch.squeeze(stacked_img[0])
            detectionMap = torch.squeeze(stacked_img[1]) # shape 3,w,h
            miniMap = torch.squeeze(stacked_img[2]) # add minimap
            seperator = torch.zeros((3,256,2))
            seperator[:,:,:] = 1.
            aug_img = torch.cat((observation, seperator, detectionMap, seperator, miniMap), dim=2)
            aug_list.append(aug_img)

    elif withVAE:
        for stacked_img in chw_list:
            stacked_img = torch.squeeze(stacked_img)
            stacked_img = torch.tensor_split(stacked_img, 2, dim=0)
            miniMap = torch.squeeze(stacked_img[1]) # add minimap
            observation = torch.squeeze(stacked_img[0])
            img = stacked_img[0]
            img = img.numpy()
            img = np.squeeze(img)
            img = np.transpose(img, (1,2,0)) # shape: w,h,3
            detectionMap = evaluater.getColoredDetectionMap(img)
            detectionMap = color_pixel(detectionMap)
            detectionMap = np.transpose(detectionMap, (2,0,1)) # shape: 3,w,h
            detectionMap = torch.as_tensor(detectionMap)
            seperator = np.zeros((3,256,2))
            seperator[:,:,:] = 1.
            seperator = torch.as_tensor(seperator)
            # aug_img = np.hstack((img, seperator, detectionMap))
            aug_img = torch.cat((observation, seperator, detectionMap, seperator, miniMap), dim=2)
            # aug_img = np.hstack((aug_img, seperator, miniMap))
            # aug_img = torch.as_tensor(np.array([aug_img]))
            aug_list.append(aug_img)
    else: # baseline
        for stacked_img in chw_list:
            stacked_img = torch.squeeze(stacked_img)
            stacked_img = torch.tensor_split(stacked_img, 2, dim=0)
            observation = torch.squeeze(stacked_img[0])
            miniMap = torch.squeeze(stacked_img[1]) # add minimap
            seperator = np.zeros((3,256,2))
            seperator[:,:,:] = 1.
            seperator = torch.as_tensor(seperator)
            aug_img =  torch.cat((observation, seperator, miniMap), dim=2)
            aug_list.append(aug_img)

    # # tchw_list = aug_list
    # # if not withAE and not concatAE: tchw_list = chw_list # when running in normal (no AE) mode
    # snapshot = aug_list[10].numpy()
    # # print(snapshot.shape)
    # snapshot = np.transpose(snapshot, (1,2,0))
    # # print(snapshot.shape)
    # plt.imsave("agent_snapshot.png", snapshot)

    tchw_list = torch.stack(aug_list)  # Adds "list" like entry --> TCHW
    tchw_list = torch.squeeze(tchw_list)
    tchw_list = tchw_list.unsqueeze(0)
    # print(tchw_list.size())
    name = name + str(reward_best)
    writer.add_video(
        tag=name, vid_tensor=tchw_list, global_step=step, fps=int(1/FIXED_DELTA_SECONDS),
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

def init_clearML(withAE, concatAE, clearmlOn):
    name = "RL-"
    if concatAE: name = name + "Obs+Anomaly"
    elif withAE: name = name + "RichReward"
    else: name = name + "Baseline"

    # Task.add_requirements("requirements.txt")
    Task.add_requirements("moviepy", "1.0.3")
    task = Task.init(project_name="bogdoll/Anomaly_detection_Moritz", task_name=name, output_uri="s3://tks-zx.fzi.de:9000/clearml")
    task.set_base_docker(
            "nvcr.io/nvidia/pytorch:21.10-py3", 
            docker_setup_bash_script="apt-get update && apt-get install -y python3-opencv",
            docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all"  # --ipc=host",   
            )
    
    parameters = {
        "scenario_path": PATH_SCENARIOS,
        "host": HOST
    }
    #start ClearML logging
    task.connect(parameters)
    logger = task.get_logger()
    if clearmlOn:
        task.execute_remotely('rtx3090', clone=False, exit_process=True) 

    return task

def conncet_to_carla(settings, port_list, current_port):
    env = None
    while env == None:
        for port in port_list:
            try:
                print(f"Trying to connect to {HOST} at port {port}...")
                env = ScenarioEnvironment(world=settings.world, host=HOST, port=port, s_width=256, s_height=256, cam_height=4.5, cam_rotation=-90, cam_zoom=130)
                current_port = port
                print(f"Connected. Continuing training!")
                break
            except:
                print(f"Connection failed!")
    
    new_port_list = []
    for port in PORT_LIST:
        if not port == current_port:
            new_port_list.append(port)
    
    return env, current_port, new_port_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--clearml", type=str, default=0)

    args = parser.parse_args()
    mode = args.mode
    clearml = args.clearml

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

    if clearml == "0":
        clearmlOn = False
    elif clearml == "1":
        clearmlOn = True
    else:
        print("!!! Wrong clearml flag. (0 = False | 1 = True)")
        
    main(withAE, concatAE, clearmlOn)
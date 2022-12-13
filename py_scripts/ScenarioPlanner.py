import glob
import os
import sys
import csv

import random
from tkinter import W
from turtle import pos
import numpy as np
import math
import json
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpllimg
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D
import cv2
from PIL import Image
from munch import DefaultMunch
import torch

from env_carla import Environment
from scenario_env import ScenarioEnvironment
from Utils import get_image_paths

IM_WIDTH = 2048
IM_HEIGHT = 2048
CAM_HEIGHT = 20.5
ROTATION = -70
CAM_OFFSET = 18.
ZOOM = 130
ROOT_STORAGE_PATH = "./scenario_sets/"
# MAP_SET = ["Town01_Opt", "Town02_Opt", "Town03_Opt", "Town04_Opt","Town05_Opt"]
MAP_SET = ["Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt","Town01_Opt","Town01_Opt"]


class ScenarioPlanner:

    def __init__(self, s_width=IM_WIDTH, s_height=IM_HEIGHT, cam_height=CAM_HEIGHT, cam_rotation=ROTATION, cam_zoom=ZOOM, cam_x_offset=10., host="localhost"):
        self.s_width = s_width
        self.s_height = s_height
        self.cam_height = cam_height
        self.cam_rotation = cam_rotation
        self.cam_zoom = cam_zoom
        self.cam_x_offset = cam_x_offset
        self.host = host

        self.world = "Town01_Opt"
        self.env = Environment(world=self.world, s_width=self.s_width, s_height=self.s_height, cam_height=self.cam_height, cam_rotation=self.cam_rotation,
                                 cam_zoom=self.cam_zoom, cam_x_offset=self.cam_x_offset, host=self.host, random_spawn=True)
        self.env.init_ego()

    def generateScenario(self, env):
        env.reset()
        anomaly_id, anomaly_transform = env.spawn_anomaly_alongRoad(max_numb=20)

        spawn_point_transform = env.getSpawnPoint()
        env.set_goalPoint(max_numb=30)
        goal_point = env.getGoalPoint()
        s_g_distance = spawn_point_transform.location.distance(goal_point.location)
        env.plotTrajectory()

        weather = env.get_Weather()
        snapshot, _ = env.get_observation()
        # plt.imshow(snapshot)
        # plt.show()
        # print(anomaly_transform)
        # print(spawn_point_transform)
        anomaly_point = {
            "location": {
                "x": anomaly_transform.location.x,
                "y": anomaly_transform.location.y,
                "z": anomaly_transform.location.z
            },
            "rotation": {
                "pitch": anomaly_transform.rotation.pitch,
                "yaw": anomaly_transform.rotation.yaw,
                "roll": anomaly_transform.rotation.roll
            }
        }
        spawn_point = {
            "location": {
                "x": spawn_point_transform.location.x,
                "y": spawn_point_transform.location.y,
                "z": spawn_point_transform.location.z
            },
            "rotation": {
                "pitch": spawn_point_transform.rotation.pitch,
                "yaw": spawn_point_transform.rotation.yaw,
                "roll": spawn_point_transform.rotation.roll
            }
        }
        goal_point = {
            "location": {
                "x": goal_point.location.x,
                "y": goal_point.location.y,
                "z": goal_point.location.z
            }
        }

        scenario_dict = {
            "anomaly": {
                "type": anomaly_id,
                "spawn_point": anomaly_point,

            },
            "agent": {
                "car_type": "vehicle.tesla.model3",
                "spawn_point": spawn_point
            },
            "goal_point": goal_point,
            "euc_distance": s_g_distance,
            "weather": weather
        }

        return scenario_dict, snapshot

    def sampleScenariosSet(self, amount):
        scenario_set = {}
        timestr = time.strftime("%Y-%m-%d_%H:%M")
        storagePath = self.create_Storage()


        for x in range(amount):
            # add to dict
            s_dict, snapshot = self.generateScenario(self.env)
            s_dict["snapshot"] = x
            scenario_set[f"scenario_{x}"] = s_dict
            
            # save snapshot
            pathToSnaps = storagePath + "snapshots/"
            if not os.path.isdir(pathToSnaps):
                os.mkdir(pathToSnaps)
            snapshot = (snapshot * 255).astype("int")
            cv2.imwrite(pathToSnaps + f"snap_{x}.png", snapshot)
        
        # destroy last env
        self.env.deleteActors()

        # type of set
        final_set = {
            "date": timestr,
            "size": amount,
            "world": self.world,
            "car_type": "vehicle.tesla.model3",
            "scenario_set": scenario_set
        }

        with open(storagePath + "scenario_set.json", "w") as fp:
            json.dump(final_set, fp, indent = 4)
    

    @staticmethod
    def create_snap_video(storagePath, max_scenes=20):
            path_list = get_image_paths(storagePath)
            tmp = cv2.imread(path_list[0])
            width = tmp.shape[0]
            height = tmp.shape[1]
            video = cv2.VideoWriter("walk.avi", 0, 1, (width ,height)) # width, height
            for x in range(len(path_list)):
                if x >= max_scenes: break
                path = path_list[x]
                video.write(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
            cv2.destroyAllWindows()
            return video.release()


# ==============================================================================
# -- Check loading  ------------------------------------------------------------
# ==============================================================================

    @staticmethod
    def createComparison(path):
        with open(path+'scenario_set.json') as json_file:
            settings = json.load(json_file)
            # convert to a dictionary that supports attribute-style access, a la JavaScript
            settings = DefaultMunch.fromDict(settings)

        snap_scenario_1 = cv2.imread(path+"snapshots/snap_0.png")

        env = ScenarioEnvironment(world=settings.world, host='localhost', port=2000, s_width=IM_WIDTH, s_height=IM_HEIGHT, cam_height=CAM_HEIGHT, cam_rotation=ROTATION, cam_zoom=ZOOM, cam_x_offset=CAM_OFFSET)
        env.init_ego(car_type=settings.car_type)
        env.reset(settings=settings.scenario_set.scenario_0)

        recreation_snapshot, _ = env.get_observation()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,12))
        ax1.set_title("scenario_snap")
        ax1.imshow(snap_scenario_1)
        ax2.set_title(f"recreation_snap")
        ax2.imshow(recreation_snapshot)

# ==============================================================================
# -- Utility methods -----------------------------------------------------------
# ==============================================================================

    # create Storage and return the path pointing towards it
    def create_Storage(self):
        if not os.path.isdir(ROOT_STORAGE_PATH):
            os.mkdir(ROOT_STORAGE_PATH)

        timestr = time.strftime("%Y-%m-%d_%H:%M")
        pathToStorage = ROOT_STORAGE_PATH + "Set_" + timestr + "/"

        if not os.path.isdir(pathToStorage):
            os.mkdir(pathToStorage)
        
        return pathToStorage

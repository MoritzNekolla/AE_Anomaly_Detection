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

import torch

from env_carla import Environment
from Utils import get_image_paths

IM_WIDTH = 2048
IM_HEIGHT = 2048
CAM_HEIGHT = 20
ROTATION = -90
ZOOM = 110
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


    def generateScenario(self):
        world = "Town01_Opt"
        env = Environment(world=world, s_width=self.s_width, s_height=self.s_height, cam_height=self.cam_height, cam_rotation=self.cam_rotation,
                                 cam_zoom=self.cam_zoom, cam_x_offset=self.cam_x_offset, host=self.host, random_spawn=True)
        env.init_ego()
        env.reset()
        anomaly_id, location = env.spawn_anomaly_alongRoad(max_numb=20)
        spawn_point = env.get_Vehicle_positionVec()
        env.set_goalPoint(max_numb=30)
        goal_point = env.getGoalPoint()
        s_g_distance = math.dist(spawn_point, goal_point)
        env.plotTrajectory()
        weather = env.get_Weather()
        snapshot, _ = env.get_observation()
        plt.imshow(snapshot)
        plt.show()
        anomaly_point = {
            "x": location.x,
            "y": location.y,
            "z": location.z
        }
        spawn_point = {
            "x": spawn_point[0],
            "y": spawn_point[1],
            "z": spawn_point[2]
        }
        goal_point = {
            "x": goal_point[0],
            "y": goal_point[1],
            "z": goal_point[2]
        }

        scenario_dict = {
            "type": anomaly_id,
            "world": world,
            "anomaly_point": anomaly_point,
            "spawn": spawn_point,
            "goal": goal_point,
            "euc_distance": s_g_distance,
            "weather": weather
        }

        env.deleteActors()
        return scenario_dict, snapshot

    def sampleScenariosSet(self, amount):
        scenario_set = {}
        timestr = time.strftime("%Y-%m-%d_%H:%M")
        storagePath = self.create_Storage()


        for x in range(amount):
            # add to dict
            s_dict, snapshot = self.generateScenario()
            s_dict["snapshot"] = x
            scenario_set[f"scenario_{x}"] = s_dict
            
            # save snapshot
            pathToSnaps = storagePath + "snapshots/"
            if not os.path.isdir(pathToSnaps):
                os.mkdir(pathToSnaps)
            snapshot = (snapshot * 255).astype("int")
            cv2.imwrite(pathToSnaps + f"snap_{x}.png", snapshot)

        # type of set
        final_set = {
            "date": timestr,
            "size": amount,
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
# -- Utility methods ------------------------------------------------------------
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

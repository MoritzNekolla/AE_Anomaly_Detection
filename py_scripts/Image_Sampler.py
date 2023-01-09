import glob
import os
import sys
import csv

import random
from tkinter import W
from turtle import pos
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpllimg
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D
import cv2
from SSIM_PIL import compare_ssim
from PIL import Image

import torch

from env_carla import Environment

IM_WIDTH = 480
IM_HEIGHT = 360
CAM_HEIGHT = 20
ROTATION = -90
ZOOM = 110
ROOT_STORAGE_PATH = "/disk/vanishing_data/is789/anomaly_samples/"
# MAP_SET = ["Town01_Opt", "Town02_Opt", "Town03_Opt", "Town04_Opt","Town05_Opt"]
MAP_SET = ["Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt", "Town01_Opt","Town01_Opt","Town01_Opt"]

DETECTION_THRESHOLD = 0.2

class Sampler:

    def __init__(self, s_width=IM_WIDTH, s_height=IM_HEIGHT, cam_height=CAM_HEIGHT, cam_rotation=ROTATION, cam_zoom=ZOOM, cam_x_offset=10., host="localhost"):
        self.s_width = s_width
        self.s_height = s_height
        self.cam_height = cam_height
        self.cam_rotation = cam_rotation
        self.cam_zoom = cam_zoom
        self.cam_x_offset = cam_x_offset
        self.host = host

    def get_waypoints(self, world_model=None, random_spawn=True):
        if world_model == None: world_model = MAP_SET[random.randrange(0,len(MAP_SET))]

        env = Environment(world=world_model, s_width=self.s_width, s_height=self.s_height, cam_height=self.cam_height,
                             cam_rotation=self.cam_rotation, cam_zoom=self.cam_zoom, host=self.host, random_spawn=random_spawn)
        env.init_ego()
        image, segmentation = env.reset()
        waypoints = env.getWaypoints()
        wp = env.getRelevantWaypoints()
        sl = env.single_lane(waypoints, wp.lane_id)
        env.deleteActors()
        return waypoints, wp, sl

    # turns carla segmentation into an image:
    # r-channel (first) represents the label
    def get_segmentation(self, obs):
        for x in range(len(obs)):
            for y in range(len(obs[x])):
                chunk = obs[x][y][2]
                if chunk == 20:
                    obs[x][y][2] = 0.98
                    obs[x][y][1] = 0.55
                    obs[x][y][0] = 0.32
                elif chunk == 7:
                    obs[x][y][:] = 0.4
                elif chunk == 8:
                    obs[x][y][:] = 0.65
                else:
                    obs[x][y][:] = 0.
        return obs
    
    # returns an image + its pixelwise target value (encoded in red channel)
    def sample(self, world_model=None, random_spawn=True):
        if world_model == None: world_model = MAP_SET[random.randrange(0,len(MAP_SET))]

        env = Environment(world=world_model, s_width=self.s_width, s_height=self.s_height, cam_height=self.cam_height,
                         cam_rotation=self.cam_rotation, cam_zoom=self.cam_zoom, host=self.host, random_spawn=random_spawn, cam_x_offset=self.cam_x_offset)
        env.init_ego()
        # image, segmentation = env.reset()
        env.reset()
        x = random.randrange(12,16)
        env.spawn_anomaly_ahead(distance=x)
        # env.spawn_anomaly_alongRoad(max_numb=6)
        # env.plotWaypoints()
        # time.sleep(20)
        time.sleep(3)
        image, segmentation = env.get_observation()
        env.deleteActors()
        return image, segmentation

    def sample_Ride(self, world_model=None, random_spawn=True, num_of_snaps=100, tick_rate=0.5, anomaly=False, anomaly_weather=False, anomalyDespawnDelay=None, random_action_prob=0.01):
        if world_model == None: world_model = MAP_SET[random.randrange(0,len(MAP_SET))]

        env = Environment(world=world_model, s_width=self.s_width, s_height=self.s_height, cam_height=self.cam_height, cam_rotation=self.cam_rotation, cam_zoom=self.cam_zoom, host=self.host, random_spawn=random_spawn)
        env.init_ego()
        env.setAutoPilot(True)
        env.reset()
        env.plotWaypoints()
        images = []
        segmentations = []
        anomalySpawnDelay = int(num_of_snaps / 2)
        anomalyObject = None
        for x in range(num_of_snaps):
            if anomaly and x == anomalySpawnDelay:
                anomalyObject = env.spawn_anomaly_ahead(distance=20)
            if anomaly and not anomalyDespawnDelay == None and x == anomalySpawnDelay + anomalyDespawnDelay:
                env.destroy_actor(anomalyObject)

            if anomaly_weather and x == anomalySpawnDelay:
                env.change_Weather()
            if anomaly_weather and not anomalyDespawnDelay == None and x == anomalySpawnDelay + anomalyDespawnDelay:
                env.reset_Weather()

            self.potentialStep(env, random_action_prob)

            image, segmentation = env.get_observation()
            images.append(image)
            segmentations.append(segmentation)
            time.sleep(tick_rate)
        env.deleteActors()

        # needs to be out of loop because render time would mess up a smooth video
        tmp = []
        if not segmentations[0] == None:
            for segment in segmentations:
                segment = self.get_segmentation(segment)
                tmp.append(segment)
            segmentations = tmp

        return images, segmentations

    def show_Example(self, random_spawn=True, segmentation=False):
        image, segment = self.sample(random_spawn=random_spawn)

        if segmentation:
            segment = self.get_segmentation(segment)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,6))
            ax1.set_title("Snapshot")
            ax1.imshow(image)
            ax2.set_title(f"Segmentation")
            ax2.imshow(segment)
        else:
            plt.figure(figsize=(6,6))
            plt.imshow(image)
            plt.show()


    def collect_Samples(self, sample_size, tick_rate=1):
        print(f"Starting to collect {sample_size} frames out of {len(MAP_SET)} worlds...")
        storagePath = self.create_Storage()
        samplesPerEnv = int(sample_size / len(MAP_SET))
        
        image_index = 0
        for x in range(len(MAP_SET)):
            images, _ = self.sample_Ride(world_model=MAP_SET[x], random_spawn=True, num_of_snaps=samplesPerEnv, tick_rate=tick_rate, anomaly=False, anomalyDespawnDelay=None)
            images = np.array(images)
            images = (images * 255).astype("int")
            print(f"finished world! {x}")
            for k in range(len(images)):
                cv2.imwrite(storagePath + f"snap_{image_index}.png", images[k]) 
                # plt.imsave(storagePath + f"snap_{image_index}.png",images[k], format="png")
                image_index = image_index + 1

        print(f"Finished | Collected: {str(samplesPerEnv * len(MAP_SET))} samples.")

    def potentialStep(self, env, proability):
        v = random.random()
        if v <= proability: env.makeRandomAction()

# ==============================================================================
# -- Collect huge amounts of samples (> 20k) --> save after each frame ---------
# ==============================================================================
    def collect_huge_Samples(self, sample_size, tick_rate=1):
        print(f"Starting to collect {sample_size} frames out of {len(MAP_SET)} worlds...")
        storagePath = self.create_Storage()
        samplesPerEnv = int(sample_size / len(MAP_SET))
        
        image_index = 0
        for x in range(len(MAP_SET)):
            image_index = self.sample_save_Ride(world_model=MAP_SET[x], random_spawn=True, num_of_snaps=samplesPerEnv, tick_rate=tick_rate, save_index=image_index, storagePath=storagePath)
            print(f"finished world! {x}")
        
        print(f"Finished | Collected: {str(samplesPerEnv * len(MAP_SET))} samples.")

    def sample_save_Ride(self, save_index, storagePath, world_model=None, random_spawn=True, num_of_snaps=100, tick_rate=1, random_action_prob=0.001):
        if world_model == None: world_model = MAP_SET[random.randrange(0,len(MAP_SET))]

        env = Environment(world=world_model, port=2000, s_width=self.s_width, s_height=self.s_height, cam_height=self.cam_height, cam_rotation=self.cam_rotation, cam_zoom=self.cam_zoom, host=self.host, random_spawn=random_spawn)
        env.init_ego()
        env.setAutoPilot(True)
        env.reset()
        pre_position = np.array([0.,0.,0.])

        x = 0
        while x < num_of_snaps:
            if x % 100 == 0 and not x == 0:
                env.reset()
                print(f"reseting env: {x}")
                pre_position = np.array([0.,0.,0.])
            
            self.potentialStep(env, random_action_prob)
            image,_ = env.get_observation()
            position = env.get_Vehicle_positionVec()
            # are we waiting at a red light ? >> ignore snap
            if Sampler.euclid_dist(pre_position, position) > .5 :
                image = (image * 255).astype("int")
                cv2.imwrite(storagePath + f"snap_{save_index}.png", image) 
                save_index += 1
                x = x + 1
                pre_position = np.array(position)
            rand_tick = random.random() * 3 + tick_rate
            env.tick_Seconds_world(rand_tick)
            # time.sleep(rand_tick)
        
        env.deleteActors()
        return save_index
# ==============================================================================
# -- End of huge sample code ---------------------------------------------------
# ==============================================================================

    # create Storage and return the path pointing towards it
    def create_Storage(self):
        if not os.path.isdir(ROOT_STORAGE_PATH):
            os.mkdir(ROOT_STORAGE_PATH)

        timestr = time.strftime("%Y-%m-%d_%H:%M:%S")
        pathToStorage = ROOT_STORAGE_PATH + "Samples_" + timestr + "/"

        if not os.path.isdir(pathToStorage):
            os.mkdir(pathToStorage)
        
        return pathToStorage

# ==============================================================================
# -- Video section -------------------------------------------------------------
# ==============================================================================

    def add_model_prediction(self, model, device, true_image, segmentation):
        true_image = np.transpose(true_image, (2,1,0))
        img = np.array([true_image])
        img = torch.as_tensor(img)
        img = img.to(device)
        out = model(img)
        out = out[0].detach().cpu().numpy()
        
        seperator1 = np.zeros((3,15,self.s_width)) #verticaly

        segmentation = np.transpose(segmentation, (2,1,0))

        prediction_img = np.concatenate((true_image, seperator1, out, seperator1, segmentation), axis=1)
        prediction_img = np.transpose(prediction_img, (2,1,0))

        return prediction_img, np.transpose(out, (2,1,0))
    

    def add_errormap(self, input, output):
        #1 grayscale
        g_input = np.dot(input[...,:3], [0.2989, 0.5870, 0.1140])
        g_output = np.dot(output[...,:3], [0.2989, 0.5870, 0.1140])
        errormap = abs(g_input-g_output)
        
        #2 heatmap
        hm = np.zeros((self.s_width,self.s_height,3))
        hm = hm.astype("float32")
        for x in range(len(hm)):
            for y in range(len(hm[x])):
                if errormap[x][y] < 0.5:
                    hm[x][y][2] = 1
                    hm[x][y][1] = 2*errormap[x][y]
                    hm[x][y][0] = 2*errormap[x][y]
                else:
                    hm[x][y][0] = 1
                    hm[x][y][1] = 1 - 2*(errormap[x][y] - 0.5)
                    hm[x][y][2] = 1 - 2*(errormap[x][y] - 0.5)

        #3 colorbar b,g,r
        bar = np.zeros((self.s_width - 62,15,3))
        ll = np.linspace(0,1,self.s_width - 62)
        for y in range(len(bar)):
            if ll[y] < 0.5:
                # bar[y,:,2] = 1
                # bar[y,:,1] = 2*ll[y]
                # bar[y,:,0] = 2*ll[y]
                bar[y,:,2] = 2*ll[y]
                bar[y,:,1] = 2*ll[y]
                bar[y,:,0] = 1
            else:
                # bar[y,:,2] = 1 - 2*(ll[y] - 0.5)
                # bar[y,:,1] = 1 - 2*(ll[y] - 0.5)
                # bar[y,:,0] = 1
                bar[y,:,2] = 1
                bar[y,:,1] = 1 - 2*(ll[y] - 0.5)
                bar[y,:,0] = 1 - 2*(ll[y] - 0.5)

        padd = np.zeros((31,15,3)) + 1.0
        bar = np.vstack((padd, bar, padd))

        #4 vstack
        padding = np.zeros((self.s_width,15,3)).astype("float32")
        padding1 = np.zeros((self.s_width,15,3)).astype("float32")
        padding1 = padding1 + 1.0
        padding2 = np.zeros((self.s_width,15,3)).astype("float32")
        padding2 = padding2 + 1.0
        heatmap = np.hstack((hm,padding, padding1, bar,padding2))

        #5 detectionMap
        detection_Map = self.get_detectionMap(errormap)

        return heatmap, detection_Map
    
    # colors the pixels that are above the avg error and returns the map
    def get_detectionMap(self, errormap):
        mu_err = np.average(errormap)
        flatten = errormap.reshape(-1)
        median, lowerSplit, upperSplit = self.get_median_set(flatten)
        median_Upper, _, upperSplit = self.get_median_set(upperSplit)
        median_Lower, _, upperSplit = self.get_median_set(lowerSplit)
        rangeIQR = median_Upper - median_Lower

        baseline = 5.5 * rangeIQR + median_Upper

        baseline = 0.5 # best for a MAE
        detectionMap = np.zeros((errormap.shape[0], errormap.shape[1], 3)).astype("float32")
        detectionMap = np.transpose(detectionMap, (2,1,0))
        detectionMap[0][errormap > baseline] = 0.98
        detectionMap[1][errormap > baseline] = 0.55
        detectionMap[2][errormap > baseline] = 0.32

        return np.transpose(detectionMap, (1,2,0))

    # returns median and the smaller, bigger array
    def get_median_set(self, values):
        median = np.median(values)
        values = list(values)
        upperSplit = []
        lowerSplit = []
        for value in values:
            if value > median:
                upperSplit.append(value)
            else:
                lowerSplit.append(value)
        
        return median, np.array(lowerSplit), np.array(upperSplit)

    def add_errorPlot(self, input, output, errorScores, ticks):
        errorMatrix = (input-output) ** 2
        errorAvg = np.sum(errorMatrix) / (errorMatrix.shape[0] * errorMatrix.shape[1] * errorMatrix.shape[2])
        errorAvg = int(errorAvg * 100000) / 100000.0
        errorScores.append(errorAvg)

        # im1 = Image.fromarray(np.uint8((input)*255))
        # im2 = Image.fromarray(np.uint8((output)*255))
        # value = 1 - compare_ssim(im1, im2) # 0 -> identical
        # errorScores.append(value)

        
        #4.27, 5.12
        fig = plt.figure(figsize=(2.11, self.s_width / 100.), dpi=100)
        # plt.xlim(-5, ticks + 20)
        # plt.ylim(-0.01, 0.3)
        plt.plot(errorScores, color="black", lw=2)
        plt.axhline(y=0.08, color='r', linestyle="solid", lw=0.75)
        # plt.title("Recon Error")
        # legend_elements = [Line2D([0], [0], color='#785EF0', label='Position in ', lw=1.5),
        #                     Line2D([0], [0], color='#DC267F', label='Violation', lw=1.5)]
        # plt.legend(handles=legend_elements, loc="upper left")
        # get image as np.array
        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        errorPlot = data.reshape(canvas.get_width_height()[::-1] + (3,))

        
        plt.close()
        return errorPlot, errorScores

    # 10 fps rendering
    def sample_canonicaly(self, model, device, seconds, anomaly, anomaly_weather, anomalyDespawnDelay, randomSpawn):
        seconds = seconds * 10
        anomalyDespawnDelay = anomalyDespawnDelay * 10
        images, segmentations = self.sample_Ride(world_model="Town01_Opt", num_of_snaps=seconds, tick_rate=0.1, anomaly=anomaly, anomaly_weather=anomaly_weather, anomalyDespawnDelay=anomalyDespawnDelay, random_spawn=randomSpawn)
        storagePath = "/disk/vanishing_data/is789/anomaly_samples/video_images/"
        path_list = Sampler.get_image_paths(storagePath)
        for path in path_list: #remove former runs
            os.remove(path)
        if not os.path.isdir(storagePath):
            os.mkdir(storagePath)
            

        tmp = images
        images = []
        errorScores = [0]
        for x in range(len(tmp)):
            image = tmp[x]
            segment = segmentations[x]
            firstLine, output = self.add_model_prediction(model, device, image, segment)
            heatmap, detectionMap = self.add_errormap(image, output)
            errorPlot, errorScores = self.add_errorPlot(image, output, errorScores, seconds)
            seperator_h = np.zeros((self.s_width,15,3))
            # deleteme = np.zeros((512,527,3)) ##Delete
            secondLine = np.hstack((heatmap, errorPlot, seperator_h, detectionMap))

            seperator_v = np.zeros((15,self.s_width*3 + 30,3))
            final_img = np.vstack((firstLine, seperator_v, secondLine))
            images.append(final_img)
            
        image_index = 0
        images = np.array(images)
        images = (images * 255).astype("int")
        for k in range(len(images)):
            fill_index = image_index
            if image_index < 10:
                fill_index = "00"+str(image_index)
            elif image_index < 100:
                fill_index = "0"+str(image_index)
            cv2.imwrite(storagePath + f"snap_{fill_index}.png", images[k])
            # plt.imsave(storagePath + f"snap_{image_index}.png",images[k], format="png")
            image_index = image_index + 1
        
        return storagePath
    
    def create_model_video(self, model, device, seconds=14, anomaly=True, anomaly_weather=False, anomalyDespawnDelay=None, randomSpawn=True):
        storagePath = self.sample_canonicaly(model, device, seconds, anomaly, anomaly_weather, anomalyDespawnDelay, randomSpawn)
        path_list = sorted(Sampler.get_image_paths(storagePath))
        video = cv2.VideoWriter("example_ride.avi", 0, 10, (self.s_width*3 + 30,self.s_width*2 + 15)) # width, height
        for path in path_list:
            video.write(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        cv2.destroyAllWindows()
        return video.release()



# ==============================================================================
# -- Static methods ------------------------------------------------------------
# ==============================================================================

    @staticmethod
    def euclid_dist(v, w):
        return ((v[0] - w[0])**2 + (v[1] - w[1])**2 + (v[2] - w[2])**2) ** 0.5

    # loads the set of images
    @staticmethod
    def load_Images(dir_path, size=9999999999):
        if size == None: size = 9999999999
        if size <= 0: size = 9999999999
        path_list = Sampler.get_image_paths(dir_path)
        img_list = []

        for x in range(len(path_list)):
            if x >= size: break
            path = path_list[x]
            img = cv2.imread(path)
            img_list.append(img)
        
        img_list = np.array(img_list) # .astype("float32")
        img_list = img_list[:,:,:,:3] # clear alpha channel
        print(f"Loaded {str(len(img_list))} images | width = {len(img_list[0])}, height = {len(img_list[0][0])}, channels = {len(img_list[0][0][0])}")
        return img_list


    # loads the set of images
    @staticmethod
    def sample_from_Set(img_list):
        size = len(img_list)
        index = random.randrange(0,size-1)


        plt.imshow(img_list[index])

        print(f"width = {len(img_list[index])}, height = {len(img_list[index][0])}, channels = {len(img_list[index][0][0])}")


    @staticmethod
    def get_image_paths(path):
        path_list = []

        for root, dirs, files in os.walk(os.path.abspath(path)):
            for file in files:
                path_list.append(os.path.join(root, file))
        return path_list



if __name__ == "__main__":
    # sampler = Sampler(s_width=512, s_height=512, cam_height=4, cam_zoom=50, cam_rotation=-18)
    # sampler = Sampler(s_width=256, s_height=256, cam_height=4, cam_zoom=50, cam_rotation=-12) # best 
    sampler = Sampler(s_width=256, s_height=256, cam_height=4.5, cam_zoom=130, cam_rotation=-90)
    # sampler.collect_Samples(sample_size=10, tick_rate=5)
    sampler.collect_huge_Samples(sample_size=40000, tick_rate=3)
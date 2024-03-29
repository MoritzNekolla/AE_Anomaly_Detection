##
# The environment for RL training. 
# Creates an environment given a settings object
# Always creates the same scenario for the same settings
##
import glob
import os
import sys

import random
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import cv2
import weakref

import torch
from Utils import plotToImage

IM_WIDTH = 256
IM_HEIGHT = 256

BEV_DISTANCE = 20

EPISODE_TIME = 30
N_ACTIONS = 9

RESET_SLEEP_TIME = 1

FACING_DEGREE = 90 # which direction the car is facing

FIXED_DELTA_SECONDS = 0.1
SUBSTEP_DELTA = 0.01
MAX_SUBSTEPS = 10

MAX_DEVIATION = 3.

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

class ScenarioEnvironment:

    def __init__(self, world=None, host='localhost', port=2000, s_width=IM_WIDTH, s_height=IM_HEIGHT,
                 cam_height=BEV_DISTANCE, cam_rotation=-90, cam_zoom=110, cam_x_offset=10.):
        weak_self = weakref.ref(self)
        self.client = carla.Client(host, port)            #Connect to server
        self.client.set_timeout(30.0)


        self.autoPilotOn = False

        self.world = self.client.load_world(world)

        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.goalPoint = None
        self.trajectory_list = None
        self.rotated_trajectory_list = None
        self.agent_transform = None

        self.s_width = s_width
        self.s_height = s_height
        self.cam_height = cam_height
        self.cam_rotation = cam_rotation
        self.cam_zoom = cam_zoom
        self.cam_x_offset = cam_x_offset

        self.anomaly_point = None

        self.actor_list = []
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT
        self.vehicle = None # important

        self.settings = None
        self.time_start = None
        
        # Synchronous mode + fixed time-step. The client will rule the simulation. The time step will be fixed. 
        # The server will not compute the following step until the client sends a tick. This is the best mode when synchrony and precision is relevant. 
        # Especially when dealing with slow clients or different elements retrieving information.
        w_settings = self.world.get_settings()
        w_settings.synchronous_mode = True
        w_settings.fixed_delta_seconds = FIXED_DELTA_SECONDS # 10 fps | fixed_delta_seconds <= max_substep_delta_time * max_substeps
        w_settings.substepping = True
        w_settings.max_substep_delta_time = SUBSTEP_DELTA
        w_settings.max_substeps = MAX_SUBSTEPS
        self.world.apply_settings(w_settings)
        self.fps_counter = 0
        self.max_fps = int(1/FIXED_DELTA_SECONDS) * EPISODE_TIME

        print(f"~~~~~~~~~~~~~~\n## Simulator settings ##\nFrames: {int(1/FIXED_DELTA_SECONDS)}\nSubstep_delta: {SUBSTEP_DELTA}\nMax_substeps: {MAX_SUBSTEPS}\n~~~~~~~~~~~~~~")


    def init_ego(self, car_type):
        self.vehicle_bp = self.bp_lib.find(car_type)
        self.ss_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.col_sensor_bp = self.bp_lib.find('sensor.other.collision')

        # Configure rgb sensors
        self.ss_camera_bp.set_attribute('image_size_x', f'{self.s_width}')
        self.ss_camera_bp.set_attribute('image_size_y', f'{self.s_height}')
        self.ss_camera_bp.set_attribute('fov', str(self.cam_zoom))

        # Location for both sensors
        self.ss_cam_location = carla.Location(self.cam_x_offset,0,self.cam_height)
        self.ss_cam_rotation = carla.Rotation(self.cam_rotation,0,0)
        self.ss_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)

        # collision sensor
        self.col_sensor_location = carla.Location(0,0,0)
        self.col_sensor_rotation = carla.Rotation(0,0,0)
        self.col_sensor_transform = carla.Transform(self.col_sensor_location, self.col_sensor_rotation)

        self.collision_hist = []



    def reset(self, settings):
        self.deleteActors()
        self.time_start = time.time()

        self.settings = settings

        self.actor_list = []
        self.collision_hist = []
        self.rotated_trajectory_list, self.rotation, self.roateted_agent_spawn, self.trajectory_list = self.loadTrajectory(settings.goal_trajectory)
        self.xAxis_min, self.xAxis_max, self.yAxis_min, self.yAxis_max = self.setAxis()

        # goal_point_list (for reward calculation)
        self.goalPointList = self.trajectory_list.copy()
        self.numReachedPoints = 0

        # Spawn vehicle
        a_location = carla.Location(self.settings.agent.spawn_point.location.x, self.settings.agent.spawn_point.location.y, self.settings.agent.spawn_point.location.z)
        a_rotation = carla.Rotation(self.settings.agent.spawn_point.rotation.pitch, self.settings.agent.spawn_point.rotation.yaw, self.settings.agent.spawn_point.rotation.roll)
        a_location.z += 0.3
        a_transform = carla.Transform(a_location, a_rotation)

        # spawn car. Note this can sometimes fail, therefore we iteratively try to spawn the car until it works
        counter = 0
        self.vehicle = None
        spawn_worked = True
        while self.vehicle == None:
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, a_transform)
            time.sleep(0.25)
            if counter > 10:
                print("Spawning vehicle error: Killed")
                print(f"Actors: {len(self.world.get_actors())}")
                spawn_worked = False
                break
            counter += 1
            
        if spawn_worked:
            self.vehicle.set_autopilot(self.autoPilotOn)
            self.actor_list.append(self.vehicle)

            # Attach and listen to image sensor (RGB)
            self.ss_cam = self.world.spawn_actor(self.ss_camera_bp, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
            self.actor_list.append(self.ss_cam)
            self.ss_cam.listen(lambda data: self.__process_sensor_data(data))

            # # Attach and listen to image sensor (Semantic Seg)
            # self.ss_cam_seg = self.world.spawn_actor(self.ss_camera_bp_sg, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
            # self.actor_list.append(self.ss_cam_seg)
            # self.ss_cam_seg.listen(lambda data: self.__process_sensor_data_Seg(data))

            # Attach and listen to collision sensor
            self.col_sensor = self.world.spawn_actor(self.col_sensor_bp, self.col_sensor_transform, attach_to=self.vehicle)
            self.actor_list.append(self.col_sensor)
            self.col_sensor.listen(lambda event: self.__process_collision_data(event))

            # spawn anomaly according to settings
            self.spawn_anomaly()

            # set weather according to settings
            self.set_Weather()
            
            # select goal_point according to settings
            self.set_goalPoint()

            self.tick_world(times=6)
            self.fps_counter = 0
            time.sleep(RESET_SLEEP_TIME)   # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

            self.episode_start = time.time()

            obs = self.get_observation()
        else:
            obs = None

        return obs, spawn_worked

    def step(self, action):
        # Easy actions: Steer left, center, right (0, 1, 2)
        action = 0
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1))
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1))
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-1))
        elif action == 8:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=1))

        # ego_transform = self.get_Vehicle_transform()
        ego_transform = self.agent_transform
        p_ego = np.array([ego_transform.location.x, ego_transform.location.y])
        # Get time
        # run_time = self.fps_counter * FIXED_DELTA_SECONDS
        # Get goal distance
        goal_distance = self.goalPoint.distance(ego_transform.location)

        
        # waypoint reward
        wp_reward = 0
        tmp_goalList = []
        for x in range(len(self.goalPointList)):
            gp = self.goalPointList[x]
            dist = np.linalg.norm(gp-p_ego)
            if dist > MAX_DEVIATION:
                tmp_goalList.append(gp)
            else:
                wp_reward += 1
                self.numReachedPoints += 1
        self.goalPointList = tmp_goalList.copy()

        # Get velocity of vehicle
        v = self.vehicle.get_velocity()
        v_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Set reward and 'done' flag
        done = False
        reward_collision = 0
        crashed = 0
        succeed = 0

        if len(self.collision_hist) != 0:
            done = True
            reward_collision = -1
            crashed = 1

        # velocity_reward = v_kmh / 40
        # if v_kmh > 40:
        #     # velocity_reward = v_kmh / (80 - 3*v_kmh)
        #     velocity_reward = -1

        # stay on road
        min_dist = 99999
        for x in range(len(self.trajectory_list)):
            gp = self.trajectory_list[x]
            distance_ego = np.linalg.norm(gp-p_ego)
            if distance_ego < min_dist:
                min_dist = distance_ego
                if min_dist < MAX_DEVIATION:
                    break

        out_of_map = 0
        if min_dist > MAX_DEVIATION:
            out_of_map = -1

        reward_total =  10*wp_reward + 200*reward_collision + 1*out_of_map - 0.1 # + reward_time + 0.*velocity_reward

        if goal_distance < 2.:
            done = True
            reward_total = 100
            succeed = 1

        return self.get_observation(), reward_total, done, crashed, succeed

    def spawn_anomaly(self):
        # select anomaly according to settings
        anomaly = self.bp_lib.filter(self.settings.anomaly.type)[0]
        if anomaly.has_attribute('is_invincible'):
            anomaly.set_attribute('is_invincible', 'false') 

        # spawn anomaly at specific point
        anomaly_location = carla.Location(self.settings.anomaly.spawn_point.location.x, self.settings.anomaly.spawn_point.location.y, self.settings.anomaly.spawn_point.location.z)
        anomaly_rotation = carla.Rotation(self.settings.anomaly.spawn_point.rotation.pitch, self.settings.anomaly.spawn_point.rotation.yaw, self.settings.anomaly.spawn_point.rotation.roll)
        anomaly_transform = carla.Transform(anomaly_location, anomaly_rotation)
        
        # spawn anomaly. Note this can sometimes fail, therefore we iteratively try to spawn the car until it works
        counter = 0
        player = None
        while player == None:
            player = self.world.try_spawn_actor(anomaly, anomaly_transform)
            if counter > 100:
                print("Spawning anomaly error: No anomaly this episode")
                print(f"Actors: {len(self.world.get_actors())}")
                break
            counter += 1

        self.actor_list.append(player)
        return player

    def set_goalPoint(self):
        location = carla.Location(self.settings.goal_point.location.x, self.settings.goal_point.location.y, self.settings.goal_point.location.z)
        self.latest_rotation = self.settings.agent.spawn_point.rotation.yaw
        self.goalPoint = location

    def set_Weather(self):
        self.weather = carla.WeatherParameters(
            cloudiness=self.settings.weather.cloudiness,
            precipitation=self.settings.weather.precipitation,
            precipitation_deposits=self.settings.weather.precipitation_deposits,
            wind_intensity=self.settings.weather.wind_intensity,
            sun_altitude_angle=self.settings.weather.sun_altitude_angle,
            fog_density=self.settings.weather.fog_density,
            wetness=self.settings.weather.wetness
        )
        self.world.set_weather(self.weather)

    def makeRandomAction(self):
        v = random.random()
        if v <= 0.33333333:
            self.step(0)
        elif v <= 0.6666666:
            self.step(1)
        elif v <= 1.0:
            self.step(2)

    #Returns only the waypoints in one lane
    def single_lane(self, waypoint_list, lane):
        waypoints = []
        for i in range(len(waypoint_list) - 1):
            if waypoint_list[i].lane_id == lane:
                waypoints.append(waypoint_list[i])
        return waypoints

    def destroy_actor(self, actor):
        actor.destroy()

    def isActorAlive(self, actor):
        if actor.is_alive:
            return True
        return False
    
    def setAutoPilot(self, value):
        self.autoPilotOn = value
        print(f"### Autopilot: {self.autoPilotOn}")

# ==============================================================================
# -- Minimap -------------------------------------------------------------------
# ==============================================================================
    def getDegree(self, root, point):
        vec = point - root
        length=np.linalg.norm(vec)
        radian = 0
        if vec[0] > 0 and vec[1] >= 0: radian=np.arctan(vec[1]/vec[0])
        elif vec[0] > 0 and vec[1] < 0: radian=np.arctan(vec[1]/vec[0]) + 2*np.pi
        elif vec[0] < 0: radian=np.arctan(vec[1]/vec[0]) + np.pi
        elif vec[0] == 0 and vec[1] > 0: radian=np.pi / 2
        elif vec[0] == 0 and vec[1] < 0: radian=3*np.pi / 2
        degree = (180/np.pi) * radian
        diff = degree - FACING_DEGREE
        result = FACING_DEGREE/(180/np.pi)
        rotated_point = np.array([length*math.cos(result),length*math.sin(result)])
        rotated_point = rotated_point + root
        return rotated_point, diff

    def rotate(self, root, point, rotation):
        vec = point - root
        length=np.linalg.norm(vec)
        radian = 0
        if vec[0] > 0 and vec[1] >= 0: radian=np.arctan(vec[1]/vec[0])
        elif vec[0] > 0 and vec[1] < 0: radian=np.arctan(vec[1]/vec[0]) + 2*np.pi
        elif vec[0] < 0: radian=np.arctan(vec[1]/vec[0]) + np.pi
        elif vec[0] == 0 and vec[1] > 0: radian=np.pi / 2
        elif vec[0] == 0 and vec[1] < 0: radian=3*np.pi / 2
        degree = (180/np.pi) * radian
        result = degree - rotation
        result = result/(180/np.pi)
        rotated_point = np.array([length*math.cos(result),length*math.sin(result)])
        rotated_point = rotated_point + root
        return rotated_point

    # load the regular waypoint trajectory from the settings file. (not a route that avoids the anomaly)
    def loadTrajectory(self, goal_trajectory):
        trajectory = []
        for x in range(len(goal_trajectory)):
            point = goal_trajectory[f"waypoint{x}"]
            point = np.array([point.x, point.y])
            trajectory.append(point)
        
        # discard every second point for computational acceleration
        tmp_list = []
        for x in range(0,len(trajectory),2):
            tmp_list.append(trajectory[x])

        if len(trajectory) % 2 == 0: tmp_list.append(trajectory[-1])
        trajectory = tmp_list

        trajectory = np.array(trajectory)
        old_trajectory = trajectory.copy()
        # carefull: mirroring the Y-axis to cope with carla coordinates (x=heading, y=rigth, z=up) 
        trajectory[:,1] = trajectory[:,1] * (-1)

        # rotate to ensure facing to the west
        p_agent = trajectory[0] # start and goal are contained in the waypoint list
        second_point, rotation = self.getDegree(p_agent, trajectory[1])
        trajectory_new = [p_agent, second_point]
        for point in trajectory[2:]:
            rotatet_point = self.rotate(p_agent, point, rotation)
            trajectory_new.append(rotatet_point)

        trajectory_new = np.array(trajectory_new)
        return trajectory_new, rotation, p_agent, old_trajectory
    
    def rotate_Map(self, anker, trajectory, rotation):
        tmp = []
        for point in trajectory:
            rotated_point = self.rotate(anker, point, rotation)
            tmp.append(rotated_point)
        
        return np.array(tmp)

    # create a total minimap
    def createMiniMap(self):
        # agent = self.get_Vehicle_transform()
        agent = self.agent_transform #synch with image
        p_agent = np.array([agent.location.x, agent.location.y])
        r_agent = agent.rotation.yaw
        # carefull: mirroring the Y-axis to cope with carla coordinates (x=heading, y=rigth, z=up) 
        p_agent[1] = p_agent[1] * (-1)


        rotation = (r_agent - self.latest_rotation) * (-1)
        # self.latest_rotation = r_agent
        
        p_agent = self.rotate(self.roateted_agent_spawn, p_agent, self.rotation + rotation)
        tra_rotated = self.rotate_Map(self.roateted_agent_spawn, self.rotated_trajectory_list, rotation)

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(IM_WIDTH/100, IM_HEIGHT/100), dpi=100)
        plt.axis("off")
        plt.plot(tra_rotated[:,0], tra_rotated[:,1], color="white", lw=8)
        plt.plot(p_agent[0], p_agent[1], color="blue", marker='^', markersize=20)
        plt.plot(tra_rotated[-1][0], tra_rotated[-1][1], color="red", marker='o', markersize=12)

        # set axis so that car starts in the middle
        # plt.xlim(self.xAxis_min, self.xAxis_max)
        # plt.ylim(self.yAxis_min, self.yAxis_max)

        y_dim = 20.5
        x_dim = (y_dim) / 2
        plt.xlim(p_agent[0] - x_dim, p_agent[0] + x_dim)
        plt.ylim(p_agent[1], p_agent[1] + y_dim)
        plt.tight_layout(pad=0.)

        miniMap = plotToImage(fig)
        plt.close()
        return miniMap.astype("float32") / 255

    # # create a total minimap
    # def createMiniMap(self):
    #     p_agent = self.get_Vehicle_positionVec()[:2]
    #     # carefull: mirroring the Y-axis to cope with carla coordinates (x=heading, y=rigth, z=up) 
    #     p_agent[1] = p_agent[1] * (-1) 
    #     p_agent = self.rotate(self.roateted_agent_spawn, p_agent, self.rotation)
    #     # self.trajectory_list = self.rotate_Map(p_agent, self.roa)
        


    #     plt.style.use('dark_background')
    #     fig = plt.figure(figsize=(IM_WIDTH/100, IM_HEIGHT/100), dpi=100)
    #     plt.axis("off")
    #     plt.plot(self.rotated_trajectory_list[:,0], self.rotated_trajectory_list[:,1], color="white", lw=3)
    #     plt.plot(p_agent[0], p_agent[1], color="blue", marker='<', markersize=12)
    #     plt.plot(self.rotated_trajectory_list[-1][0], self.rotated_trajectory_list[-1][1], color="red", marker='o', markersize=12)

    #     # set axis so that car starts in the middle
    #     plt.xlim(self.xAxis_min, self.xAxis_max)
    #     plt.ylim(self.yAxis_min, self.yAxis_max)

    #     miniMap = plotToImage(fig)
    #     plt.close()
    #     return miniMap.astype("float32") / 255

    #     # create a total minimap
    # # def createMiniMap(self):
    # #     p_agent = self.get_Vehicle_positionVec()[:2]
    # #     agetn_transform = self.get_Vehicle_transform()
    # #     fwdVector = agetn_transform.rotation.get_forward_vector()



    # #     plt.style.use('dark_background')
    # #     fig = plt.figure(figsize=(IM_WIDTH/100, IM_HEIGHT/100), dpi=100)
    # #     plt.axis("off")
    # #     plt.plot(self.trajectory_list[:,0], self.trajectory_list[:,1], color="white", lw=3)
    # #     plt.plot(p_agent[0], p_agent[1], color="blue", marker='<', markersize=12)
    # #     plt.plot(self.trajectory_list[-1][0], self.trajectory_list[-1][1], color="red", marker='o', markersize=12)

    # #     # set axis so that car starts in the middle
    # #     plt.xlim(self.xAxis_min, self.xAxis_max)
    # #     plt.ylim(self.yAxis_min, self.yAxis_max)

    # #     miniMap = plotToImage(fig)
    # #     plt.close()
    # #     return miniMap.astype("float32") / 255

    # set axis so that car starts in the middle
    def setAxis(self):
        x_minimum = min(self.rotated_trajectory_list[:,0])
        x_maximum = max(self.rotated_trajectory_list[:,0])
        x_dist_min = abs(self.roateted_agent_spawn[0] - x_minimum)
        x_dist_max = abs(self.roateted_agent_spawn[0] - x_maximum)
        if x_dist_min < 2.:
            x_minimum = self.roateted_agent_spawn[0] - 2
            x_dist_min = 2.
        if x_dist_max < 2.:
            x_maximum = self.roateted_agent_spawn[0] + 2
            x_dist_max = 2
        if x_dist_max > x_dist_min:
            x_end = x_maximum
            x_start = self.roateted_agent_spawn[0] - x_dist_max
        else:
            x_start = x_minimum
            x_end = self.roateted_agent_spawn[0] + x_dist_min
        
        x_start = x_start - 5.5
        x_end = x_end + 5.5
        y_start = min(self.rotated_trajectory_list[:,1]) - 3
        y_end = max(self.rotated_trajectory_list[:,1]) + 3

        return x_start, x_end, y_start, y_end
        
# ==============================================================================
# -- Getter --------------------------------------------------------------------
# ==============================================================================
    
    def getReachedPoints(self):
        return self.numReachedPoints

    def getGoalTrajectory(self):
        return self.trajectory_list
    
    def getGoalPoint(self):
        return self.goalPoint

    def get_Weather(self):
        wheather = self.world.get_weather()
        w_dict = {
            "cloudiness": wheather.cloudiness,
            "precipitation": wheather.precipitation,
            "precipitation_deposits": wheather.precipitation_deposits,
            "wind_intensity": wheather.wind_intensity,
            # "sun_azimuth_angle": wheather.sun_azimuth_angle,
            "sun_altitude_angle": wheather.sun_altitude_angle,
            "fog_density": wheather.fog_density,
            # "fog_distance": wheather.fog_distance,
            # "fog_falloff": wheather.fog_falloff,
            "wetness": wheather.wetness
            # "scattering_intensity": wheather.scattering_intensity,
            # "mie_scattering_scale": wheather.mie_scattering_scale,
            # "rayleigh_scattering_scale": wheather.rayleigh_scattering_scale
        }
        return w_dict

    def getEgoWaypoint(self):
        # vehicle_loc = self.vehicle.get_location()
        vehicle_loc = self.agent_transform.location
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True,
                      lane_type=carla.LaneType.Driving)

        return wp
    
    def getWaypoints(self):
        return self.map_waypoints

    #get vehicle location and rotation (0-360 degrees)
    def get_Vehicle_transform(self):
        return self.vehicle.get_transform()

    #get vehicle location
    def get_Vehicle_positionVec(self):
        position = self.vehicle.get_transform().location
        return np.array([position.x, position.y, position.z])

    def getFPS_Counter(self):
        return self.fps_counter

    def isTimeExpired(self):
        if self.fps_counter > self.max_fps:
            return True
        return False

# ==============================================================================
# -- Sensor processing ---------------------------------------------------------
# ==============================================================================

    # perform a/multiple world tick
    def tick_world(self, times=1):
        for x in range(times):
            self.world.tick()
            self.fps_counter += 1

    def get_observation(self):
        """ Observations in PyTorch format BCHW """
        frame = self.observation
        self.agent_transform = self.get_Vehicle_transform()
        frame = frame.astype(np.float32) / 255
        frame = self.arrange_colorchannels(frame)

        return frame,None

    def __process_sensor_data(self, image):
        """ Observations directly viewable with OpenCV in CHW format """
        # image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i2 = i.reshape((self.s_height, self.s_width, 4))
        i3 = i2[:, :, :3]
        self.observation = i3

    def __process_collision_data(self, event):
        self.collision_hist.append(event)

    # changes order of color channels. Silly but works...
    def arrange_colorchannels(self, image):
        mock = image.transpose(2,1,0)
        tmp = []
        tmp.append(mock[2])
        tmp.append(mock[1])
        tmp.append(mock[0])
        tmp = np.array(tmp)
        tmp = tmp.transpose(2,1,0)
        return tmp

    def exit_env(self):
        self.deleteEnv()
    
    def deleteActors(self):
        if not self.vehicle == None:
            self.vehicle.set_autopilot(False)

        for actor in self.actor_list:
            if not actor == None:
                actor.destroy()   

    def __del__(self):
        print("__del__ called")
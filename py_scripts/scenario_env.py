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

FIXED_DELTA_SECONDS = 0.05
SUBSTE_DELTA = 0.007
MAX_SUBSTEPS = 10

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

    def __init__(self, world=None, host='localhost', port=2000, s_width=IM_WIDTH, s_height=IM_HEIGHT, cam_height=BEV_DISTANCE, cam_rotation=-90, cam_zoom=110, cam_x_offset=10.):
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
        w_settings.max_substep_delta_time = SUBSTE_DELTA
        w_settings.max_substeps = MAX_SUBSTEPS
        self.world.apply_settings(w_settings)
        self.fps_counter = 0
        self.max_fps = int(1/FIXED_DELTA_SECONDS) * EPISODE_TIME

        print(f"~~~~~~~~~~~~~~\n## Simulator settings ##\nFrames: {int(1/FIXED_DELTA_SECONDS)}\nSubstep_delta: {SUBSTE_DELTA}\nMax_substeps: {MAX_SUBSTEPS}\n~~~~~~~~~~~~~~")


    def init_ego(self, car_type):
        self.vehicle_bp = self.bp_lib.find(car_type)
        self.ss_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        # self.ss_camera_bp_sg = self.bp_lib.find('sensor.camera.semantic_segmentation')
        self.col_sensor_bp = self.bp_lib.find('sensor.other.collision')

        # Configure rgb sensors
        self.ss_camera_bp.set_attribute('image_size_x', f'{self.s_width}')
        self.ss_camera_bp.set_attribute('image_size_y', f'{self.s_height}')
        self.ss_camera_bp.set_attribute('fov', str(self.cam_zoom))

        # Location for both sensors
        self.ss_cam_location = carla.Location(self.cam_x_offset,0,self.cam_height)
        self.ss_cam_rotation = carla.Rotation(self.cam_rotation,0,0)
        self.ss_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)

        # # Configure segmantic sensors
        # self.ss_camera_bp_sg.set_attribute('image_size_x', f'{self.s_width}')
        # self.ss_camera_bp_sg.set_attribute('image_size_y', f'{self.s_height}')
        # self.ss_camera_bp_sg.set_attribute('fov', str(self.cam_zoom))
        
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

        # Spawn vehicle
        a_location = carla.Location(self.settings.agent.spawn_point.location.x, self.settings.agent.spawn_point.location.y, self.settings.agent.spawn_point.location.z)
        a_rotation = carla.Rotation(self.settings.agent.spawn_point.rotation.pitch, self.settings.agent.spawn_point.rotation.yaw, self.settings.agent.spawn_point.rotation.roll)
        a_transform = carla.Transform(a_location, a_rotation)

        self.vehicle = self.world.spawn_actor(self.vehicle_bp, a_transform)
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

        self.tick_world(times=5)
        self.fps_counter = 0
        time.sleep(RESET_SLEEP_TIME)   # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        self.episode_start = time.time()
        return self.get_observation()

    def step(self, action):
        # Easy actions: Steer left, center, right (0, 1, 2)
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

        # Get time
        run_time = self.fps_counter * FIXED_DELTA_SECONDS
        # Get goal distance
        goal_distance = self.goalPoint.distance(self.get_Vehicle_transform().location)
        # Get velocity of vehicle
        # v = self.vehicle.get_velocity()
        # v_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Set reward and 'done' flag
        done = False
        reward_collision = 0
        crashed = 0
        succeed = 0

        if len(self.collision_hist) != 0:
            done = True
            reward_collision = -100
            crashed = 1

        # elif v_kmh < 20:
        #     reward = v_kmh / (80 - 3*v_kmh)
        # else:
        #     reward = 1

        reward_time = (EPISODE_TIME - run_time)/ EPISODE_TIME
        reward_distance = (self.settings.euc_distance - goal_distance) / self.settings.euc_distance

        reward_total = reward_time + 2*reward_distance + reward_collision

        if goal_distance < 2.:
            done = True
            reward_total = 100
            succeed = 1

        return self.get_observation(), reward_total, done, crashed, succeed

    def spawn_anomaly(self):
        # select anomaly according to settings
        anomaly = self.bp_lib.filter(self.settings.anomaly.type)[0]

        # spawn anomaly at specific point
        anomaly_location = carla.Location(self.settings.anomaly.spawn_point.location.x, self.settings.anomaly.spawn_point.location.y, self.settings.anomaly.spawn_point.location.z)
        anomaly_rotation = carla.Rotation(self.settings.anomaly.spawn_point.rotation.pitch, self.settings.anomaly.spawn_point.rotation.yaw, self.settings.anomaly.spawn_point.rotation.roll)
        anomaly_transform = carla.Transform(anomaly_location, anomaly_rotation)
        player = self.world.try_spawn_actor(anomaly, anomaly_transform)

        self.actor_list.append(player)
        return player

    def set_goalPoint(self):
        location = carla.Location(self.settings.goal_point.location.x, self.settings.goal_point.location.y, self.settings.goal_point.location.z)
        self.goalPoint = location

    def set_Weather(self):
        self.weather = carla.WeatherParameters(
            cloudiness=self.settings.weather.cloudiness,
            precipitation=self.settings.weather.precipitation,
            precipitation_deposits=self.settings.weather.precipitation_deposits,
            wind_intensity=self.settings.weather.wind_intensity,
            # sun_azimuth_angle=self.settings.weather.sun_azimuth_angle,
            sun_altitude_angle=self.settings.weather.sun_altitude_angle,
            fog_density=self.settings.weather.fog_density,
            # fog_distance=self.settings.weather.fog_distance,
            # fog_falloff=self.settings.weather.fog_falloff,
            wetness=self.settings.weather.wetness
            # scattering_intensity=self.settings.weather.scattering_intensity,
            # mie_scattering_scale=self.settings.weather.mie_scattering_scale,
            # rayleigh_scattering_scale=self.settings.weather.rayleigh_scattering_scale
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
    
    # def plotWaypoints(self):
    #     vehicle_loc = self.vehicle.get_location()

    #     self.world.debug.draw_string(vehicle_loc, str("Hallo"), draw_shadow=False, life_time=-1)

    #     for w in self.map_waypoints:
    #         # self.world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
    #         #                                 color=carla.Color(r=255, g=0, b=0), life_time=-1,
    #         #                                 persistent_lines=True)
    #         # print(w.transform.location)
    #         self.world.debug.draw_point(w.transform.location, size=0.05, life_time=-1., color=carla.Color(r=255, g=0, b=0))

    #     wp = self.map.get_waypoint(vehicle_loc, project_to_road=True,
    #             lane_type=carla.LaneType.Driving)
        
    #     self.world.debug.draw_point(wp.transform.location, size=0.05, life_time=-1., color=carla.Color(r=0, g=0, b=255))


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
        
        trajectory = np.array(trajectory)
        old_trajectory = trajectory
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

    # create a total minimap
    def createMiniMap(self):
        p_agent = self.get_Vehicle_positionVec()[:2]
        # carefull: mirroring the Y-axis to cope with carla coordinates (x=heading, y=rigth, z=up) 
        p_agent[1] = p_agent[1] * (-1) 
        p_agent = self.rotate(self.roateted_agent_spawn, p_agent, self.rotation)


        plt.style.use('dark_background')
        fig = plt.figure(figsize=(IM_WIDTH/100, IM_HEIGHT/100), dpi=100)
        plt.axis("off")
        plt.plot(self.rotated_trajectory_list[:,0], self.rotated_trajectory_list[:,1], color="white", lw=3)
        plt.plot(p_agent[0], p_agent[1], color="blue", marker='<', markersize=12)
        plt.plot(self.rotated_trajectory_list[-1][0], self.rotated_trajectory_list[-1][1], color="red", marker='o', markersize=12)

        # set axis so that car starts in the middle
        plt.xlim(self.xAxis_min, self.xAxis_max)
        plt.ylim(self.yAxis_min, self.yAxis_max)

        miniMap = plotToImage(fig)
        plt.close()
        return miniMap.astype("float32") / 255

    #     # create a total minimap
    # def createMiniMap(self):
    #     p_agent = self.get_Vehicle_positionVec()[:2]
    #     agetn_transform = self.get_Vehicle_transform()
    #     fwdVector = agetn_transform.rotation.get_forward_vector()



    #     plt.style.use('dark_background')
    #     fig = plt.figure(figsize=(IM_WIDTH/100, IM_HEIGHT/100), dpi=100)
    #     plt.axis("off")
    #     plt.plot(self.trajectory_list[:,0], self.trajectory_list[:,1], color="white", lw=3)
    #     plt.plot(p_agent[0], p_agent[1], color="blue", marker='<', markersize=12)
    #     plt.plot(self.trajectory_list[-1][0], self.trajectory_list[-1][1], color="red", marker='o', markersize=12)

    #     # set axis so that car starts in the middle
    #     plt.xlim(self.xAxis_min, self.xAxis_max)
    #     plt.ylim(self.yAxis_min, self.yAxis_max)

    #     miniMap = plotToImage(fig)
    #     plt.close()
    #     return miniMap.astype("float32") / 255

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
        vehicle_loc = self.vehicle.get_location()
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
        frame = frame.astype(np.float32) / 255
        frame = self.arrange_colorchannels(frame)

        # seg = self.observation_seg
        # seg = seg.astype(np.float32)
        # seg = self.arrange_colorchannels(seg)
        # return frame, seg
        return frame,None

    def __process_sensor_data(self, image):
        """ Observations directly viewable with OpenCV in CHW format """
        # image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i2 = i.reshape((self.s_height, self.s_width, 4))
        i3 = i2[:, :, :3]
        self.observation = i3

    # def __process_sensor_data_Seg(self, image):
    #     """ Observations directly viewable with OpenCV in CHW format """
    #     # image.convert(carla.ColorConverter.CityScapesPalette)
    #     i = np.array(image.raw_data)
    #     i2 = i.reshape((self.s_height, self.s_width, 4))
    #     i3 = i2[:, :, :3]
    #     self.observation_seg = i3

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
            actor.destroy()       
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def __del__(self):
        print("__del__ called")
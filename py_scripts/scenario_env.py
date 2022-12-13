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

IM_WIDTH = 256
IM_HEIGHT = 256

BEV_DISTANCE = 20

N_ACTIONS = 9

RESET_SLEEP_TIME = 1

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

        # traffic_manager = self.client.get_trafficmanager(port)
        # traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        # traffic_manager.set_respawn_dormant_vehicles(True)
        # traffic_manager.set_synchronous_mode(True)

        self.autoPilotOn = False

        self.world = self.client.load_world(world)

        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.goalPoint = None
        self.trajectory_list = None

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
        
        # Todo: set settings
        self.settings = settings

        self.actor_list = []
        self.collision_hist = []

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

        time.sleep(RESET_SLEEP_TIME)   # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

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


        # Get velocity of vehicle
        v = self.vehicle.get_velocity()
        v_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Set reward and 'done' flag
        done = False
        if len(self.collision_hist) != 0:
            done = True
            reward = -100
        elif v_kmh < 20:
            reward = v_kmh / (80 - 3*v_kmh)
        else:
            reward = 1

        return self.get_observation(), reward, done, None

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
            sun_azimuth_angle=self.settings.weather.sun_azimuth_angle,
            sun_altitude_angle=self.settings.weather.sun_altitude_angle,
            fog_density=self.settings.weather.fog_density,
            fog_distance=self.settings.weather.fog_distance,
            fog_falloff=self.settings.weather.fog_falloff,
            wetness=self.settings.weather.wetness,
            scattering_intensity=self.settings.weather.scattering_intensity,
            mie_scattering_scale=self.settings.weather.mie_scattering_scale,
            rayleigh_scattering_scale=self.settings.weather.rayleigh_scattering_scale
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

    def plotTrajectory(self):
        for w in self.trajectory_list:
            self.world.debug.draw_point(w.transform.location, size=0.2, life_time=-1., color=carla.Color(r=0, g=0, b=255))
        # color goal point in red   
        self.world.debug.draw_point(self.trajectory_list[-1].transform.location, size=0.3, life_time=-1., color=carla.Color(r=255, g=140, b=0))

        # color anomaly
        bounding_box_set = self.world.get_level_bbs(carla.CityObjectLabel.Dynamic) + self.world.get_level_bbs(carla.CityObjectLabel.Static)
        best = 100000.
        bbox = None
        for box in bounding_box_set:
            distance = box.location.distance(self.anomaly_point.get_transform().location)
            if distance < best:
                best = distance
                bbox = box
        # self.world.debug.draw_box(carla.BoundingBox(self.anomaly_point.get_transform().location,carla.Vector3D(3.5,3.5,4)),self.anomaly_point.get_transform().rotation, 0.3, carla.Color(255,140,0,0),-1.)
        self.world.debug.draw_box(bbox, self.anomaly_point.get_transform().rotation, 0.2, carla.Color(0,0,0,0),-1.)
        time.sleep(2)

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
# -- Getter --------------------------------------------------------------------
# ==============================================================================
    def getGoalPoint(self):
        return self.goalPoint

    def get_Weather(self):
        wheather = self.world.get_weather()
        w_dict = {
            "cloudiness": wheather.cloudiness,
            "precipitation": wheather.precipitation,
            "precipitation_deposits": wheather.precipitation_deposits,
            "wind_intensity": wheather.wind_intensity,
            "sun_azimuth_angle": wheather.sun_azimuth_angle,
            "sun_altitude_angle": wheather.sun_altitude_angle,
            "fog_density": wheather.fog_density,
            "fog_distance": wheather.fog_distance,
            "fog_falloff": wheather.fog_falloff,
            "wetness": wheather.wetness,
            "scattering_intensity": wheather.scattering_intensity,
            "mie_scattering_scale": wheather.mie_scattering_scale,
            "rayleigh_scattering_scale": wheather.rayleigh_scattering_scale
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

# ==============================================================================
# -- Sensor processing ---------------------------------------------------------
# ==============================================================================
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
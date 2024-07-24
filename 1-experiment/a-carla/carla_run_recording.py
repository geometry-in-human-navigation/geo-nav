# -*- coding: utf-8 -*-
import carla
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from datetime import datetime
from queue import Queue
from queue import Empty
import os
from scipy.io import savemat
import time
import sys
from scipy.io import loadmat

from agents.navigation.behavior_agent import BehaviorAgent

date = datetime.now().strftime("%Y_%m_%d")
global vehicle

def select_town_weather(selected_town_index, selected_weather_index):
    global selected_towns
    global selected_weathers
    global selected_town
    global selected_weather
    global selected_map_unloadlayers
    global output_path
    
    selected_towns =[
        "Town01",
        "Town02",
        "Town03",
        "Town04",
        "Town05",
        "Town06",
        "Town07",    
        "Town10HD",    
    ]
    selected_town = selected_towns[selected_town_index]

    selected_weathers = [
        "CloudyNoon",
        "ClearNight",
        "WetSunset",
        "WetCloudyNoon",
        "ClearNoon",
        "ClearSunset",
        "HardRainSunset",
        "FogMorning",
    ]

    selected_weather = selected_weathers[selected_weather_index]

    selected_map_unloadlayers = 0

    # create directory for outputs
    global output_path_base
    output_path_base = '/media/statespace/Spatial/sptialworkspace/spatialfMRI/simulation/carla_ws/recording/output_synchronized'
    # output_path = '/media/statespace/0743-18EC/carla_ws/recording'
    # output_path = os.path.join(output_path,selected_town+"_"+selected_weather+"_"+str(selected_map_unloadlayers))
    output_path = os.path.join(output_path_base,selected_town+"_"+selected_weather)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def town_destinations(town):
    global town_destinations_dict

    town_destinations_dict = { 
    "Town01" : np.array([[9.8,2,0.2], 
                    [371,330,0.2], 
                    [9.8,2,0.2]]),
    "Town02" : np.array([[-7.386,129.2,0.2], 
                    [189.7,282.6,0.2], 
                    [-7.386,129.2,0.2], 
                    [189.7,282.6,0.2], 
                    [-7.386,129.2,0.2],
                    [189.7,282.6,0.2]]),
    "Town03" : np.array([[-5.28,-89.16,0.2], 
                    [-74.15,80.53,0.2], 
                    [154.5,-161,0.2],
                    [-74.15,80.53,0.2],
                    [46.55,-7.78,0.2],
                    [114.7,-76.38,0.2]]),
    "Town04" : np.array([[201.1,-278.5,0.2], 
                    [-507.6,201.4,0.2], 
                    [201.1,-278.5,0.2]]),
    "Town05" : np.array([[-9.24,-87.91,0.2], 
                    [-161.8,91.38,0.2], 
                    [-9.24,-87.91,0.2],
                    [-210.5,-88.21,0.2],
                    [-6.682,91.56,0.2],
                    [-9.24,-87.91,0.2]]),
    "Town06" : np.array([[-12.1,-65.54,0.2], 
                    [573.2,145.3,0.2],
                    [-359.3,169.3,0.2]]),
    "Town07" :np.array([[-4.323,-138.2,0.2], 
                    [-174.3,48.62,0.2], 
                    [-4.323,-138.2,0.2], 
                    [-198.3,-119.5,0.2], 
                    [-58.88,-2.768,0.2],
                    [-4.323,-138.2,0.2]]),
    "Town10HD" : np.array([[-45.23,-31.33,0.2], 
                    [106,49.27,0.2], 
                    [11.85,133.9,0.2], 
                    [19.81,-60.91,0.2], 
                    [-107.5,66.65,0.2]]),
    }

    return town_destinations_dict[town]

def setup_world(world):
    
    # set sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.03 ##0.03
    world.apply_settings(settings)

    return world

def set_map_layer(world):  
    # Buildings, Foliage, ParkedVehicles, Props, StreetLights, Walls
    # world = client.load_world("Town03_Opt", 
    #         carla.MapLayer.Buildings | carla.MapLayer.Foliage |
    #         carla.MapLayer.ParkedVehicles | carla.MapLayer.Props |
    #         carla.MapLayer.StreetLights | carla.MapLayer.Walls )

    map_layers_dict = {
        "Buildings":carla.MapLayer.Buildings,
        "Foliage":carla.MapLayer.Foliage,
        "Props":carla.MapLayer.Props,
        "StreetLights":carla.MapLayer.StreetLights,
        "Walls":carla.MapLayer.Walls,
        "ParkedVehicles":carla.MapLayer.ParkedVehicles,
    }

    map_layers_list = list(map_layers_dict.values())[0:selected_map_unloadlayers]
    for i in range(len(map_layers_list)):
        world.unload_map_layer(map_layers_list[i])
        print("unload " + str(map_layers_list[i]))
        time.sleep(1)

    return world

def set_weather(world):    

    global weather_dict
    
    weather_dict = { "ClearNoon" : carla.WeatherParameters.ClearNoon,
    "CloudyNoon" : carla.WeatherParameters.CloudyNoon,
    "WetNoon" : carla.WeatherParameters.WetNoon,
    "WetCloudyNoon" : carla.WeatherParameters.WetCloudyNoon,
    "MidRainyNoon" : carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon" : carla.WeatherParameters.HardRainNoon,
    "SoftRainNoon" : carla.WeatherParameters.SoftRainNoon,
    "ClearSunset" : carla.WeatherParameters.ClearSunset,
    "CloudySunset" : carla.WeatherParameters.CloudySunset,
    "WetSunset" : carla.WeatherParameters.WetSunset,
    "WetCloudySunset" : carla.WeatherParameters.WetCloudySunset,
    "MidRainSunset" : carla.WeatherParameters.MidRainSunset,
    "HardRainSunset" : carla.WeatherParameters.HardRainSunset,
    "SoftRainSunset" : carla.WeatherParameters.SoftRainSunset,
    "ClearNight":carla.WeatherParameters(cloudiness=0,
                                    precipitation=0,
                                    precipitation_deposits=0,
                                    fog_density=0,
                                    sun_altitude_angle=1,
                                    sun_azimuth_angle=0),
    "FogMorning":carla.WeatherParameters(cloudiness=0,
                                    precipitation=0,
                                    precipitation_deposits=0,
                                    fog_density=30,
                                    fog_distance=5,
                                    sun_altitude_angle=5,
                                    sun_azimuth_angle=180)
    }

    world.set_weather(weather_dict[selected_weather])

    return weather_dict

def set_destinations(world, agent, starting, destination):

    starting_location = carla.Location(starting[0],starting[1],starting[2])

    # set the destination spot
    destination_location = carla.Location(destination[0],destination[1],destination[2])

    # nearest waypoint on the center of a Driving or Sidewalk lane.
    destination_waypoint = world.get_map().get_waypoint(destination_location,project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
    destination_location = destination_waypoint.transform.location
    destination_location.z = 0.2

    # generate the route
    agent.set_destination(starting_location, destination_location)

def sensor_callback(sensor_data, sensor_name):
    if 'rbg_camera' in sensor_name:
        sensor_data.save_to_disk(os.path.join(output_path, 'rgb_%06d.jpg' % sensor_data.frame))
        agent_speed_list.append([agent.speed])
        transform_list.append([transform.location.x,transform.location.y,transform.location.z, \
            transform.rotation.roll,transform.rotation.pitch,transform.rotation.yaw])

    if 'depth_camera' in sensor_name:
        sensor_data.save_to_disk(os.path.join(output_path, 'depth_%06d.jpg' % sensor_data.frame), carla.ColorConverter.LogarithmicDepth)


def setup_vehicle(world):

    blueprint_library = world.get_blueprint_library()

    # set starting point    
    destinations = town_destinations(selected_town)
    start = destinations[0]
    start_location = carla.Location(x=start[0],y=start[1],z=start[2])
    start_rotation = carla.Rotation(roll=0,pitch=0,yaw=0)
    spawn_point = carla.Transform(start_location, start_rotation)

    # nearest waypoint on the center of a Driving or Sidewalk lane.
    spwan_waypoint = world.get_map().get_waypoint(spawn_point.location, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
    spawn_point = spwan_waypoint.transform
    spawn_point.location.z = 0.5

    # create the blueprint library
    ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    ego_vehicle_bp.set_attribute('color', '0, 0, 0')
    # spawn the vehicle
    vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)

    ##---add depth camera
    # camera_bp_depth = world.get_blueprint_library().find('sensor.camera.depth')
    # camera_bp_depth.set_attribute('image_size_x', str(1600))
    # camera_bp_depth.set_attribute('image_size_y', str(900))
    # camera_bp_depth.set_attribute('fov', str(105))
    # camera_bp_depth.set_attribute('sensor_tick', '0.0')
    # camera_transform = carla.Transform(carla.Location(x=0.5,y=-0.4, z=2.0))

    # global depth_camera
    # depth_camera = world.spawn_actor(camera_bp_depth, camera_transform, attach_to=vehicle)
    # # set the callback function
    # depth_camera.listen(lambda image: sensor_callback(image, "depth_camera"))
    # output_depth_path = os.path.join(output_path_base,selected_town+"_Depth")
    # if not os.path.exists(output_depth_path):
    #     os.makedirs(output_depth_path)
    # depth_camera.listen(lambda image: image.save_to_disk(os.path.join(output_depth_path, 'depth_%06d.jpg' % image.frame), carla.ColorConverter.LogarithmicDepth))

    ##---add driver view camera
    camera_bp_driverview = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp_driverview.set_attribute('image_size_x', str(1600))
    camera_bp_driverview.set_attribute('image_size_y', str(900))
    camera_bp_driverview.set_attribute('fov', str(105))
    camera_bp_driverview.set_attribute('sensor_tick', '0.0')
    camera_transform = carla.Transform(carla.Location(x=0.5,y=-0.4, z=2.0))

    global driverview_camera
    driverview_camera = world.spawn_actor(camera_bp_driverview, camera_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    # driverview_camera.listen(lambda image: image.save_to_disk(os.path.join(output_path, 'rgb_%06d.jpg' % image.frame)))
    # driverview_camera.listen(lambda image: sensor_callback(image, "rbg_camera"))

    ##---add semantic segmentation camera
    camera_bp_sem = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    camera_bp_sem.set_attribute('image_size_x', str(1600))
    camera_bp_sem.set_attribute('image_size_y', str(900))
    camera_bp_sem.set_attribute('fov', str(105))
    camera_bp_sem.set_attribute('sensor_tick', '0.0')
    camera_transform = carla.Transform(carla.Location(x=0.5,y=-0.4, z=2.0))

    global sem_camera
    sem_camera = world.spawn_actor(camera_bp_sem, camera_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    sem_camera.listen(lambda image: image.save_to_disk(os.path.join(output_path, 'sem_%06d.jpg' % image.frame), carla.ColorConverter.CityScapesPalette))

    # sem_camera.listen(lambda image: sensor_callback(image, "rbg_camera"))

    ##---add lidar
    # attach_lidar(world, vehicle)

    ##---add gnss
    attach_gnss(world, vehicle)

    # create the behavior agent
    agent = BehaviorAgent(vehicle, behavior='normal')
    
    return vehicle, agent

# Add GNSS sensor to ego vehicle. 
def gnss_callback(gnss):
    
    global gnss_longitude, gnss_latitude
    gnss_longitude = gnss.longitude
    gnss_latitude = gnss.latitude

    # transform_list.append([gnss.longitude,gnss.latitude])
    # print("GNSS measure:\n"+str(gnss)+'\n')

def attach_gnss(world, vehicle):
    gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
    gnss_location = carla.Location(0,0,0)
    gnss_rotation = carla.Rotation(0,0,0)
    gnss_transform = carla.Transform(gnss_location,gnss_rotation)
    # gnss_bp.set_attribute("sensor_tick",str(10.0))
    global ego_gnss
    ego_gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
    ego_gnss.listen(lambda gnss: gnss_callback(gnss))

def attach_lidar(world, vehicle):
    lidar_transform = carla.Transform(carla.Location(z=2.0))
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    # Set the time in seconds between sensor captures
    lidar_bp.set_attribute('channels', str(32))
    lidar_bp.set_attribute('points_per_second', str(90000))
    lidar_bp.set_attribute('rotation_frequency', str(40))
    lidar_bp.set_attribute('range', str(20))

    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_sensor.listen(lambda point_cloud: point_cloud.save_to_disk(os.path.join(output_path,'lidar_%06d.ply' % point_cloud.frame)))
    
    return lidar_sensor
    
def spawn_actor(world):

    # set map layter
    # world = set_map_layer(world)

    # set world
    world = setup_world(world)

    # set weather
    set_weather(world)

    # set vehicle
    vehicle, agent = setup_vehicle(world)
    
    return agent, vehicle

def load_cam_rotation_yaw():
    
    file_dir = "/media/statespace/Spatial/sptialworkspace/spatialfMRI/simulation/carla_ws/run_recording/"
    matdic = loadmat(file_dir+"cam_rotation_yaw_array.mat")

    cam_rotation_yaw_array = matdic["cam_rotation_yaw_array"]
    # print("cam_rotation_yaw_array shape:", cam_rotation_yaw_array.shape)
    
    return cam_rotation_yaw_array

def run_town_weather(selected_town_index, selected_weather_index):
    global actor_list, town_destinations_dict, weather_dict, transform_list, agent_speed_list
    global first_town
    first_town = True

    actor_list = []
    town_destinations_dict = {}
    weather_dict = {}
    transform_list = []
    agent_speed_list = []

    # select town weather
    select_town_weather(selected_town_index, selected_weather_index)

    # config
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world(selected_town)
    
    # spawn
    global agent, vehicle
    agent, vehicle = spawn_actor(world)

    # top view
    spectator = world.get_spectator()

    # destination
    total_desitnation_reached = 1
    destinations = town_destinations(selected_town)

    set_destinations(world, agent, destinations[total_desitnation_reached-1],destinations[total_desitnation_reached])

    # stop or change goals
    num_min_waypoints = 1

    # load_cam_rotation_yaw
    cam_rotation_yaw_index = 0
    cam_rotation_yaw_array = load_cam_rotation_yaw()

    try:
        while True:
            # update vehicle info
            agent.update_information(vehicle)
            world.tick()

            # set camera transform
            # cam_location = carla.Location(x=0.5,y=-0.4, z=2.0)
            # cam_rotation_yaw = int(np.random.uniform(-90, 90))
            # cam_rotation_yaw = 0
            # cam_rotation_yaw = cam_rotation_yaw_array[0, cam_rotation_yaw_index]
            # cam_rotation_yaw_index = cam_rotation_yaw_index + 1

            # cam_rotation = carla.Rotation(yaw=float(cam_rotation_yaw))
            # cam_transform = carla.Transform(cam_location,cam_rotation)
            # driverview_camera.set_transform(cam_transform)
            # depth_camera.set_transform(cam_transform)

            # spectator camera
            # spectator_camera = driverview_camera
            # spectator.set_transform(spectator_camera.get_transform())
            # top view
            global transform
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
                                                    carla.Rotation(pitch=-90)))

            ##set traffic light
            if vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            ##reach the goal
            current_location = transform.location
            next_waypoint = destinations[total_desitnation_reached]
            # print("current_location:",current_location.x,",",current_location.y)
            # print("next_waypoint:",next_waypoint[0],",",next_waypoint[1])

            waypoint_distance = np.sqrt(np.power(current_location.x-next_waypoint[0], 2.0) \
                                    + np.power(current_location.y-next_waypoint[1], 2.0))

            if waypoint_distance < 40.0:
                print("waypoint_distance:",waypoint_distance)
            # if agent.incoming_waypoint is None:
            # if len(agent.get_local_planner().waypoints_queue) <= num_min_waypoints:
                total_desitnation_reached = total_desitnation_reached + 1
                remaining_destination = destinations.shape[0] - total_desitnation_reached
                if remaining_destination == 0:
                    print('Reach all destinations, stop...')
                    control = carla.VehicleControl(brake = 2)
                    vehicle.apply_control(control)
                    break
                else:
                    set_destinations(world, agent, destinations[total_desitnation_reached-1],destinations[total_desitnation_reached])
                    # agent.update_information(vehicle)
                    # world.tick()
                    print('Go to next destination: ',destinations[total_desitnation_reached])
            
            # print('current speed:',agent.speed, ', speed_limit:',agent.speed_limit, 'min speed:', agent.min_speed)
            vehicle.apply_control(agent.run_step(debug=False))

    finally:
        print('saving agent speed...')
        agent_speed_dict = {"agent_speed":agent_speed_list}
        agent_speed_path = '/media/statespace/Spatial/sptialworkspace/spatialfMRI/simulation/carla_ws/run_recording/analysis/'
        save_name = agent_speed_path+selected_town+"_agent_speed.mat"
        savemat(save_name, agent_speed_dict)

        print('saving transform data...')
        transform_dict = {"transform":transform_list}
        transform_path = '/media/statespace/Spatial/sptialworkspace/spatialfMRI/simulation/carla_ws/run_recording/analysis/'
        save_name = transform_path+selected_town+"_transform.mat"
        savemat(save_name, transform_dict)

        # --------------
        # Destroy actors
        # --------------
        # if vehicle is not None:
        #     if depth_camera is not None:
        #         depth_camera.stop()
        #         depth_camera.destroy()
        #     if driverview_camera is not None:
        #         driverview_camera.stop()
        #         driverview_camera.destroy()
        #     if ego_gnss is not None:
        #         ego_gnss.stop()
        #         ego_gnss.destroy()
        #     vehicle.destroy()
            
        # print('Destroy done.')

def main():
    ###-------start main--------###
    # select town weather
    # selected_towns =[
    #     "Town01",
    #     "Town02",
    #     "Town03",
    #     "Town04",
    #     "Town05",
    #     "Town06",
    #     "Town07",    
    #     "Town10HD",    
    # ]
    # selected_weathers = [
        # "CloudyNoon",
        # "ClearNight",
        # "WetSunset",
        # "WetCloudyNoon",
        # "ClearNoon",
        # "ClearSunset",
        # "HardRainSunset",
        # "FogMorning",
    # ]

    selected_town_index = 0 # 0-8
    selected_weather_index = 0 # 0-5

    # for town_index in range(0, 8):
    #     for weather_index in range(0, 8):
    for town_index in range(0, 8):
        for weather_index in range(4, 5):
            selected_town_index = town_index
            selected_weather_index = weather_index
            run_town_weather(selected_town_index, selected_weather_index)
            print(selected_towns[selected_town_index]+', '+selected_weathers[selected_weather_index]+' is done')

    sys.exit()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')

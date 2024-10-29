from magpie.ur5 import UR5_Interface as robot
from magpie.ur5 import pose_vector_to_homog_coord, homog_coord_to_pose_vector
from magpie.realsense_wrapper import RealSense
import time
import cv2
import numpy as np
import random
import os
import pyrealsense2 as rs
import math

topview_vec = [-0.16188220333609551, -0.6234229524443915, 0.5474838984217083, 1e-5, -math.pi, -1e-5]
#topview_vec[1] = topview_vec[1] + 0.15
#topview_vec[0] = topview_vec[0] - 0.01
sideview_vec =[0.0360674358115564, -0.20624107287146376, 0.2646274319314355, 1.8434675848139614, 1.4569842711938066, -1.2315497051361715]#[-0.1422445979238582, -0.286205181894235, 0.21758935908098342, -0.013462424428371759, -2.461550255061062, 1.8468177270833177]
def get_frames(rsWrapper):
    pipe, config = rsWrapper.pipe, rsWrapper.config
    frames = pipe.wait_for_frames()
    #alignOperator = rs.align(rs.stream.color)
    #alignOperator.process(frames)
    depthFrame = frames.get_depth_frame()  # pyrealsense2.depth_frame
    colorFrame = frames.get_color_frame()
    return colorFrame, depthFrame
def get_pictures(rsWrapper):
    colorFrame, depthFrame = get_frames(rsWrapper)
    #print(f"{type(starting_img)=}")
    #print(f"{dir(starting_img)=}")
    color_image = np.asarray(colorFrame.get_data())
    #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    depth_image = np.asarray(depthFrame.get_data())
    return color_image, depth_image
def goto_topview(UR_interface, asynch):
    #print(f"{sideview_vec=}")
    topview_matrix = UR_interface.poseVectorToMatrix(topview_vec)
    #print(f"{sideview_matrix=}")
    UR_interface.moveL(topview_matrix, linSpeed=0.1, asynch=asynch)
    #UR_interface.align_tcp(lock_roll = False, lock_pitch = False, lock_yaw = False)
def goto_sideview(UR_interface, asynch):
    print(f"{sideview_vec=}")
    sideview_matrix = UR_interface.poseVectorToMatrix(sideview_vec)
    print(f"{sideview_matrix=}")
    UR_interface.moveL(sideview_matrix, linSpeed=0.1, asynch=asynch)
    ##-- this bellow causes problems with colliding with itself --##
    #UR_interface.align_tcp(lock_roll = False, lock_pitch = False, lock_yaw = False)
def take_sideview_img(UR_interface, rsWrapper, save_path = None):
    UR_interface.open_gripper()
    goto_sideview(UR_interface, asynch=False)
    rgb_img, depth_img = get_pictures(rsWrapper)
    if save_path is not None:
        cv2.imwrite(save_path, rgb_img)
    return rgb_img, depth_img
def take_topview_img(UR_interface, rsWrapper, save_path = None):
    UR_interface.open_gripper()
    goto_topview(UR_interface, asynch=False)
    rgb_img, depth_img = get_pictures(rsWrapper)
    if save_path is not None:
        cv2.imwrite(save_path, rgb_img)
    return rgb_img, depth_img

if __name__ == "__main__":
    myrs = RealSense()
    myrs.initConnection()
    myrobot = robot()
    myrobot.align_tcp(lock_roll = False, lock_pitch = False, lock_yaw = False)

    myrobot.start()

    print("hello world")
    

    #for i in range(10):
        #rgb_img, depth_img = take_sideview_img(myrobot, myrs, save_path=f"{dir_path}side.png")
    #rgb_img, depth_img = take_topview_img(myrobot, myrs)#, save_path=f"{dir_path}top.png")
    goto_topview(myrobot, asynch = False)
    #topview_vec[0] = topview_vec[0] + 0.1
    #goto_topview(myrobot, asynch = False)
    #topview_vec[1] = topview_vec[1] + 0.1
    #goto_topview(myrobot, asynch=False)
    c, d = get_pictures(myrs)
    print(f"{c.shape}")
    print(f"{d.shape}")


    myrobot.stop()
    myrs.disconnect()
    
        
        
        


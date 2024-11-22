import numpy as np
from config import arm_speed
import warnings

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
def pose_vector_distance(goal_vec, actual_pose):
    """
    Check if two 6-degree pose vectors are equivalent within a specified tolerance.

    Parameters:
    - goal_vec: List of target pose [x, y, z, rx, ry, rz] in meters.
    - actual_pose: List of actual pose [x, y, z, rx, ry, rz] in meters.
    - tolerance_cm: Tolerance in centimeters (default is 0.01 meters).

    Returns:
    - True if the poses are equivalent within the tolerance, False otherwise.
    """


    # Calculate linear distance
    linear_distance = np.linalg.norm(np.array(goal_vec[:3]) - np.array(actual_pose[:3]))

    # Calculate angular differences (wrap around at 2*pi)
    angular_diffs = [
        np.arctan2(np.sin(goal_vec[i] - actual_pose[i]), np.cos(goal_vec[i] - actual_pose[i]))
        for i in range(3, 6)
    ]
    angular_distance = np.linalg.norm(angular_diffs)

    # Total distance check (considering both linear and angular)
    return linear_distance, angular_distance

def goto_vec(UR_interface, goal_vec, warning_tolorance=0.01, failure_tolerance=0.1):
    #print(f"{goal_vec=}")
    goal_matrix = UR_interface.poseVectorToMatrix(goal_vec).A
    #print(f"{goal_matrix=}")
    #print(type(goal_matrix))
    #print(f"{goal_matrix[:3, 3]=}")
    UR_interface.moveL(goal_matrix, linSpeed=arm_speed, asynch=False)
    #UR_interface.move_tcp_cartesian(goal_matrix, z_offset=0)

    actual_pose = UR_interface.recv.getActualTCPPose()
    #print(f"{actual_pose=}")
    linear_error, angular_error = pose_vector_distance(goal_vec, actual_pose)
    
    success = True
    if linear_error >= warning_tolorance:
        assert linear_error < failure_tolerance, f"Linear Error greater than failure tolerance {linear_error=} {goal_vec=} {actual_pose=}"
        warnings.warn(f"Linear Error greater than warning tolerance {linear_error=} {goal_vec=} {actual_pose=}")
        success = False
    if angular_error >= warning_tolorance:
        #assert angular_error < failure_tolerance, f"Angular Error greater than failure tolerance {linear_error=} {goal_vec=} {actual_pose=}"
        warnings.warn(f"Angular Error greater than warning tolerance {angular_error=} {goal_vec=} {actual_pose=}")
        success = False
    return success

def get_depth_frame_intrinsics(rs_wrapper):
    _, depth_frame = get_frames(rs_wrapper)
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    #print(f"{dir(depth_frame.profile.as_video_stream_profile())=}")
    #print(f"{dir(depth_frame.profile)=}")

    depth_scale = 1/rs_wrapper.depthScale
    #print(f"{depth_scale=}")
    return depth_scale, intrinsics

if __name__ == "__main__":
    from magpie_control import realsense_wrapper as real
    from magpie_control.ur5 import UR5_Interface as robot
    from config import sideview_vec, topview_vec
    import matplotlib.pyplot as plt


    myrs = real.RealSense()
    myrs.initConnection()
    myrobot = robot()
    print(f"starting robot from control_scripts")
    myrobot.start()

    for i in range(10):
        goto_vec(myrobot, sideview_vec)
        goto_vec(myrobot, topview_vec)
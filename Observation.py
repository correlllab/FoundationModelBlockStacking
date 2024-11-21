import torch
import numpy as np
from control_scripts import get_pictures, get_frames
from config import n_depth_samples, PC_X_offset, PC_Y_offset, PC_Z_offset, realSenseFPS
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
import cv2
import warnings
from sklearn.decomposition import PCA
from magpie_control.ur5 import pose_vector_to_homog_coord, homog_coord_to_pose_vector

def align_axis_to_vector(vector, axis):
    """
    Computes a rotation matrix to align the given axis with a target vector.
    
    Parameters:
        vector (np.ndarray): Target vector to align the axis with (3D).
        axis (np.ndarray): Axis to align with the vector (default is z-axis [0, 0, 1]).
    
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    # Normalize the input vector and axis
    vector = vector / np.linalg.norm(vector)
    axis = axis / np.linalg.norm(axis)
    
    # Compute the cross product and the sine of the angle
    cross = np.cross(axis, vector)
    sin_theta = np.linalg.norm(cross)
    
    # Compute the dot product and the cosine of the angle
    cos_theta = np.dot(axis, vector)
    
    # Create the skew-symmetric cross-product matrix
    cross_matrix = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    
    # Compute the rotation matrix using Rodrigues' rotation formula
    I = np.eye(3)
    R = I + cross_matrix + cross_matrix @ cross_matrix * ((1 - cos_theta) / (sin_theta**2))
    
    return R
def rotation_matrix_to_rpy(R):
    roll = np.arctan2(R[2, 1], R[2, 2])  # atan2(R32, R33)
    pitch = -np.arcsin(R[2, 0])          # -asin(R31)
    yaw = np.arctan2(R[1, 0], R[0, 0])   # atan2(R21, R11)
    return roll, pitch, yaw
def rpy_to_rotation_matrix(roll, pitch, yaw):
    # Compute individual rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x
    return R
def display_observations(observations, observer_pose = None):
    print(f"{len(observations)=}") 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    if observer_pose is not None:
        observer_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        observer_sphere.translate(observer_pose[:3])
        observer_sphere.paint_uniform_color([1, 0, 0])
        vis.add_geometry(observer_sphere)

    for observation in observations:
        vis.add_geometry(observation.pcd)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(observation.centroid)
        sphere.paint_uniform_color([0, 0, 0])
        vis.add_geometry(sphere)

        pickPose_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=0.1)
        rot_mat = rpy_to_rotation_matrix(observation.pickPose[3], observation.pickPose[4], observation.pickPose[5])
        pickPose_cylinder.rotate(rot_mat, center=(0, 0, 0))
        pickPose_cylinder.paint_uniform_color([0.2, 0.2, 0.2])
        pickPose_cylinder.translate(observation.pickPose[:3])


        vis.add_geometry(pickPose_cylinder)        
    vis.add_geometry(axis)
    opt = vis.get_render_option()
    opt.point_size = 1.0  # Set to a smaller size (default is 5.0)
    # Run the visualizer
    vis.run()
def get_observation_patch(obs, edge_color = "r"):
    rect = patches.Rectangle(
                            (obs.xmin, obs.ymin),
                                obs.xmax - obs.xmin,
                                obs.ymax - obs.ymin,
                                linewidth=2, edgecolor=edge_color, facecolor='none'
                            )
    return rect

def get_depth_frame_intrinsics(rs_wrapper):
    _, depth_frame = get_frames(rs_wrapper)
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    #print(f"{dir(depth_frame.profile.as_video_stream_profile())=}")
    #print(f"{dir(depth_frame.profile)=}")

    depth_scale = 1/rs_wrapper.depthScale
    #print(f"{depth_scale=}")
    return depth_scale, intrinsics
def get_refined_depth(rs_wrapper):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    depth_images = []
    rgb_img = None
    for i in range(n_depth_samples):
        rgb_img, depth_image = get_pictures(rs_wrapper)
        depth_images.append(depth_image)

    depth_stack = np.stack(depth_images, axis=0)
    #print(f"{depth_stack.shape=}")

    #Compute mean ignoring 0 values
    sum_depth_stack = np.sum(depth_stack, axis=0)
    non_zero_counts = np.count_nonzero(depth_stack, axis=0)
    #print(f"{sum_depth_stack.shape=}")
    #print(f"{non_zero_counts.shape=}")
    mean_depth_image = sum_depth_stack/non_zero_counts#np.divide(sum_depth_stack, non_zero_counts, where=non_zero_counts != 0)
    mean_depth_image = np.nan_to_num(mean_depth_image, nan=0)
    #print(mean_depth_image.shape)

    #compute std deviation ignoring 0 values
    squared_diff_stack = (depth_stack - mean_depth_image[None, :, :]) ** 2
    squared_diff_stack[depth_stack == 0] = 0  # Ignore zero values
    sum_squared_diff = np.sum(squared_diff_stack, axis=0)
    std_dev_image = np.sqrt(sum_squared_diff / non_zero_counts)
    std_dev_image = np.nan_to_num(std_dev_image, nan=0)
    #print(f"{std_dev_image.shape=}")

    #get mask of points within 1 standard deviation
    lower_bounds = mean_depth_image - std_dev_image
    upper_bounds = mean_depth_image + std_dev_image
    mask = (depth_stack >= lower_bounds[None, :, :]) & (depth_stack <= upper_bounds[None, :, :])
    #set points not within one standard deviation to 0
    filtered_depth_stack = np.where(mask, depth_stack, 0)

    #Compute mean ignoring 0 values and values not within 1 standard deviation
    sum_depth_stack = np.sum(filtered_depth_stack, axis=0)
    non_zero_counts = np.count_nonzero(filtered_depth_stack, axis=0)
    filtered_depth_image = sum_depth_stack/non_zero_counts
    filtered_depth_image = filtered_depth_image.astype(np.float32)

    warnings.simplefilter("default", category=RuntimeWarning)
    return filtered_depth_image

class observation:
    def __init__(self, str_label):
        self.str_label = str_label
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        
        self.mask = None
        
        self.rgb_segment = None
        self.depth_segment = None
        self.pcd = None

        self.pickPose = None
        self.placePose = None

    def calc_bbox(self, label_vit, rgb_img):
        bbox = None
        with torch.no_grad():
            bbox = label_vit.label(rgb_img, self.str_label, self.str_label, plot=False, topk=True)
            bbox = bbox[1][0].tolist()
        self.xmin = int(bbox[0])
        self.ymin = int(bbox[1])
        self.xmax = int(bbox[2])
        self.ymax = int(bbox[3])

        #self.xmin = np.clip(self.xmin, 0+invalid_border_px_x, rgb_img.shape[1]-invalid_border_px_x)
        #self.xmax = np.clip(self.xmax, 0+invalid_border_px_x, rgb_img.shape[1]-invalid_border_px_x)
        #self.ymin = np.clip(self.ymin, 0+invalid_border_px_y, rgb_img.shape[0]-invalid_border_px_y)
        #self.ymax = np.clip(self.ymax, 0+invalid_border_px_y, rgb_img.shape[0]-invalid_border_px_y)


    def calc_pc(self, sam_predictor, rgb_img, depth_img, K, depth_scale, pose2world_transform, display = False):
        sam_predictor.set_image(rgb_img)
        sam_box = np.array([self.xmin,  self.ymin,  self.xmax,  self.ymax])
        sam_mask, sam_scores, sam_logits = sam_predictor.predict(box=sam_box)
        sam_mask = np.all(sam_mask, axis=0)
        expanded_sam_mask = np.expand_dims(sam_mask, axis=-1)
        

        self.mask = sam_mask
        self.rgb_segment = rgb_img
        self.rgb_segment[~sam_mask] = 0
        self.depth_segment = depth_img
        self.depth_segment[~sam_mask] = 0

        temp_rgb_img = o3d.geometry.Image(self.rgb_segment)
        temp_depth_img = o3d.geometry.Image(self.depth_segment)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(temp_rgb_img, temp_depth_img, depth_scale=depth_scale, depth_trunc=10.0, convert_rgb_to_intensity=False)
        #print(f"{dir(K)=}")
        intrinsic = o3d.camera.PinholeCameraIntrinsic(K.width, K.height, K.fx, K.fy, K.ppx, K.ppy)
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        offset_matrix = np.array([
            [1, 0, 0, PC_X_offset],
            [0, 1, 0, PC_Y_offset],
            [0, 0, 1, PC_Z_offset],
            [0, 0, 0, 1]
        ])
        transform_matrix = pose2world_transform @ offset_matrix
        self.pcd.transform(transform_matrix)
        
        self.centroid = np.asarray(self.pcd.points).mean(axis=0)
        #print(f"self.centroid={self.centroid}")
        if display:
            display_observations([self])

    def calc_pick_pose(self):
        self.pickPose = [0,0,0,0,0,0]
        self.pickPose[0] = self.centroid[0]
        self.pickPose[1] = self.centroid[1]
        self.pickPose[2] = self.centroid[2]

        pca = PCA(n_components=3)
        points = np.asarray(self.pcd.points)
        #print(f"{points.shape=}")
        pca.fit(points)
        principal_component = pca.components_[2]
        roll, pitch, yaw = rotation_matrix_to_rpy(align_axis_to_vector(principal_component, axis=np.array([0, 1, 0])))
        pitch *= -1
        self.pickPose[3] = roll
        self.pickPose[4] = pitch
        self.pickPose[5] = yaw

    def calc_place_pose(self):
        self.placePose = self.pickPose.copy()
        self.placePose[2] = self.placePose[2] + 0.05

    def update_observation(self, rs_wrapper, label_vit, sam_predictor, pose2world_transform, display = False):
        rgb_img, depth_img = get_pictures(rs_wrapper)
        depth_img = get_refined_depth(rs_wrapper)
        rgb_img = cv2.rotate(rgb_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        depth_scale, K = get_depth_frame_intrinsics(rs_wrapper)

        self.calc_bbox(label_vit, rgb_img)

        self.calc_pc(sam_predictor, rgb_img, depth_img, K, depth_scale, pose2world_transform)

        self.calc_pick_pose()
        self.calc_place_pose()

        if display:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            axes[0, 0].imshow(rgb_img)
            axes[0, 0].add_patch(get_observation_patch(self, "r"))
            axes[0, 0].text(self.xmin, self.ymin - 10, f"{self.str_label}", color='r', fontsize=12, ha='left', va='bottom')
            axes[0, 0].set_title("RGB Image")

            axes[1, 0].imshow(self.mask)
            axes[1, 0].set_title("Mask")
            
            axes[0, 1].imshow(self.rgb_segment)
            axes[0, 1].set_title("RGB segment")

            axes[1, 1].imshow(self.depth_segment)
            axes[1, 1].set_title("Depth segment")


            plt.tight_layout()
            #print(f"Showing observation for {self.str_label}")
            plt.show(block = False)
            plt.pause(1)  # Keeps the figure open for 3 seconds


class observation_manager:
    def __init__(self, things_to_observe, rs_wrapper, label_vit, sam_predictor, UR_interface):
        self.rs_wrapper = rs_wrapper
        self.label_vit = label_vit
        self.sam_predictor = sam_predictor
        self.UR_interface = UR_interface
        self.observation_pose = None

        self.observations = {}
        for thing in things_to_observe:
            self.observations[thing] = observation(thing)

    def update_observations(self, display = False):
        transform = self.UR_interface.get_tcp_pose()
        self.observation_pose = homog_coord_to_pose_vector(transform)
        #print(f"{self.observation_pose=}")
        observation_list = list(self.observations.values())
        for obs in observation_list:
            #print(f"Updating observation for {obs.str_label}")
            obs.update_observation(self.rs_wrapper, self.label_vit, self.sam_predictor, transform)
        if display:
            display_observations(observation_list, observer_pose = self.observation_pose)

if __name__ == "__main__":
    from magpie_control import realsense_wrapper as real
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from magpie_perception.label_owlv2 import LabelOWLv2
    from magpie_control.ur5 import UR5_Interface as robot
    from control_scripts import goto_vec
    from config import topview_vec

    myrobot = robot()
    print(f"starting robot from observation")
    myrobot.start()


    myrs = real.RealSense(fps=realSenseFPS)
    myrs.initConnection()

    label_vit = LabelOWLv2(topk=1, score_threshold=0.01, cpu_override=False)
    label_vit.model.eval()
    print(f"{label_vit.model.device=}")

    sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    print(f"{sam_predictor.model.device=}")

    goto_vec(myrobot, topview_vec)
    print(f"{topview_vec=}")

    cam2base_transform = myrobot.get_tcp_pose()
    observation_list = ["red block", "blue block", "green block", "yellow block", "white paper"]
    om = observation_manager(observation_list, myrs, label_vit, sam_predictor, myrobot)
    om.update_observations(display=False)
    for target in observation_list:
        target_pose = om.observations[target].pickPose
        target_pose[3] = topview_vec[3]
        target_pose[4] = topview_vec[4]
        target_pose[5] = topview_vec[5]
        print(f"{target_pose=}")

        goto_vec(myrobot, target_pose)
        input()
        goto_vec(myrobot, topview_vec)
    myrobot.stop()
    myrs.disconnect()
    
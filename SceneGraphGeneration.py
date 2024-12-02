import torch
import numpy as np
from control_scripts import get_pictures, get_frames, get_depth_frame_intrinsics
from config import n_depth_samples, realSenseFPS, tcp_Z_offset, topview_vec
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
import cv2
import warnings
from sklearn.decomposition import PCA
from magpie_control.ur5 import pose_vector_to_homog_coord, homog_coord_to_pose_vector
#import magpie_control.ur5
#print(f"{magpie_control.ur5.__file__=}")
from magpie_perception.pcd import get_pca_frame
from gpt_planning import get_state
import time


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
def get_observation_patch(obs, edge_color = "r"):
    rect = patches.Rectangle(
                            (obs.pix_xmin, obs.pix_ymin),
                                obs.pix_xmax - obs.pix_xmin,
                                obs.pix_ymax - obs.pix_ymin,
                                linewidth=2, edgecolor=edge_color, facecolor='none'
                            )
    return rect
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
def display_graph(graph):
    print(dir(graph))
    print(f"{graph.nodes=}")
    print(f"{graph.edges=}")  
    print(f"{graph.observation_pose=}")  
    print(f"{graph.timestamp=}") 
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    T = pose_vector_to_homog_coord(graph.observation_pose)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    axis.transform(T)
    vis.add_geometry(axis)

    for node in graph.nodes.values():
        vis.add_geometry(node.pcd)
        vis.add_geometry(node.pcd_bbox)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])    
    vis.add_geometry(axis)
    opt = vis.get_render_option()
    opt.point_size = 1.0  # Set to a smaller size (default is 5.0)
    # Run the visualizer
    vis.run()

class Node:
    def __init__(self, str_label, label_vit, sam_predictor):
        self.str_label = str_label
        self.label_vit = label_vit
        self.sam_predictor = sam_predictor

        self.rgb_img = None
        self.depth_img = None

        self.pix_xmin = None
        self.pix_xmax = None
        self.pix_ymin = None
        self.pix_ymax = None
        
        self.mask = None
        
        self.rgb_segment = None
        self.depth_segment = None
        self.pcd = None
        self.pcd_bbox = None
        
    def calc_bbox(self):
        bbox = None
        with torch.no_grad():
            bbox = self.label_vit.label(self.rgb_img, self.str_label, self.str_label, plot=False, topk=True)
            bbox = bbox[1][0].tolist()
        self.pix_xmin = int(bbox[0])
        self.pix_ymin = int(bbox[1])
        self.pix_xmax = int(bbox[2])
        self.pix_ymax = int(bbox[3])

    def calc_pcd(self, K, depth_scale, observation_pose):
        self.sam_predictor.set_image(self.rgb_img)
        sam_box = np.array([self.pix_xmin,  self.pix_ymin,  self.pix_xmax,  self.pix_ymax])
        sam_mask, sam_scores, sam_logits = self.sam_predictor.predict(box=sam_box)
        sam_mask = np.all(sam_mask, axis=0)
        #expanded_sam_mask = np.expand_dims(sam_mask, axis=-1)
        
        self.mask = sam_mask
        self.rgb_segment = self.rgb_img.copy()
        self.rgb_segment[~sam_mask] = 0
        self.depth_segment = self.depth_img.copy()
        self.depth_segment[~sam_mask] = 0

        temp_rgb_img = o3d.geometry.Image(self.rgb_segment)
        temp_depth_img = o3d.geometry.Image(self.depth_segment)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(temp_rgb_img, temp_depth_img, depth_scale=depth_scale, depth_trunc=10.0, convert_rgb_to_intensity=False)
        #print(f"{dir(K)=}")
        intrinsic = o3d.camera.PinholeCameraIntrinsic(K.width, K.height, K.fx, K.fy, K.ppx, K.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        #pcd = pcd.uniform_down_sample(every_k_points=5)
        #pcd = pcd.voxel_down_sample(voxel_size=0.001)  # Down-sample with finer detail
        transform_matrix = pose_vector_to_homog_coord(observation_pose)
        pcd.transform(transform_matrix)
        
        
        #if self.pcd is None:
        self.pcd = pcd
        #else:
        #    self.pcd = self.pcd + pcd
        self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=1.0)
        #self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1)
        self.pcd_bbox = self.pcd.get_axis_aligned_bounding_box()
        #self.pcd_bbox = pcd.get_minimal_oriented_bounding_box()
        self.pcd_bbox.color = (1,0,0)
        
    def update_observation(self, rgb_img, depth_img, K, depth_scale, observation_pose, display = False):
        self.rgb_img = rgb_img
        self.depth_img = depth_img

        self.calc_bbox()

        self.calc_pcd(K, depth_scale, observation_pose)

        if display:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            axes[0, 0].imshow(self.rgb_img)
            axes[0, 0].add_patch(get_observation_patch(self, "r"))
            axes[0, 0].text(self.pix_xmin, self.pix_ymin - 10, f"{self.str_label}", color='r', fontsize=12, ha='left', va='bottom')
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


class Graph:
    def __init__(self):
        self.Nodes = {}
        self.edges = []
        self.timestamp = None
        self.observation_pose = None

class Graph_Manager:
    def __init__(self, UR_interface, rs_wrapper, label_vit, sam_predictor, OAI_Client):
        self.rs_wrapper = rs_wrapper
        self.UR_interface = UR_interface

        self.label_vit = label_vit
        self.sam_predictor = sam_predictor
        self.OAI_Client = OAI_Client

        self.graph_history = []

    def get_next_graph(self, display = False):
        graph = Graph()

        rgb_img, depth_img = get_pictures(self.rs_wrapper)
        graph.timestamp = time.time()

        #depth_img = get_refined_depth(self.rs_wrapper)
        depth_scale, K = get_depth_frame_intrinsics(self.rs_wrapper)
        
        observation_pose = homog_coord_to_pose_vector(self.UR_interface.get_cam_pose())
        graph.observation_pose = observation_pose
        

        _, state_json, _, _ = get_state(self.OAI_Client, rgb_img)
        graph.edges = state_json["object_relationships"]

        nodes = {}
        for object in state_json["objects"]:
            obj_node = Node(object, self.label_vit, self.sam_predictor)
            obj_node.update_observation(rgb_img, depth_img, K, depth_scale, observation_pose, display = False)
            nodes[object] = obj_node
            
        graph.nodes = nodes
        self.graph_history.append(graph)
        if display:
            display_graph(self.graph_history[-1])

    

if __name__ == "__main__":
    from magpie_control import realsense_wrapper as real
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from magpie_perception.label_owlv2 import LabelOWLv2
    from magpie_control.ur5 import UR5_Interface as robot
    from control_scripts import goto_vec
    from config import frontview_vec, leftview_vec, rightview_vec, behindview_vec
    from APIKeys import API_KEY
    from openai import OpenAI
    

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

    client = OpenAI(
        api_key= API_KEY,
    )
    """
    obs = Node("dark blue block", label_vit, sam_predictor)
    rgb_img, depth_img = get_pictures(myrs)
    depth_scale, K = get_depth_frame_intrinsics(myrs)
    observation_pose = homog_coord_to_pose_vector(myrobot.get_cam_pose())
    obs.update_observation(rgb_img, depth_img, K, depth_scale, observation_pose, display = True)
    """

    GM = Graph_Manager(myrobot, myrs, label_vit, sam_predictor, client)
    goto_vec(myrobot, topview_vec)

    graph = GM.get_next_graph(display=True)
    print(type(graph))


    myrobot.stop()
    myrs.disconnect()
    
import torch
import numpy as np
from control_scripts import get_pictures, get_frames, get_depth_frame_intrinsics
from config import n_depth_samples, realSenseFPS, topview_vec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
import warnings
from magpie_control.ur5 import pose_vector_to_homog_coord, homog_coord_to_pose_vector
#import magpie_control.ur5
#print(f"{magpie_control.ur5.__file__=}")
import networkx as nx
from gpt_planning import get_state
import time

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

class Node:
    def __init__(self, str_label, rgb_img, depth_img, label_vit, sam_predictor, K, depth_scale, observation_pose):
        self.str_label = str_label

        self.pix_xmin = None
        self.pix_xmax = None
        self.pix_ymin = None
        self.pix_ymax = None
        
        self.mask = None
        
        self.rgb_segment = None
        self.depth_segment = None

        self.points = None
        self.colors = None

        self.calc_bbox(rgb_img, label_vit)
        self.calc_pcd(rgb_img, depth_img, K, depth_scale, observation_pose, sam_predictor)
    def calc_bbox(self, rgb_img, label_vit):
        bbox = None
        with torch.no_grad():
            bbox = label_vit.label(rgb_img, self.str_label, self.str_label, plot=False, topk=True)
            bbox = bbox[1][0].tolist()
        self.pix_xmin = int(bbox[0])
        self.pix_ymin = int(bbox[1])
        self.pix_xmax = int(bbox[2])
        self.pix_ymax = int(bbox[3])

    def calc_pcd(self, rgb_img, depth_img, K, depth_scale, observation_pose, sam_predictor):
        sam_predictor.set_image(rgb_img)
        sam_box = np.array([self.pix_xmin,  self.pix_ymin,  self.pix_xmax,  self.pix_ymax])
        sam_mask, sam_scores, sam_logits = sam_predictor.predict(box=sam_box)
        sam_mask = np.all(sam_mask, axis=0)
        #expanded_sam_mask = np.expand_dims(sam_mask, axis=-1)
        
        self.mask = sam_mask
        self.rgb_segment = rgb_img.copy()
        self.rgb_segment[~sam_mask] = 0
        self.depth_segment = depth_img.copy()
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
        
     
        #self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=1.0)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
        voxel_size = 0.005  # adjust based on your data
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        #self.pcd_bbox = self.pcd.get_axis_aligned_bounding_box()
        #self.pcd_bbox = pcd.get_minimal_oriented_bounding_box()
        #self.pcd_bbox.color = (1,0,0)
        self.points = np.asarray(pcd.points)
        self.colors = np.asarray(pcd.colors)
        
    def display(self):
        fig, axes = plt.subplots(ncols=2, nrows=2)
        axes[0, 0].imshow(self.rgb_segment)
        axes[0, 0].add_patch(get_observation_patch(self, "r"))
        axes[0, 0].text(self.pix_xmin, self.pix_ymin - 10, f"{self.str_label}", color='r', fontsize=12, ha='left', va='bottom')
        axes[0, 0].set_title("RGB segment")

        axes[0, 1].imshow(self.depth_segment)
        axes[0, 1].add_patch(get_observation_patch(self, "r"))
        axes[0, 1].text(self.pix_xmin, self.pix_ymin - 10, f"{self.str_label}", color='r', fontsize=12, ha='left', va='bottom')
        axes[0, 1].set_title("Depth segment")
    
        plt.tight_layout()
        #print(f"Showing observation for {self.str_label}")
        plt.show(block = False)
        plt.pause(1)  # Keeps the figure open for 3 seconds
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], c=self.colors, s=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(f"{self.str_label} Point Cloud")
        plt.show()
 
            

def get_graph(OAI_Client, label_vit, sam_predictor, rs_wrapper, UR_interface):
    G = nx.DiGraph()

    rgb_img, depth_img = get_pictures(rs_wrapper)
    G.graph["timestamp"] = time.time()

    #depth_img = get_refined_depth(self.rs_wrapper)
    depth_scale, K = get_depth_frame_intrinsics(rs_wrapper)
    
    observation_pose = homog_coord_to_pose_vector(UR_interface.get_cam_pose())
    G.graph["observation_pose"] = observation_pose
    

    _, state_json, _, _ = get_state(OAI_Client, rgb_img)

    for object in state_json["objects"]:
        obj_node = Node(object, rgb_img, depth_img, label_vit, sam_predictor, K, depth_scale, observation_pose)
        G.add_node(object, data=obj_node)

    for edge in state_json["object_relationships"]:
        G.add_edge(edge[0], edge[2], connection=edge[1])
        
    return G

def get_points(G):
    points = []
    colors = []

    for obj, node in G.nodes(data=True):
        #C = node["data"].pcd.get_center()
        #sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        #sphere.paint_uniform_color([0, 0, 0])  # Red color for sphere A
        #sphere.translate(C)
        #vis.add_geometry(sphere)
        points.append(node["data"].points)
        colors.append(node["data"].colors)
    
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)

    return points, colors

def display_graph(G, blocking = False):
    
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 12))
    points, colors = get_points(G)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for edge in G.edges(data=True):
        relation_str = f"{edge[0]} {edge[2]['connection']} {edge[1]}"

        source = edge[0]
        source_node = G.nodes[source]
        source_centroid = np.mean(source_node["data"].points, axis=0)

        dest = edge[1]
        dest_node = G.nodes[dest]
        dest_centroid = np.mean(dest_node["data"].points, axis=0)

        direction = dest_centroid - source_centroid
        length = np.linalg.norm(direction)

        ax.quiver(source_centroid[0], source_centroid[1], source_centroid[2],
                    direction[0], direction[1], direction[2],
                    length=length, normalize=False, color='r')
    
    




    pos = nx.spring_layout(G)
    # Draw the graph on the specified axes
    nx.draw(G, pos=pos, ax=axes[0], with_labels=True, node_color='lightblue', node_size=1500, font_size=15)
    # Draw edge labels on the same axes
    edge_labels = nx.get_edge_attributes(G, 'connection')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=12, ax=axes[0])
    # Show without blocking, then pause
    plt.tight_layout()
    plt.show(block=blocking)
    if not blocking:
        plt.pause(1)

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

    goto_vec(myrobot, topview_vec)

    rgb_img, depth_img = get_pictures(myrs)
    depth_scale, K = get_depth_frame_intrinsics(myrs)
    pose = homog_coord_to_pose_vector(myrobot.get_cam_pose())
    #__init__(self, str_label, rgb_img, depth_img, label_vit, sam_predictor, K, depth_scale, observation_pose)

    #obs = Node("dark blue block", rgb_img, depth_img, label_vit, sam_predictor, K, depth_scale, pose)
    #obs.display()

    graph = get_graph(client, label_vit, sam_predictor, myrs, myrobot)
    display_graph(graph, blocking = True)


    myrobot.stop()
    myrs.disconnect()
    
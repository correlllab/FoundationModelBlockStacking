import torch
import numpy as np
from control_scripts import get_pictures, get_frames
from config import n_depth_samples
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_observation_patch(obs, edge_color = "r"):
    rect = patches.Rectangle(
                            (obs.xmin, obs.ymin),
                                obs.xmax - obs.xmin,
                                obs.ymax - obs.ymin,
                                linewidth=2, edgecolor=edge_color, facecolor='none'
                            )
    return rect

def get_depth_frame_intrinsics(rs_wrapper):
    rgb_frame, depth_frame = get_frames(rs_wrapper)
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    return depth_frame, intrinsics

def get_depths(point_list, rs_wrapper):
    depth_measurements = [[] for p in point_list]
    intrinsics = None
    for i in range(n_depth_samples):
        depth_frame, intrinsics = get_depth_frame_intrinsics(rs_wrapper)
        for i, (x, y) in enumerate(point_list):
            depth_val = depth_frame.get_distance(x, y)  # in meters
            if depth_val > 0:
                depth_measurements[i].append(depth_val)
    
    depth_measurements = [np.array(point_measurements) for point_measurements in depth_measurements]
    final_depth_measurements = [0 for point in point_list]
    for i, measurements in enumerate(depth_measurements):
        std = np.std(measurements)
        mean = np.mean(measurements)
        in_std_mask = np.abs(measurements-mean) <= std
        depth_measurements = measurements[in_std_mask]
        #print(f"{depth_measurements=}")
        depth_val = sum(measurements)/len(measurements)
        #print(f"{depth_val=}")
        assert depth_val > 0, f"not able to get depth val after {n_depth_samples} samples {depth_val=}"
        final_depth_measurements[i] = depth_val
    return final_depth_measurements

def deproject_top_view_point(K, pixel_x, pixel_y, depth):
    return rs.rs2_deproject_pixel_to_point(K, [pixel_x, pixel_y], depth)

class observation:
    def __init__(self, str_label):
        self.str_label = str_label
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.xCenter = None
        self.yCenter = None
        self.sidelength = None
        self.pickPose = None
        self.placePose = None
        self.Mask = None
        self.observation_pose = None
        self.mask = None
        self.ImgFrameWorldCoord = None
    def update_observation(self, rs_wrapper, label_vit, sam_predictor, observation_position, display = False):
        rgb_img, depth_img = get_pictures(rs_wrapper)
        self.observation_pose = observation_position
        bbox = None
        with torch.no_grad():
            queries = [self.str_label]
            abbrevq = [self.str_label]
            bbox = label_vit.predict(rgb_img, querries=queries)
            #bbox = bbox[1][0].tolist()
        self.xmin = int(bbox[0])
        self.ymin = int(bbox[1])
        self.xmax = int(bbox[2])
        self.ymax = int(bbox[3])
        
        invalid_border_px_x = 200
        invalid_border_px_y = 20
        self.xmin = np.clip(self.xmin, 0+invalid_border_px_x, rgb_img.shape[1]-invalid_border_px_x)
        self.xmax = np.clip(self.xmax, 0+invalid_border_px_x, rgb_img.shape[1]-invalid_border_px_x)
        self.ymin = np.clip(self.ymin, 0+invalid_border_px_y, rgb_img.shape[0]-invalid_border_px_y)
        self.ymax = np.clip(self.ymax, 0+invalid_border_px_y, rgb_img.shape[0]-invalid_border_px_y)

        self.xCenter = int((self.xmin + self.xmax)/2)
        self.yCenter = int((self.ymin + self.ymax)/2)

        sam_predictor.set_image(rgb_img)
        sam_box = np.array([self.xmin,  self.ymin,  self.xmax,  self.ymax])
        sam_mask, sam_scores, sam_logits = sam_predictor.predict(box=sam_box)
        sam_mask = np.transpose(sam_mask, (1, 2, 0))
        self.mask = sam_mask
        depth_querry_list = [
            (self.xCenter, self.yCenter),
            (self.xmin, self.ymin),
            (self.xmax, self.ymin)
        ]
        center_depth, ll_depth, lr_depth = get_depths(depth_querry_list, rs_wrapper)
        _, K = get_depth_frame_intrinsics(rs_wrapper)
        self.ImgFrameWorldCoord = deproject_top_view_point(K, self.xCenter, self.yCenter, center_depth)
        LL_X, LL_Y, LL_Z = deproject_top_view_point(K, self.xmin, self.ymin, ll_depth)
        LR_X, LR_Y, LR_Z = deproject_top_view_point(K, self.xmax, self.ymin, lr_depth)
        sidelength =  (LL_X-LR_X)**2
        sidelength += (LL_Y-LR_Y)**2
        sidelength += (LL_Z-LR_Z)**2
        sidelength = np.sqrt(sidelength)
        self.sidelength = sidelength
        self.sidelength = 0.04

        if display:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            axes[0, 0].imshow(rgb_img)
            axes[1, 0].imshow(depth_img)

            axes[0, 0].add_patch(get_observation_patch(self, "r"))
            axes[1, 0].add_patch(get_observation_patch(self, "r"))

            axes[0, 0].text(self.xmin, self.ymin - 10, f"{self.str_label}", color='r', fontsize=12, ha='left', va='bottom')
            axes[1, 0].text(self.xmin, self.ymin - 10, f"{self.str_label}", color='r', fontsize=12, ha='left', va='bottom')

            axes[0, 1].imshow(sam_mask)

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    from magpie_control import realsense_wrapper as real
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from magpie_perception.label_owlv2 import LabelOWLv2

    myrs = real.RealSense()
    myrs.initConnection()

    label_vit = LabelOWLv2(topk=1, score_threshold=0.2, cpu_override=False)
    label_vit.model.eval()
    print(f"{label_vit.model.device=}")

    sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    print(f"{sam_predictor.model.device=}")


    red_block_obs = observation("green block")
    red_block_obs.update_observation(myrs, label_vit, sam_predictor, 0, display=True)
    
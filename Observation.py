import torch
import numpy as np
from control_scripts import get_pictures, get_frames
from config import n_depth_samples
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d

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
    print(f"{dir(depth_frame.profile.as_video_stream_profile())=}")
    print(f"{dir(depth_frame.profile)=}")

    depth_scale = rs_wrapper.device.first_depth_sensor().get_depth_scale()
    print(f"{depth_scale=}")
    return depth_scale, intrinsics

class observation:
    def __init__(self, str_label):
        self.str_label = str_label
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.xCenter = None
        self.yCenter = None
        
        self.observation_pose = None
        self.mask = None
        
        self.rgb_segment = None
        self.depth_segment = None
        self.pcd = None

        self.ImgFrameWorldCoord = None
        self.sidelength = None
        self.pickPose = None
        self.placePose = None

    def calc_bbox(self, label_vit, rgb_img):
        with torch.no_grad():
            bbox = label_vit.label(rgb_img, self.str_label, self.str_label, plot=False, topk=True)
            bbox = bbox[1][0].tolist()
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

    def calc_pc(self, sam_predictor, rgb_img, depth_img, K, depth_scale):
        sam_predictor.set_image(rgb_img)
        sam_box = np.array([self.xmin,  self.ymin,  self.xmax,  self.ymax])
        sam_mask, sam_scores, sam_logits = sam_predictor.predict(box=sam_box)
        #sam_mask = np.transpose(sam_mask, (1, 2, 0))
        sam_mask = np.all(sam_mask, axis=0)
        expanded_sam_mask = np.expand_dims(sam_mask, axis=-1)
        print(f"{sam_mask.shape=}")
        print(f"{sam_scores.shape=}")
        print(f"{sam_logits.shape=}")

        self.mask = sam_mask
        self.rgb_segment = rgb_img*expanded_sam_mask
        self.depth_segment = depth_img*sam_mask

        temp_rgb_img = o3d.geometry.Image(self.rgb_segment)
        temp_depth_img = o3d.geometry.Image(self.depth_segment)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(temp_rgb_img, temp_depth_img, depth_scale=depth_scale, depth_trunc=10.0)
        #print(f"{dir(K)=}")
        intrinsic = o3d.camera.PinholeCameraIntrinsic(K.width, K.height, K.fx, K.fy, K.ppx, K.ppy)
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        print(f"{dir(self.pcd)=}")
        print(f"n points = {len(self.pcd.points)}")
        o3d.visualization.draw_geometries([self.pcd])



    def update_observation(self, rs_wrapper, label_vit, sam_predictor, observation_position, display = False):
        rgb_img, depth_img = get_pictures(rs_wrapper)
        self.observation_pose = observation_position
        depth_scale, K = get_depth_frame_intrinsics(rs_wrapper)

        self.calc_bbox(label_vit, rgb_img)

        self.calc_pc(sam_predictor, rgb_img, depth_img, K, depth_scale)

        if display:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            axes[0, 0].imshow(rgb_img)
            axes[0, 0].add_patch(get_observation_patch(self, "r"))
            axes[0, 0].text(self.xmin, self.ymin - 10, f"{self.str_label}", color='r', fontsize=12, ha='left', va='bottom')

            axes[1, 0].imshow(self.mask)
            axes[0, 1].imshow(self.rgb_segment)
            axes[1, 1].imshow(self.depth_segment)

            


            plt.tight_layout()
            plt.show()

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


    myrs = real.RealSense()
    myrs.initConnection()

    label_vit = LabelOWLv2(topk=1, score_threshold=0.2, cpu_override=False)
    label_vit.model.eval()
    print(f"{label_vit.model.device=}")

    sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    print(f"{sam_predictor.model.device=}")

    goto_vec(myrobot, topview_vec)
    red_block_obs = observation("green block")
    red_block_obs.update_observation(myrs, label_vit, sam_predictor, 0, display=True)
    myrobot.stop()
    myrs.disconnect()
    
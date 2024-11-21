import math
topview_vec = [-0.16188220333609551, -0.6234229524443915, 0.5474838984217083, 1e-5, -math.pi, -1e-5]
sideview_vec =[0.0360674358115564, -0.20624107287146376, 0.2646274319314355, 1.8434675848139614, 1.4569842711938066, -1.2315497051361715]
#tcp_X_offset = 0.0
#tcp_Y_offset = 0.015
#tcp_Z_offset = 0.133 #Higher number is higher off the table
PC_X_offset = 0.16
PC_Y_offset = -0.16
PC_Z_offset = -0.09
realSenseFPS = 30
invalid_border_px_x = 250
invalid_border_px_y = 50 #limiting bounding box by units of pixels 
n_depth_samples = 5
arm_speed = 0.4
vit_thresh = 0.1
gpt_model = "gpt-4o"
gpt_temp = 0.2
tower = ["dark blue block","green block","yellow block", "red block"]
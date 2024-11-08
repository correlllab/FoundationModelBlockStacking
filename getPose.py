from magpie.ur5 import UR5_Interface as robot
from magpie.ur5 import pose_vector_to_homog_coord, homog_coord_to_pose_vector
if __name__ == "__main__":
    myrobot = robot()
    myrobot.start()
    starting_pose_matrix = myrobot.getPose()
    starting_pose_vector = homog_coord_to_pose_vector(starting_pose_matrix)
    print(f"{starting_pose_matrix=}")
    print(f"{starting_pose_vector=}")
    myrobot.stop()

import open3d as o3d
from SceneGraphGeneration import get_graph, get_geometries
from threading import Thread
import time

class Graph_Manager:
    def __init__(self):
        self.graph_history = []

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1000, height=1000, visible=True)

        print(f"{dir(self.vis)=}")

        #self.thread = Thread(target=self.update_display)
        #self.thread.daemon = True
        #self.thread.start()

    def update_display(self):
        print(f"update display called")
        #while True:
        if len(self.graph_history) == 0:
            geo = get_geometries(None)  # Get geometries for the initial graph
        else:
            geo = get_geometries(self.graph_history[-1])  # Get geometries for the latest graph
        
        print(f"{geo=}")

        # Clear previous geometry and add new geometries to the visualizer
        self.vis.clear_geometries()
        for i, geometry in enumerate(geo):
            self.vis.add_geometry(geometry)

        # Post to the main thread to update the display
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(1)  # Sleep to control the update frequency


    def add_graph(self, graph):
        self.graph_history.append(graph)
if __name__ == "__main__":
    from magpie_control import realsense_wrapper as real
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from magpie_perception.label_owlv2 import LabelOWLv2
    from magpie_control.ur5 import UR5_Interface as robot
    from control_scripts import goto_vec
    from APIKeys import API_KEY
    from openai import OpenAI
    from config import realSenseFPS, topview_vec

    

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

    gm = Graph_Manager()
    gm.update_display()
    
    input("Press Enter to continue...")
    graph = get_graph(client, label_vit, sam_predictor, myrs, myrobot)
    gm.add_graph(graph)
    gm.update_display()

    input("Press Enter to continue...")
    graph = get_graph(client, label_vit, sam_predictor, myrs, myrobot)
    gm.add_graph(graph)
    gm.update_display()


    myrobot.stop()
    myrs.disconnect()
    
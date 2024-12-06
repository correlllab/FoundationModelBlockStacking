import open3d as o3d
from SceneGraphGeneration import get_graph, get_geometries
from threading import Thread
import time

class Graph_Manager:
    def __init__(self):
        self.graph_history = []
        self.last_displayed_labels = []

        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()

        self.vis = o3d.visualization.O3DVisualizer(title="Scene Graph Visualizer", width=1000, height=1000)
        self.app.add_window(self.vis)

        self.app.run_in_thread(self.update_display)

    def update_display(self):
        while True:
            print(f"update display looped")
            if len(self.graph_history) == 0:
                geo = get_geometries(None)  # Get geometries for the initial graph
            else:
                geo = get_geometries(self.graph_history[-1])  # Get geometries for the latest graph
            
            #print(f"{geo=}")

            # Clear previous geometry and add new geometries to the visualizer
            #print(f"{dir(self.vis)=}")
            for old_geometry in self.last_displayed_labels:
                #print(f"removing {old_geometry}")
                self.vis.remove_geometry(old_geometry)

            self.last_displayed_labels = []
            for label, geometry in geo:
                self.vis.add_geometry(label, geometry)
                self.last_displayed_labels.append(label)

            self.vis.post_redraw()
            #self.app.post_to_main_thread(self.vis, self.vis.post_redraw)
            #self.app.run_one_tick()
            self.app.post_to_main_thread(self.vis, self.app.run_one_tick)
            
            time.sleep(0.05)

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
    inp = "a"
    while inp != "q":
        #graph = get_graph(client, label_vit, sam_predictor, myrs, myrobot)
        #gm.add_graph(graph)
        #gm.update_display()
        inp = input("press q to quit: ")
    myrobot.stop()
    myrs.disconnect()

    
    
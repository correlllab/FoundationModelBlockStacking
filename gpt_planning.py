from config import gpt_model
import json
from PIL import Image
import base64
import io

#Function that takes a list of blocks in order of the tower [red, green, blue] and produces a prompt to be given to the GPT4o in sideview
#https://platform.openai.com/docs/guides/structured-outputs/examples
def get_state_querry_prompt():
    system_prompt = ("""
You are a block stacking robot.
You should output a json output with fields:
objects:list of objects in the scene relevant to the block stacking task. include the table as an object.
object_relationships: list of <len(objects)> tuples of objects, where <OBJECT1, OBJECT2> means OBJECT1 is directly on top of OBJECT2. Include what objects are on the table in the object_relationships. Do not give transitive relationships, if block A is on block B an block B is on the table that does not mean block A is on the table""")
    user_prompt = f"Give me the state in the given image"
    return system_prompt, user_prompt

def get_instruction_prompt(str_list_stack_order, state_obj):
    system_prompt = ("""
You are a block stacking planner.
You should output a json output with fields:
explanation:str why this a good move to get to our desired tower.
next_step:json with identical fields to this one detailing what the next pick and place will be, next step should be None if Done 
pick:str object to be picked up.
place:str object to place the pick object on top of.
Done:0/1 boolean for whether or not the tower is complete.
This json will be consumed so that the pick block is ontop of the place object. Do not abbriviate pick and place strings. We do not care about the state of other objects outside of the tower we are trying to stack. The tower may already be partially stacked, completly unstacked, or even incorrectly stacked. Only place an object on the table to free the object underneath it.
""")
    
    user_prompt = f"""Give me the next step so the blocks are stacked with the {str_list_stack_order[0]} at the base of the tower"""
    for i in range(1, len(str_list_stack_order)):
        user_prompt += f""", and the {str_list_stack_order[i]} on the {str_list_stack_order[i-1]}"""
    user_prompt += "."

    assistant_prompt = """The objects are currently stacked as follows:\n"""
    for i in range(0,len(state_obj["object_relationships"])):
        assistant_prompt += f"""{state_obj["object_relationships"][i][0]} is on top of {state_obj["object_relationships"][i][1]}\n"""

    return system_prompt, user_prompt, assistant_prompt

#helper function that formats the image for GPT api
def encode_image(img_array):
    # Convert the ndarray to a PIL Image
    image = Image.fromarray(img_array)
    
    # Create a BytesIO object to save the image
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # Specify the format you want
    buffered.seek(0) #Possibly not needed
    # Get the byte data and encode to base64
    encoded_string = base64.b64encode(buffered.read()).decode('utf-8')
    
    return encoded_string

#api calling function
def get_gpt_next_instruction(client, rgb_image, desired_tower_order):
    image = encode_image(rgb_image)
    img_type = "image/jpeg"

    state_querry_system_prompt, state_querry_user_prompt = get_state_querry_prompt()
    print(f"{state_querry_system_prompt=}")
    print(f"{state_querry_user_prompt=}")
    state_response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            { "role": "system", "content":[{"type": "text", "text":f"{state_querry_system_prompt}"}]},  # Only text in the system message
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": state_querry_user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{encode_image(rgb_image)}"}}
                ]
            },
        ],
        response_format={"type": "json_object"}
    )
    state_json = json.loads(state_response.choices[0].message.content)
    instruction_system_prompt, instruction_user_prompt, instruction_assitant_prompt = get_instruction_prompt(desired_tower_order, state_json)
    print(f"{instruction_system_prompt=}")
    print(f"{instruction_user_prompt=}")
    print(f"{instruction_assitant_prompt=}")


    instruction_response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            { "role": "system", "content":[{"type": "text", "text":f"{instruction_system_prompt}"}]},
            { "role": "assistant", "content":[{"type": "text", "text":f"{instruction_assitant_prompt}"}]},
            { "role": "user", "content":[{"type": "text", "text":f"{instruction_user_prompt}"}]}
        ],
        response_format={"type": "json_object"}
    )
    instruction_json = json.loads(instruction_response.choices[0].message.content)
    return (state_response, state_json), (instruction_response, instruction_json)

    


if __name__ == "__main__":
    from APIKeys import API_KEY
    from control_scripts import goto_vec, get_pictures
    from magpie_control import realsense_wrapper as real
    from magpie_control.ur5 import UR5_Interface as robot
    from config import sideview_vec
    import matplotlib.pyplot as plt
    from openai import OpenAI




    myrs = real.RealSense()
    myrs.initConnection()
    myrobot = robot()
    myrobot.start()

    client = OpenAI(
        api_key= API_KEY,
    )
    goto_vec(myrobot, sideview_vec)
    rgb_img, depth_img = get_pictures(myrs)

    plt.figure()
    plt.imshow(rgb_img)
    plt.show(block = False)

    ##--string for GPT QUERY--##
    (state_response, state_json), (instruction_response, instruction_json) = get_gpt_next_instruction(client, rgb_img, ["green block", "blue block", "yellow block"])


    print()
    print(f"{state_json['objects']=}")
    print(f"{state_json['object_relationships']=}")

    print(f"{instruction_json['pick']=}")

    print(f"{instruction_json['place']=}")

    print(f"{instruction_json['Done']=}")

    print(f"{instruction_json['explanation']=}")
    #print(f"{instruction_json['next_step']=}")
    print()
    print(f"{state_response=}")
    print(f"{instruction_response=}")

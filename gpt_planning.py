from config import gpt_model, gpt_temp
import json
from PIL import Image
import base64
import io

#Function that takes a list of blocks in order of the tower [red, green, blue] and produces a prompt to be given to the GPT4o in sideview
#https://platform.openai.com/docs/guides/structured-outputs/examples
def get_state_querry_prompt():
    system_prompt = ("""
You are a block stacking robot. Your task is to analyze the scene, determine the objects present, and infer their relationships through detailed chain of thought reasoning.

# Instructions

You should output a JSON object containing the following fields:

- **objects**: A list of all objects visible in the scene that are relevant to the block stacking task. Make sure to include the table as one of the objects.
  
- **object_relationships**: A list of tuples describing relationships between the objects. Each tuple should be in the format `<OBJECT1, OBJECT2>`, where `OBJECT1` is directly on top of `OBJECT2`. Include relationships where the objects are directly on the table. Do not include transitive relationships; for example, if block A is on block B and block B is on the table, do not state that block A is on the table.

Ensure that every object is on at least one other object or the table. No object should be unplaced. 

# Chain of Thought Reasoning

1. **Identify Objects**: Begin by analyzing the scene to identify all visible objects relevant to the block stacking task. This includes blocks and the table itself.
  
2. **Determine Object Positions**: For each object, determine its placement in relation to other objects:
   - Is the object on another block or on the table?
   - Make sure no object is left unplaced.

3. **Establish Relationships**: Once object positions are determined, establish relationships following these rules:
   - Record relationships where one object is directly on top of another.
   - Each relationship is a pair `<OBJECT1, OBJECT2>`, where `OBJECT1` is directly above `OBJECT2`.
   - Avoid transitive relationships to ensure clarity. 

4. **Verify Completeness**: Ensure that all objects are covered in the relationships and that none remain without being stacked or placed on the table.

# Output Format

Your output should be formatted as a JSON object, like the example below:

```json
{
  "objects": ["table", "block A", "block B", "block C"],
  "object_relationships": [["block A", "block B"], ["block B", "table"], ["block C", "table"]]
}
```

Make sure the output JSON adheres strictly to the specified structure and validates that each object is accounted for in the relationships.

# Examples

**Input Scene Description**:
- Block A is on Block B.
- Block B is on the table.
- Block C is also on the table.

**Chain of Thought Reasoning**:
1. Identify Objects: The scene includes "Block A", "Block B", "Block C", and the "table".
2. Determine Object Positions:
   - Block A is on Block B.
   - Block B is on the table.
   - Block C is on the table.
3. Establish Relationships:
   - `<Block A, Block B>`
   - `<Block B, Table>`
   - `<Block C, Table>`

**Output JSON**:
```json
{
  "objects": ["table", "block A", "block B", "block C"],
  "object_relationships": [["block A", "block B"], ["block B", "table"], ["block C", "table"]]
}
```

# Notes

- The table itself should also be visible in the object list.
- Ensure no object is left unplaced; every object must be included in the relationships field either on another object or on the table.
- Follow the reasoning steps explicitly before outputting to ensure correctness and completeness.
""")
    user_prompt = f"Give me the state in the given image"
    return system_prompt, user_prompt

def get_instruction_prompt(str_list_stack_order, state_obj):
    system_prompt = ("""
You are a block stacking planner. Your job is to generate specific instructions to achieve a desired tower stack configuration using a backward planning approach. Note that the tower may already be partially, completely, or incorrectly built. Work backward from the final configuration, ensuring only essential movements are made to free other blocks or achieve the intended structure. The plan should be informed by the user-provided current state of the tower.

# Details

Your output should be a JSON with the following fields:

- **explanation**: A string explaining why this step is crucial in achieving the desired tower configuration, clarifying its role in the backward plan to achieve the target, considering the current state of the blocks provided by the user.
- **next_step**: JSON with identical fields to this one, detailing the next pick and place move. Set it to `null` if no further steps are required.
- **pick**: A string representing the object that should be picked up.
- **place**: A string representing the object on which to place the picked object.
- **Done**: A boolean (0 or 1) indicating whether the tower is complete (1) or not (0).
- **desired_order**: A list representing the order of blocks in their final configuration.

Ensure to update the "Done" field to **1 only when all blocks are correctly positioned according to the final tower state specified by the user**. Ensure that your reasoning clearly justifies how the arrangement completes the configuration.

The goal is to ensure that each block is positioned most efficiently to reach the desired tower configuration, avoiding redundant steps such as moving blocks onto the table more than necessary.

# Steps

1. **Determine Final Positioning**: Visualize the completed tower configuration, considering the placement of each block in its final structure.
2. **Work Backward**: Identify which block must be in which position just before the target is reached. Plan the task backward to understand which blocks need repositioning.
3. **Select Blocks to Move**: Pick blocks that need to move in order to progressively build the final configuration.
4. **Identify Required Location**: Place each block only where it will directly contribute to the final tower structure, avoiding unnecessary "table" placements unless used to free required blocks.
5. **Incorporate Current State**: Use the provided current state of the blocks to determine the most efficient move, ensuring compatibility with the provided tower conditions.
6. **Explain the Move**: Provide an explanation based on backward reasoning of why this move will advance toward the final configuration systematically, specifically referencing the current state provided by the user.
7. **Next Step Update**: Create a next move in the `next_step` field, if needed, repeating the format as above.
8. **Completion Status**: Set the `Done` field to **1** only if the final configuration is achieved, otherwise leave it as **0**.

# Output Format

Please provide output as a JSON object strictly following the format below:

```json
{
  "explanation": "string explaining why this move is crucial in backward planning towards the goal, taking into account the current state",
  "next_step": null or {
    "explanation": "string explaining the next step based on the backward planning and the current state",
    "pick": "object name",
    "place": "target object name",
    "Done": 0 or 1,
    "desired_order": ["list of blocks in final desired order"]
  },
  "pick": "object name to be picked",
  "place": "target object name to place on",
  "Done": 0 or 1,
  "desired_order": ["list of blocks in final desired order"]
}
```

# Example Output

Here is an example of the expected output:

```json
{
  "explanation": "To complete the final structure, block A needs to be moved to free access to block B, which must be stacked next. Currently, block A is blocking access to block B.",
  "next_step": {
    "explanation": "Now, pick block B to place it on block C to achieve its required position in the final configuration. Block B is currently on block A, so moving it is necessary.",
    "pick": "block B",
    "place": "block C",
    "Done": 0,
    "desired_order": ["A", "B", "C"]
  },
  "pick": "block A",
  "place": "table",
  "Done": 0,
  "desired_order": ["A", "B", "C"]
}
```

# Notes

- The starting point for each move must consider the **current state** provided by the user. The state may already be partially or incorrectly built.
- Follow a **backward reasoning approach** to identify the sequence of moves starting from the desired tower configuration.
- Ensure every pick-and-place action has a purposeful explanation about how it advances progress toward the goal.
- Do **not** place blocks on the table unless it is strictly necessary to free another block needed for the target configuration.
- Update the **Done** field only at the exact point the final configuration is realized, ensuring this value is not set prematurely.""")
    
    user_prompt = f"""Give me the next step so the blocks are stacked with the {str_list_stack_order[0]} at the base of the tower"""
    for i in range(1, len(str_list_stack_order)):
        user_prompt += f""", and the {str_list_stack_order[i]} on the {str_list_stack_order[i-1]}"""
    user_prompt += ".\n"

    user_prompt += """The objects are currently stacked as follows:\n"""
    for i in range(0,len(state_obj["object_relationships"])):
        user_prompt += f"""{state_obj["object_relationships"][i][0]} is on top of {state_obj["object_relationships"][i][1]}\n"""

    return system_prompt, user_prompt

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
    #print(f"{state_querry_system_prompt=}")
    #print()
    #print(f"{state_querry_user_prompt=}")
    #print()
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
        response_format={"type": "json_object"},
        temperature=gpt_temp
    )
    state_json = json.loads(state_response.choices[0].message.content)
    instruction_system_prompt, instruction_user_prompt = get_instruction_prompt(desired_tower_order, state_json)
    #print(f"{instruction_system_prompt=}")
    #print()
    print(f"{instruction_user_prompt=}")
    #print()
    #print(f"{instruction_assitant_prompt=}")


    instruction_response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            { "role": "system", "content":[{"type": "text", "text":f"{instruction_system_prompt}"}]},
            #{ "role": "assistant", "content":[{"type": "text", "text":f"{instruction_assitant_prompt}"}]},
            { "role": "user", "content":[{"type": "text", "text":f"{instruction_user_prompt}"}]}
        ],
        response_format={"type": "json_object"},
        temperature=gpt_temp
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
    print(f"starting robot from gpt planning")

    myrobot.start()

    client = OpenAI(
        api_key= API_KEY,
    )
    goto_vec(myrobot, sideview_vec)
    rgb_img, depth_img = get_pictures(myrs)

    

    ##--string for GPT QUERY--##
    tower = ["green block", "blue block", "yellow block"]
    (state_response, state_json), (instruction_response, instruction_json) = get_gpt_next_instruction(client, rgb_img, tower)

    print()
    #print(f"{state_response=}")
    print()
    #print(f"{instruction_response=}")
    print()
    print()
    print(f"{state_json['objects']=}")
    print(f"{state_json['object_relationships']=}")
    print()

    print()
    print(f"{instruction_json['pick']=}\n")

    print(f"{instruction_json['place']=}\n")

    print(f"{instruction_json['Done']=}\n")

    print(f"{instruction_json['explanation']=}\n")
    print(f"{instruction_json['next_step']=}\n")
    print(f"{instruction_json['desired_order']=}")
    print()
    #print(f"{dir(myrobot)=}")
    #print(f"{dir(myrs)=}")
    myrobot.stop()
    myrs.disconnect()
    plt.figure()
    plt.imshow(rgb_img)
    plt.show(block = False)
    input()
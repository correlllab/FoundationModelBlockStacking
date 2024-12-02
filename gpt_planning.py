from config import gpt_model, gpt_temp
import json
from PIL import Image
import base64
import io

#Function that takes a list of blocks in order of the tower [red, green, blue] and produces a prompt to be given to the GPT4o in sideview
#https://platform.openai.com/docs/guides/structured-outputs/examples
def get_state_querry_prompt():
    system_prompt = ("""
You are a robots eyes. Your task is to analyze the scene, determine the objects present, and infer their relationships through detailed chain of thought reasoning.

# Instructions

You should output a JSON object containing the following fields:

- **objects**: A list of all objects visible in the scene.
  
- **object_relationships**: A list of tuples describing relationships between the objects. Each tuple should be in the format `<OBJECT1, RELATIONSHIP, OBJECT2>`, where `OBJECT1` is related to `OBJECT2`. 

# Chain of Thought Reasoning

1. **Identify Objects**: Begin by analyzing the scene to identify all visible objects.
  
2. **Determine Object Positions**: For each object, determine its placement in relation to other objects:
   - Is the object on another block or on the table?
   - Make sure no object is left unplaced.

3. **Establish Relationships**: Once object positions are determined, establish relationships following these rules:
   - Record relationships where one object is directly on top of another.
   - Each relationship is a triple `<OBJECT1, RELATIONSHIP, OBJECT2>`, where `OBJECT1` is related to `OBJECT2`.

4. **Verify Completeness**: Ensure that all objects are covered in the relationships and that none remain without being stacked or placed on the table.

# Output Format

Your output should be formatted as a JSON object, like the example below:

```json
{
  "objects": ["table", "A", "B", "C"],
  "object_relationships": [["A", "is on", "B"], ["B", "is on", "table"], ["C", "is on", "table"], ["C", "is next to", "B"]]
}
```

Make sure the output JSON adheres strictly to the specified structure and validates that each object is accounted for in the relationships.

# Examples

**Input Scene Description**:
- A is on B.
- B is on the table.
- C is also on the table.
- C is next to B.

**Chain of Thought Reasoning**:
1. Identify Objects: The scene includes "A", "B", "C", and the "table".
2. Determine Object Positions:
   - A is on B.
   - B is on the table.
   - C is on the table.
   - C is next to B.
3. Establish Relationships:
   - `<A, is on, B>`
   - `<B, is on, Table>`
   - `<C, is on, Table>`
   - `<C, is next to, B>`
                     

**Output JSON**:
```json
{
  "objects": ["table", "A", "B", "C"],
  "object_relationships": [["A", "is on", "B"], ["B", "is on", "table"], ["C", "is on", "table"], ["C", "is next to", "B"]]
}
```

# Notes

- Ensure no object is left unplaced; every object must be included in the relationships field either on another object or on the table.
- Follow the reasoning steps explicitly before outputting to ensure correctness and completeness.
""")
    user_prompt = f"Give me the state in the given image"
    return system_prompt, user_prompt

def get_instruction_prompt(str_list_stack_order, state_obj, action_history, previous_plan):
    system_prompt = ("""
Generate specific instructions to achieve a desired tower stack configuration using a forward planning approach, adding a chain of thought reasoning for each step.

Your role is to act as a block-stacking planner who creates detailed sequential steps to build/modify a tower. The provided current state of the tower may vary, including situations where the tower is partially or incorrectly built. If the tower is already correctly configured, your first entry (entry `0`) should indicate no action is needed and mark the task as done.

# Details

- The plan should be informed by the user-provided **current state** of the tower.
- Use **forward planning**, breaking down how each move helps approach the final goal in a logical manner.

# Steps

- **Step-by-Step Chain of Thought**: Provide a thorough explanation of why each action is being taken and how it logically contributes to achieving the correct tower configuration.

- **Forward Reasoning Approach**:
    1. Identify the **current state** and determine the immediate discrepancies versus the **desired order**.
    2. Formulate practical steps, describing why each block needs to be moved, how it enables the next action, and how it advances toward the completion.
    3. Include each planned **pick** and **place** operation and the logical order they should follow, with reasoning detailed before each action.

# Output Format

The output should be a JSON object where each key is an integer indicating an order of execution, and the corresponding value is a JSON object detailing:
- **current state**: (string) explains what the current state is
- **done**: (boolean) indicates whether all conditions of the desired order are fulfilled.
- **explanation**: (string) explains the reasoning behind the action taken.
- **pick**: (string) specifies the block being picked in the current action.
- **place**: (string) specifies where the block is being placed.
- **end state**: (string) explains what the state should be after execution


# Notes

- The **done** field is updated only once the final desired configuration is achieved. It is set only in the step where the tower is complete.
- **Avoid Unnecessary Actions**: Refrain from moving blocks to intermediate places (like the table) unless required to access another needed block.
- **Sequence Consistency**: Always begin with entry `0` and incrementally proceed (`0`, `1`, `2`, `3`, etc.) without skipping entries.
- **Post-Completion Check**: Conduct a final check to ensure that the configuration matches the desired state before concluding the task.
- **If no steps are required because the desired configuration is already met, entry `0` should reflect that the tower is complete, and the `done` status should be `true`.
- **Pick and place strings should be concise nouns
- **DO NOT PLACE THINGS ON THE TABLE THAT ARE ALREADY ON THE TABLE
- **To place things on the table use the word "white paper" as the object name
                     
                     """)
    
    user_prompt = f"Give me the next step so the blocks are stacked with the {str_list_stack_order[0]} at the base of the tower"
    for i in range(1, len(str_list_stack_order)):
        user_prompt += f"\nthe {str_list_stack_order[i]} on the {str_list_stack_order[i-1]}"
    user_prompt += "."

    user_prompt += "The objects are currently stacked as follows:\n"
    for i in range(0,len(state_obj["object_relationships"])):
        user_prompt += f"   {state_obj['object_relationships'][i][0]} is on top of {state_obj['object_relationships'][i][1]}.\n"

    user_prompt +="\n"
    if len(action_history) > 0:
        user_prompt += "up until this point the actions you took in order were:\n"
        for i, action in enumerate(action_history):
            user_prompt += f"   {action}\n"
    
    if len(previous_plan) > 0:
        user_prompt += "before you took your last action your plan was to next:\n"
        for i, action in previous_plan:
            user_prompt += f"   {action}\n"

    

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
def get_state(client, rgb_image):
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
    return state_response, state_json, state_querry_system_prompt, state_querry_user_prompt

def get_instruction(client, desired_tower_order, state_json, action_history, previous_plan):
    instruction_system_prompt, instruction_user_prompt = get_instruction_prompt(desired_tower_order, state_json, action_history, previous_plan)

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
    min_key = min(instruction_json.keys(), key=lambda x: int(x))
    next_instruction_json = instruction_json.pop(min_key)
    future_instructions_json = [(k,v) for k, v in sorted(instruction_json.items(), key=lambda item: item[0])]

    return instruction_response, next_instruction_json, future_instructions_json, instruction_system_prompt, instruction_user_prompt


def get_gpt_next_instruction(client, rgb_image, desired_tower_order, action_history, previous_plan):
    state_response, state_json, state_querry_system_prompt, state_querry_user_prompt = get_state(client, rgb_image)
    instruction_response, next_instruction_json, future_instructions_json, instruction_system_prompt, instruction_user_prompt = get_instruction(client, desired_tower_order, state_json, action_history, previous_plan)
    
    return (state_response, state_json, state_querry_system_prompt, state_querry_user_prompt), (instruction_response, next_instruction_json, future_instructions_json, instruction_system_prompt, instruction_user_prompt)
    
def print_json(j, name=""):
    out_str = f"{name}={json.dumps(j, indent=4)}"
    print(out_str)
    return out_str


if __name__ == "__main__":
    from APIKeys import API_KEY
    from control_scripts import goto_vec, get_pictures
    from magpie_control import realsense_wrapper as real
    from magpie_control.ur5 import UR5_Interface as robot
    from config import frontview_vec
    import matplotlib.pyplot as plt
    from openai import OpenAI

    myrs = real.RealSense()
    myrs.initConnection()
    myrobot = robot()
    print(f"starting robot from gpt planning")

    myrobot.start()
    myrobot.open_gripper()

    client = OpenAI(
        api_key= API_KEY,
    )
    goto_vec(myrobot, frontview_vec)
    rgb_img, depth_img = get_pictures(myrs)

    

    ##--string for GPT QUERY--##
    tower = ["green block", "blue block", "yellow block"]
    action_history = []
    previous_plan = []
    for i in range(2):
        (state_response, state_json, _, _), (instruction_response, next_instruction, future_instructions, _, _) = get_gpt_next_instruction(client, rgb_img, tower, action_history, previous_plan)
        print("\n\n")
        print_json(state_json, name="state")
        print()
        print_json(next_instruction, name="next instruction")
        print()
        print_json(future_instructions, name="plan")
        action_history.append(next_instruction)
        previous_plan = future_instructions
        

    #print(f"{dir(myrobot)=}")
    #print(f"{dir(myrs)=}")
    myrobot.stop()
    myrs.disconnect()
    plt.figure()
    plt.imshow(rgb_img)
    plt.show(block = False)
    input()
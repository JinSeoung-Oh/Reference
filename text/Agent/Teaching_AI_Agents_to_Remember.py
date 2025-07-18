## Check this link for more understanding From https://towardsdatascience.com/key-insights-for-teaching-ai-agents-to-remember-c23deffe7f1a

[{
    "model": "gpt-4o",
    "api_key": "<YOUR API KEY>",
    "azure_endpoint": "<YOUR ENDPOINT>",
    "api_type": "azure",
    "api_version": "2024-06-01"
}]

!pip install ag2

business_info = """
# This manufacturing plant manufactures the following vehicle parts:
- Body panels (doors, hoods, fenders, etc.)
- Engine components (pistons, crankshafts, camshafts)
- Transmission parts
- Suspension components (springs, shock absorbers)
- Brake system parts (rotors, calipers, pads)

# This manufactoring plant produces parts for the following models:
- Ford F-150
- Ford Focus
- Ford Explorer
- Ford Mustang
- Ford Escape
- Ford Edge
- Ford Ranger

# Equipment for Specific Automotive Parts and Their Uses

## 1. Body Panels (doors, hoods, fenders, etc.)
- Stamping presses: Form sheet metal into body panel shapes
- Die sets: Used with stamping presses to create specific panel shapes
- Hydraulic presses: Shape and form metal panels with high pressure
- Robotic welding systems: Automate welding of body panels and structures
- Laser cutting machines: Precisely cut sheet metal for panels
- Sheet metal forming machines: Shape flat sheets into curved or complex forms
- Hemming machines: Fold and crimp edges of panels for strength and safety
- Metal finishing equipment (grinders, sanders): Smooth surfaces and remove imperfections
- Paint booths and spraying systems: Apply paint and protective coatings
- Drying ovens: Cure paint and coatings
- Quality control inspection systems: Check for defects and ensure dimensional accuracy

## 2. Engine Components (pistons, crankshafts, camshafts)
- CNC machining centers: Mill and drill complex engine parts
- CNC lathes: Create cylindrical parts like pistons and camshafts
- Boring machines: Enlarge and finish cylindrical holes in engine blocks
- Honing machines: Create a fine surface finish on cylinder bores
- Grinding machines: Achieve precise dimensions and smooth surfaces
- EDM equipment: Create complex shapes in hardened materials
- Forging presses: Shape metal for crankshafts and connecting rods
- Die casting machines: Produce engine blocks and cylinder heads
- Heat treatment furnaces: Alter material properties for strength and durability
- Quenching systems: Rapidly cool parts after heat treatment
- Balancing machines: Ensure rotating parts are perfectly balanced
- Coordinate Measuring Machines (CMMs): Verify dimensional accuracy

## 3. Transmission Parts
- Gear cutting machines: Create precise gear teeth on transmission components
- CNC machining centers: Mill and drill complex transmission housings and parts
- CNC lathes: Produce shafts and other cylindrical components
- Broaching machines: Create internal splines and keyways
- Heat treatment equipment: Harden gears and other components
- Precision grinding machines: Achieve extremely tight tolerances on gear surfaces
- Honing machines: Finish internal bores in transmission housings
- Gear measurement systems: Verify gear geometry and quality
- Assembly lines with robotic systems: Put together transmission components
- Test benches: Evaluate completed transmissions for performance and quality

## 4. Suspension Components (springs, shock absorbers)
- Coil spring winding machines: Produce coil springs to exact specifications
- Leaf spring forming equipment: Shape and form leaf springs
- Heat treatment furnaces: Strengthen springs and other components
- Shot peening equipment: Increase fatigue strength of springs
- CNC machining centers: Create precision parts for shock absorbers
- Hydraulic cylinder assembly equipment: Assemble shock absorber components
- Gas charging stations: Fill shock absorbers with pressurized gas
- Spring testing machines: Verify spring rates and performance
- Durability test rigs: Simulate real-world conditions to test longevity

## 5. Brake System Parts (rotors, calipers, pads)
- High-precision CNC lathes: Machine brake rotors to exact specifications
- Grinding machines: Finish rotor surfaces for smoothness
- Die casting machines: Produce caliper bodies
- CNC machining centers: Mill and drill calipers for precise fit
- Precision boring machines: Create accurate cylinder bores in calipers
- Hydraulic press: Compress and form brake pad materials
- Powder coating systems: Apply protective finishes to calipers
- Assembly lines with robotic systems: Put together brake components
- Brake dynamometers: Test brake system performance and durability

"""

business_rules_over50 = """
- The engine components are critical and machinery should be kept online that corresponds to producing these components when capacity constraint is more or equal to 50%: engine components
- Components for the following models should be prioritized when capacity constraint is more or equal to 50%: 1.Ford F-150
"""

business_rules_25to50 = """
- The following components are critical and machinery should be kept online that corresponds to producing these components when capacity constraint is between 25-50%: engine components and transmission parts 
- Components for the following models should be prioritized when capacity constraint is between 25-50%: 1.Ford F-150 2.Ford Explorer
"""

business_rules_0to25 = """
- The following components are critical and machinery should be kept online that corresponds to producing these components when capacity constraint is between 0-25%: engine components,transmission parts, Brake System Parts
- Components for the following models should be prioritized when capacity constraint is between 0-25%: 1.Ford F-150 2.Ford Explorer 3.Ford Mustang 4.Ford Focus
"""

import autogen
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen import ConversableAgent, UserProxyAgent

config_list = autogen.config_list_from_json(
    env_or_file="autogenconfig.json", #the json file name that stores the config 
    file_location=".", #this means the file is in the same directory
    filter_dict={
        "model": ["gpt-4o"], #select a subset of the models in your config
    },
)

teachable_agent = ConversableAgent(
    name="teachable_agent",  # the name can't contain spaces
    llm_config={"config_list": config_list, "timeout": 120, "cache_seed": None},  # in this example we disable caching but if it is enabled it caches API requests so that they can be reused when the same request is used
) 

user = UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS", #I want to have full control over the code executed so I am setting human_input_mode to ALWAYS. Other options are NEVER and TERMINATE.
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False, #setting a termination  message is VERY important because it tells the agent when to finish.
    max_consecutive_auto_reply=0, #we don't need this agent to replies multiple times in a row
    code_execution_config={
        "use_docker": False
    },  # If you are planning on running code look into setting use_docker=True. For this example I am not because then I have to walk through the docker setup, but it is safer than running the code directly.
)

#simple question
user.initiate_chat(teachable_agent, message="The facility is experiencing a power shortage of 40%. What models need to be prioritized?", clear_history=True)
#multistep complex question
user.initiate_chat(teachable_agent, message="The facility is experiencing a power shortage of 30%. Provide me a detailed breakdown of what machines should be deactivated and which machines should remain active.", clear_history=True)

teachability = Teachability(
    verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=True, # we want to reset the db because we are creating a new agent so we don't want any existing memories. If we wanted to use an existing memory store we would set this to false.
    path_to_db_dir="./tmp/notebook/teachability_db", #this is the default path you can use any path you'd like
    recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
    max_num_retrievals=10 #10 is default bu you can set the max number of memos to be retrieved lower or higher
)

teachability.add_to_agent(teachable_agent)

user.initiate_chat(teachable_agent, message=business_info, clear_history=True)
user.initiate_chat(teachable_agent, message=business_rules_over50, clear_history=True)
user.initiate_chat(teachable_agent, message=business_rules_25to50, clear_history=True)
user.initiate_chat(teachable_agent, message=business_rules_0to25, clear_history=True)


#simple question
user.initiate_chat(teachable_agent, message="The facility is experiencing a power shortage of 40%. What models need to be prioritized?", clear_history=True)
#multistep complex question
user.initiate_chat(teachable_agent, message="The facility is experiencing a power shortage of 30%. Provide me a detailed breakdown of what machines should be deactivated and which machines should remain active.", clear_history=True)




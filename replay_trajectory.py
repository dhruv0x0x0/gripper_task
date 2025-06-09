import h5py
from mani_skill.utils import common, io_utils, wrappers
import gymnasium as gym
import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
import time
import argparse
import sim_env.maniskill_env.envs
import cv2
#from einops import rearrange
import numpy as np
import sapien
np.set_printoptions(precision=3, suppress=True)
def get_relevant_info(ori_env):
    initial_pose = np.array(
        [
            [1, 0, 0.1, 0.097],
            [0, 1.0, 0.0, 0.0],
            [-0.1, 0.0, 1, 0.378],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    obs = ori_env.get_obs()
    image_dict = ori_env.get_sensor_images()
    wrist_cam = image_dict['wrist_cam']['rgb'][0].cpu().numpy()
    frame = cv2.cvtColor(wrist_cam,cv2.COLOR_RGB2BGR)
    #frame = cv2.resize(wrist_cam,(1280,720))
    new_frame = np.zeros((400,400,3),np.uint8)
    pad = int((400-224)/2)
    new_frame[pad:224+pad,:,:] = frame
    new_frame = cv2.resize(new_frame,(224,224),interpolation=cv2.INTER_AREA)
    tcp_pose = obs['extra']["tcp_pose"][0]
    gripper_state = obs['agent']["qpos"][0,7]/0.04
    tcp_pose_matrix = sapien.Pose(p=tcp_pose[:3],q=tcp_pose[3:]).to_transformation_matrix()
    rel_tcp_pose_matrix = np.linalg.inv(initial_pose) @ tcp_pose_matrix
    return new_frame, rel_tcp_pose_matrix,gripper_state

# traj_path = "monster_move_arc/2025_03_01_20_23_13_PickAnything.h5"
#traj_path = "monster_move_arc_non_upright/2025_03_01_20_44_13_PickAnything.h5"#args.traj_path
traj_path = "monster_row_wise/2025_02_28_18_27_44_PickAnything.h5"#"raw_data/ball_rows/2025_03_04_14_55_10_PickAnything.h5"#"monster_move_arc/2025_03_01_20_23_13_PickAnything.h5"
ori_h5_file = h5py.File(traj_path, "r")

# Load ass ociated json
json_path = traj_path.replace(".h5", ".json")
json_data = io_utils.load_json(json_path)
json_data['env_info']["env_kwargs"]['show_goal_site']= False
#json_data['env_info']["env_kwargs"]['spawn_goal_site']= False

env_info = json_data["env_info"]
print(env_info)
env_id = env_info["env_id"]
ori_env_kwargs = env_info["env_kwargs"]



ori_env = gym.make(env_id, **ori_env_kwargs)
# for i in range(100):
#     pose = ori_env.get_link_pose("link06")
#     root_pose = ori_env.get_link_pose("link00")
#     pp = root_pose.inv() * pose
#     print(pp)
#     action = np.zeros(8)
#     action[:3] = pp.p
#     action[3:7] = pp.q
#     action[7] = 1.0
#     obs, reward, terminated, truncated, info = ori_env.step(action)
#     #env.render()
#     #time.sleep(0.1)


def print_structure(data, indent=0):
    spacing = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spacing}{key}:")
            print_structure(value, indent + 1)
    elif isinstance(data, np.ndarray):
        print(f"{spacing}array of shape {data.shape}", data)
    elif isinstance(data, list):
        print(f"{spacing}list of length {len(data)}")
        for i, item in enumerate(data):
            print(f"{spacing}[{i}]:")
            print_structure(item, indent + 2)
    else:
        print(f"{spacing}{type(data).__name__}")

for ix in range(len(json_data["episodes"])):
    episode = json_data["episodes"][ix] # picks the first
    ori_env.reset(**episode["reset_kwargs"])
    ori_env_states = trajectory_utils.dict_to_list_of_dicts(
                    ori_h5_file[f"traj_{ix}"]["env_states"]
                    )
    ori_actions = trajectory_utils.dict_to_list_of_dicts(
                    ori_h5_file[f"traj_{ix}"]["actions"]
                    )
    print(episode['episode_len'])
    #ori_env.set_state_dict(ori_env_states[0])
    for i in range(episode['episode_len']-2):
       if i<len(ori_actions):
        #print(ori_env.step(ori_actions[i]))
        
        #print(obs)
        a,b,c= get_relevant_info(ori_env)

        agent = ori_env.agent  # floating_gripper_v1
        #new_frame = rearrange(new_frame,'h w c -> c h w')
        cv2.imshow("wrist",a)
        cv2.waitKey(1)
        #print(a.shape, b,c)
        ori_env.set_state_dict(ori_env_states[i])
        #print(ori_env_states[i]['articulations']['floating_gripper_v1'])#, "\naction\n" ,ori_actions[i], "\n")
        #print(ori_env_states[i]['actors']['hero_object'])
        ori_env.render()
        time.sleep(0.01)
        # obs = ori_env._get_obs()
        # if "agent" in obs:
        #     agent_obs = obs["agent"]
        #     print("Agent Observations (Z1_arm):")
        #     print_structure(agent_obs) #prints the structure and data inside the agent observations.
        # else:
        #     print("Agent observations not found in the environment.")

#position, orie, width, frame

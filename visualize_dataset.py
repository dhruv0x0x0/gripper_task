import argparse
import time
import h5py
import cv2
import numpy as np
import sapien
import rerun as rr
import gymnasium as gym
from mani_skill.utils import io_utils
from mani_skill.trajectory import utils as trajectory_utils
import mani_skill.envs
import time
import argparse
import sim_env.maniskill_env.envs
import cv2
from einops import rearrange
from collections import deque 
np.set_printoptions(precision=3, suppress=True)
def get_relevant_info(ori_env):
    initial_pose = np.array([
        [1.0,  0.0,  0.1, 0.097],
        [0.0,  1.0,  0.0, 0.0  ],
        [-0.1, 0.0,  1.0, 0.378],
        [0.0,  0.0,  0.0, 1.0  ],
    ])

    obs = ori_env.get_obs()
    images = ori_env.get_sensor_images()
    wrist_rgb = images['wrist_cam']['rgb'][0].cpu().numpy()

    frame = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)

    new_frame = np.zeros((400, 400, 3), np.uint8)
    pad = (400 - 224) // 2
    new_frame[pad:pad+224] = frame
    new_frame = cv2.resize(new_frame, (224, 224), interpolation=cv2.INTER_AREA)

    tcp = obs['extra']['tcp_pose'][0]
    grip_fraction = obs['agent']['qpos'][0, 7] / 0.04  # normalized width

    tcp_mat = sapien.Pose(p=tcp[:3], q=tcp[3:]).to_transformation_matrix()
    rel_tcp = np.linalg.inv(initial_pose) @ tcp_mat

    return np.array(new_frame), np.array(rel_tcp), grip_fraction.float()


def main(traj_path: str):
    rec = rr.RecordingStream(application_id="tcp_visualizer")
    rec.spawn()
    rr.set_global_data_recording(rec)
    h5_file = h5py.File(traj_path, 'r')
    json_path = traj_path.replace('.h5', '.json')
    json_data = io_utils.load_json(json_path)
    env_info = json_data['env_info']
    env_info['env_kwargs']['render_mode'] = "rgb_array"
    ori_env = gym.make(env_info['env_id'], **env_info['env_kwargs'])

    trail = deque(maxlen=30)

    ix = 0  #set ep no. 
    ep = json_data['episodes'][ix]
    ori_env.reset(**ep['reset_kwargs'])

    states = trajectory_utils.dict_to_list_of_dicts(
        h5_file[f"traj_{ix}"]["env_states"]
    )

    for i in range(ep['episode_len'] - 2):
        frame, tcp_pose, grip = get_relevant_info(ori_env)
        cv2.imshow('wrist_cam', frame)
        cv2.waitKey(1)

        origin = tcp_pose[:3, 3]
        rot = tcp_pose[:3, :3]
        trail.append(origin)

        axes_orig = np.tile(origin, (3, 1))
        axes_vecs = np.stack([rot[:, 0]/10,  rot[:, 1]/20,  rot[:, 2]/20], axis=0)
        axes_colors = [[1,0,0,0.5], [0,1,0,0.5], [0,0,1,0.5]]
        rec.log("tcp/axes", rr.Arrows3D(origins=axes_orig, vectors=axes_vecs, colors=axes_colors))

        w = ((grip*0.04)/2.0).detach().cpu().numpy()
        o = origin + rot[:, 0]/10
        l = o - w*rot[:, 1]
        r = o + w*rot[:, 1]
        rec.log("tcp/gripper", rr.LineStrips3D(
            np.stack([l, r]), colors=[[1,1,0,0.5]]
        ))

        if len(trail) > 1:
            poss = np.stack(trail)
            rec.log("tcp/trail", rr.LineStrips3D(
                [poss], colors=[[1,0.5,0.5,0.5]]
            ))

        # video_asset = rr.AssetVideo(path="/home/dhruv0x0x0/sim-env")
        # rr.log("video", video_asset, static=True)

        ori_env.set_state_dict(states[i])
        ori_env.render()
        time.sleep(0.1)

    h5_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--traj_path', type=str,
        default="monster_row_wise/2025_02_28_18_27_44_PickAnything.h5",
        help='traj_path'
    )
    args = parser.parse_args()
    main(args.traj_path)

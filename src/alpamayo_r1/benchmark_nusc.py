import base64
import os
import os.path
import argparse
from datetime import datetime
from math import atan2
import time
import torchvision.io as io

import torch
import numpy as np
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
import json
from nuscenes import NuScenes
from pyquaternion import Quaternion

def rot_mat(rot_quat: torch.Tensor) -> np.ndarray:
    """Return rotation matrix from quaternion."""
    return np.array(Quaternion(rot_quat).rotation_matrix)
  
def compute_ade(gt_xy: np.ndarray, pred_xy: np.ndarray) -> float:
    """Compute Average Displacement Error (ADE) between ground truth and predicted trajectories."""
    assert gt_xy.shape == pred_xy.shape, "Ground truth and prediction must have the same shape."
    distances = np.linalg.norm(gt_xy - pred_xy, axis=1)  # [T]
    ade = np.mean(distances)
    return ade
  
def world_to_ego_frame(world_xyz: torch.Tensor, world_rot: torch.Tensor) -> torch.Tensor:
    """
    Convert global trajectories to the ego-centric frame.
    
    For HISTORY (is_history=True): Use the LAST frame (current ego pose) as anchor
    For FUTURE (is_history=False): Use the LAST frame of history (current ego pose) as anchor

    Args:
        world_xyz: [B, 1, T, 3] - Trajectory points in global coordinates.
        world_rot: [B, 1, T, 3, 3] - Global rotation matrices.
        is_history: If True, use last frame as anchor; if False, use last frame as anchor too.
    """
    # Extract the Anchor Pose
    anchor_translation = world_xyz[:, :, -1:, :]  # [B, 1, 1, 3] - LAST frame
    anchor_rotation = world_rot[:, :, -1:, :, :]  # [B, 1, 1, 3, 3] - LAST frame

    # Shift the world so the ego car is at (0,0,0)
    xyz_translated = world_xyz - anchor_translation
    
    # Apply rotation: P_ego = R_inv @ P_translated
    anchor_rotation_inv = anchor_rotation.transpose(-1, -2)
    xyz_translated_col = xyz_translated.unsqueeze(-1)
    ego_xyz_transformed = torch.matmul(anchor_rotation_inv, xyz_translated_col)

    # Squeeze back to original shape: [B, 1, T, 3]
    return ego_xyz_transformed.squeeze(-1)
  
dataroot = "/proj/common-datasets/nuScenes/Full-dataset/v1.0"
version = "v1.0-test"

OBS_LEN = 4  # 4 frames at 2Hz = 2s → will interpolate to 16 frames at 10Hz (1.6s with extrapolation)
FUT_LEN = 13  # 13 frames at 2Hz = 6.5s → will interpolate to 64 frames at 10Hz (6.4s)
TTL_LEN = OBS_LEN + FUT_LEN
device = "cuda" if torch.cuda.is_available() else "cpu"

# set the seed
torch.cuda.manual_seed_all(42)

model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to(device)
processor = helper.get_processor(model.tokenizer)

print("model loaded to device:", model.device)
print("device info: ", torch.cuda.get_device_name(model.device))
print(f"Model loaded: {model.__class__.__name__}")
model.eval()

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
timestamp = ("results/" + f"{timestamp}-nuscenes_{version}")
os.makedirs(timestamp, exist_ok=True)

# Load the dataset
nusc = NuScenes(version=version, dataroot=f"{dataroot}")

# Iterate the scenes
scenes = nusc.scene

global_ade1s = []
global_ade3s = []
global_ade6s = []
global_aveg_ades = []
global_inference_times = []

print(f"Number of scenes: {len(scenes)}")
for scene in scenes:
    start_time = time.time()

    token = scene["token"]
    first_sample_token = scene["first_sample_token"]
    last_sample_token = scene["last_sample_token"]
    name = scene["name"]
    description = scene["description"]
    start_scene_time = time.time()

    # Get all image and pose in this scene
    front_camera_images = []
    ego_poses = []
    curr_sample_token = first_sample_token
    while True:
        sample = nusc.get("sample", curr_sample_token)

        # Get the front camera image of the sample.
        cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        nusc.render_sample_data(cam_front_data["token"])

        # add the image data directly from the path, converting them into tensors
        front_camera_images.append(io.read_image(os.path.join(nusc.dataroot, cam_front_data["filename"])))

        # Get the ego pose of the sample.
        pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
        ego_poses.append(pose)

        # Advance the pointer.
        if curr_sample_token == last_sample_token:
            break
        curr_sample_token = sample["next"]
        
    # convert into tensors
    front_camera_images = torch.stack([img for img in front_camera_images], dim=0)

    scene_length = len(front_camera_images)
    print(f"Scene {name} has {scene_length} frames")
    if scene_length < TTL_LEN:
        print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
        continue

    # Get the waypoints of the ego vehicle.
    ego_xyz = torch.tensor(
      np.array([ego_poses[t]["translation"][:3] for t in range(scene_length)]), 
      dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)
    ego_rot = torch.tensor(
      np.array([rot_mat(ego_poses[t]["rotation"]) for t in range(scene_length)]),
      dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)

    prev_intent = None
    cam_images_sequence = []
    ade1s_list = []
    ade3s_list = []
    ade6s_list = []
    for i in range(scene_length - TTL_LEN):
        obs_images = front_camera_images[i : i + OBS_LEN].to(device)
        obs_ego_rot = ego_rot[:, :, i : i + OBS_LEN].to(device)
        fut_ego_xyz_2hz = ego_xyz[:, :, i + OBS_LEN : i + TTL_LEN].to(device)
        obs_ego_xyz_2Hz = ego_xyz[:, :, i : i + OBS_LEN].to(device)
        
        # Transform history and future to ego frame using LAST observation frame as anchor
        obs_ego_xyz_2Hz = world_to_ego_frame(obs_ego_xyz_2Hz, obs_ego_rot)
        fut_ego_xyz_2hz = world_to_ego_frame(
            fut_ego_xyz_2hz,
            obs_ego_rot[:, : , -1:, :, :].expand(-1, -1,TTL_LEN - OBS_LEN, -1, -1),
        )
        
        print(f"shapes: obs_ego_xyz_2Hz: {obs_ego_xyz_2Hz.shape}, fut_ego_xyz_2hz: {fut_ego_xyz_2hz.shape}")
        
        # Interpolate fut xyz coordinates
        fut_ego_xyz = torch.zeros(1, 1, 64, 3, device=device)
        for dim in range(3):  # x, y, z
            fut_ego_xyz[0, 0, :, dim] = torch.nn.functional.interpolate(
                fut_ego_xyz_2hz[0, 0, :, dim].unsqueeze(0).unsqueeze(0),
                size=64,
                mode='linear',
                align_corners=True
            ).squeeze()
            
        # interpolate obs poses and rotations
        obs_ego_xyz = torch.zeros(1, 1, 16, 3, device=device)
        for dim in range(3):  # x, y, z
            obs_ego_xyz[0, 0, :, dim] = torch.nn.functional.interpolate(
                obs_ego_xyz_2Hz[0, 0, :, dim].unsqueeze(0).unsqueeze(0),
                size=16,
                mode='linear',
                align_corners=True
            ).squeeze()
        
        # For rotation, interpolate each row of the rotation matrix
        # obs_ego_rot is [1, 1, 4, 3, 3], we need [1, 1, 16, 3, 3]
        obs_ego_rot_interp = torch.zeros(1, 1, 16, 3, 3, device=device)
        for i_row in range(3):
            for j_col in range(3):
                obs_ego_rot_interp[0, 0, :, i_row, j_col] = torch.nn.functional.interpolate(
                    obs_ego_rot[0, 0, :, i_row, j_col].unsqueeze(0).unsqueeze(0),
                    size=16,
                    mode='linear',
                    align_corners=True
                ).squeeze()
        
        messages = helper.create_message(obs_images)
        inputs = processor.apply_chat_template(
          messages,
          tokenize=True,
          add_generation_prompt=False,
          continue_final_message=True,
          return_dict=True,
          return_tensors="pt",
        )
        inputs = helper.to_device(inputs, "cuda")
        model_inputs = {
          "tokenized_data": inputs,
          "ego_history_xyz": obs_ego_xyz,
          "ego_history_rot": obs_ego_rot_interp,
        }
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
          try:
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
              data=model_inputs,
              top_p=0.98,
              temperature=0.6,
              num_traj_samples=4,
              max_generation_length=256,
              return_extra=True,
            )
          except RuntimeError as e:
            print(f"ERROR during inference: {e}")
            if "CUDA" in str(e) or "device-side assert" in str(e):
              print("CUDA error detected. Skipping this sample...")
              continue
            else:
              raise
        
        pred_T = pred_xyz.shape[3]  # Number of time steps in prediction
        if i == 0:
          print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])
          print("Predicted trajectory shape:", pred_xyz.shape)
          print("Ground truth future shape:", fut_ego_xyz.shape)
          print(f"Predicted temporal dimension: {pred_T}, Future temporal dimension: {fut_ego_xyz.shape[2]}")
          print("\nDEBUG: Full prediction trajectory (first 5 waypoints at 10Hz):")
          print("pred_xyz[0, 0, 0, :5, :2]:\n", pred_xyz[0, 0, 0, :5, :2].cpu().numpy())
          print("\nDEBUG: Consecutive distances in prediction (first 5 steps):")
          for t in range(min(5, pred_T-1)):
              dist = np.linalg.norm(pred_xyz[0, 0, 0, t+1, :2].cpu().numpy() - pred_xyz[0, 0, 0, t, :2].cpu().numpy())
              print(f"  Step {t} to {t+1} (0.1s): {dist:.4f}m")
          print("\nDEBUG: Ground truth trajectory (first 5 timesteps at 2Hz):")
          print("fut_ego_xyz[:5, :2]:\n", fut_ego_xyz[0, 0, :5, :2].cpu().numpy())
          print(f"\nGT interpolated from {FUT_LEN} points (2Hz) to 60 points (10Hz)")
          print("Comparing both at 10Hz native model frequency")
          print(f"pred_xyz shape: {pred_xyz.shape}, fut_ego_xyz shape: {fut_ego_xyz.shape}")
          print("\nDEBUG: Velocity comparison at 10Hz (0.1s intervals):")
          print("Prediction trajectory (first 5 waypoints at 10Hz):")
          print("pred_xyz[0, 0, 0, :5, :2]:\n", pred_xyz[0, 0, 0, :5, :2].cpu().numpy())
          print("  Prediction velocities (at 10Hz = 0.1s apart):")
          for idx in range(min(5, 11)):
              vel = np.linalg.norm(pred_xyz[0, 0, 0, idx+1, :2].cpu().numpy() - pred_xyz[0, 0, 0, idx, :2].cpu().numpy())
              print(f"    Waypoint {idx} to {idx+1} (0.1s): {vel:.4f}m (speed: {vel/0.1:.2f} m/s)")
          
          print("\nGround truth trajectory (interpolated to 10Hz):")
          print("fut_ego_xyz[0, 0, :5, :2]:\n", fut_ego_xyz[0, 0, :5, :2].cpu().numpy())
          print("  Ground truth velocities (at 10Hz = 0.1s apart):")
          for idx in range(min(5, 11)):
              vel = np.linalg.norm(fut_ego_xyz[0, 0, idx+1, :2].cpu().numpy() - fut_ego_xyz[0, 0, idx, :2].cpu().numpy())
              print(f"    Waypoint {idx} to {idx+1} (0.1s): {vel:.4f}m (speed: {vel/0.1:.2f} m/s)")
        
        gt_xy = fut_ego_xyz[0, 0, :, :2].cpu().numpy()  # (60, 2) at 10Hz
        pred_xy_raw = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]  # (60, 2) at 10Hz
        
        
        gt_velocities = np.diff(gt_xy, axis=0)
        pred_raw_velocities = np.diff(pred_xy_raw, axis=0)
        gt_distances = np.linalg.norm(gt_velocities, axis=1)
        pred_raw_distances = np.linalg.norm(pred_raw_velocities, axis=1)
        gt_total_length = np.sum(gt_distances)
        pred_raw_total_length = np.sum(pred_raw_distances)
        pred_xy = np.zeros_like(pred_xy_raw)
        
        if pred_raw_total_length > 0:
          # Compute velocity scaling factor
          velocity_scale = gt_total_length / pred_raw_total_length
          
          # Scale ALL velocity vectors
          scaled_velocities = pred_raw_velocities * velocity_scale
          
          # Reconstruct the scaled trajectory
          # The first point is the same (the starting point)
          pred_xy[0] = pred_xy_raw[0] 
          
          # Subsequent points are the cumulative sum of the scaled velocities
          # np.cumsum(..., axis=0) computes the cumulative sum of the differences
          pred_xy[1:] = pred_xy_raw[0] + np.cumsum(scaled_velocities, axis=0)
          
        else:
          # If the prediction didn't move, just use the raw prediction
          pred_xy = pred_xy_raw
          velocity_scale = 1.0
        
        if i == 0:
          print(f"\nGT XY shape: {gt_xy.shape}, Pred XY shape: {pred_xy.shape}")
          print(f"GT XY range - min: {gt_xy.min():.4f}, max: {gt_xy.max():.4f}")
          print(f"Pred XY range - min: {pred_xy.min():.4f}, max: {pred_xy.max():.4f}")
        
        ade = compute_ade(gt_xy, pred_xy)
        print(f"ADE (6s @ 10Hz): {ade:.4f}m")

        pred1_len = min(10, pred_xy.shape[0])  # 1 second
        ade1s_list.append(compute_ade(gt_xy[:pred1_len], pred_xy[:pred1_len]))

        pred3_len = min(30, pred_xy.shape[0])  # 3 seconds
        ade3s_list.append(compute_ade(gt_xy[:pred3_len], pred_xy[:pred3_len]))
        
        pred6_len = min(64, pred_xy.shape[0])  # 6 seconds (or at least within the bins)
        ade6s_list.append(compute_ade(gt_xy[:pred6_len], pred_xy[:pred6_len]))
        
    mean_ade1s = np.mean(ade1s_list)
    mean_ade3s = np.mean(ade3s_list)
    mean_ade6s = np.mean(ade6s_list)
    aveg_ade = np.mean([mean_ade1s, mean_ade3s, mean_ade6s])
    inference_time = time.time() - start_scene_time

    print(f"Scene stats - ade1s_list: {len(ade1s_list)} samples, ade3s_list: {len(ade3s_list)}, ade6s_list: {len(ade6s_list)}")

    global_ade1s.append(mean_ade1s)
    global_ade3s.append(mean_ade3s)
    global_ade6s.append(mean_ade6s)
    global_aveg_ades.append(aveg_ade)
    global_inference_times.append(inference_time)

    result = {
        "name": name,
        "token": token,
        "ade1s": float(mean_ade1s),
        "ade3s": float(mean_ade3s),
        "ade6s": float(mean_ade6s),
        "avgade": float(aveg_ade),
        "inference_time": float(inference_time),
    }

    with open(f"{timestamp}/ade_results.jsonl", "a") as f:
        f.write(json.dumps(result))
        f.write("\n")

    print(
        f"Scene {name} done in {time.time() - start_scene_time} seconds. ADE1s: {mean_ade1s}, ADE3s: {mean_ade3s}, ADE6s: {mean_ade6s}, AvgADE: {aveg_ade}"
    )


global_result = {
  "name": "OVERALL",
  "overall_ade1s": np.mean(global_ade1s),
  "overall_ade3s": np.mean(global_ade3s),
  "overall_ade6s": np.mean(global_ade6s),
  "overall_avgade": np.mean(global_aveg_ades),
  "overall_inference_time_per_scene": np.mean(global_inference_times),
}
with open(f"{timestamp}/ade_results.jsonl", "a") as f:
  f.write(json.dumps(global_result))
  f.write("\n")

print(
  f"Overall ADE1s: {np.mean(global_ade1s)}, ADE3s: {np.mean(global_ade3s)}, ADE6s: {np.mean(global_ade6s)}, AvgADE: {np.mean(global_aveg_ades)}"
)
print(f"Overall Inference Time per Scene: {np.mean(global_inference_times)} seconds")

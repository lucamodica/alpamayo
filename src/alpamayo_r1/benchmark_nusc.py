import os
import os.path
from datetime import datetime
import time
import torchvision.io as io
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
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


def world_to_ego_frame(
    world_xyz: torch.Tensor,
    anchor_translation: torch.Tensor,
    anchor_rotation: torch.Tensor,
) -> torch.Tensor:
    """
    Convert global trajectory points to the ego-centric frame using an explicit anchor pose.

    Args:
        world_xyz: [B, 1, T, 3] - Points in global coordinates.
        anchor_translation: [B, 1, 1, 3] - Anchor translation in global frame.
        anchor_rotation: [B, 1, 1, 3, 3] - Anchor rotation (ego->world) in global frame.
    """
    # Translate so anchor is at origin
    xyz_translated = world_xyz - anchor_translation

    # Rotate into anchor's ego frame: P_ego = R_anchor^T @ (P_world - T_anchor)
    anchor_rotation_inv = anchor_rotation.transpose(-1, -2)
    xyz_translated_col = xyz_translated.unsqueeze(-1)
    ego_xyz_transformed = torch.matmul(anchor_rotation_inv, xyz_translated_col)

    return ego_xyz_transformed.squeeze(-1)


dataroot = "/proj/common-datasets/nuScenes/Full-dataset/v1.0"
version = "v1.0-test"

OBS_LEN = 4  # 4 frames at 2Hz = 2s → will interpolate to 16 frames at 10Hz
FUT_LEN = 13  # 13 frames at 2Hz = 6.4s → will interpolate to 64 frames at 10Hz (6.4s)
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
timestamp = "results/" + f"{timestamp}-nuscenes_{version}"
os.makedirs(timestamp, exist_ok=True)

# Load the dataset
nusc = NuScenes(version=version, dataroot=f"{dataroot}")
scenes = nusc.scene

global_ade1s = []
global_ade2s = []
global_ade3s = []
# global_ade6s = []
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
        cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])

        img = io.read_image(os.path.join(nusc.dataroot, cam_front_data["filename"]))
        front_camera_images.append(img.float() / 255.0)

        pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
        ego_poses.append(pose)

        if curr_sample_token == last_sample_token:
            break
        curr_sample_token = sample["next"]

    front_camera_images = torch.stack([img for img in front_camera_images], dim=0)
    scene_length = len(front_camera_images)
    print(f"Scene {name} has {scene_length} frames")
    if scene_length < TTL_LEN:
        print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
        continue

    ego_xyz = (
        torch.tensor(
            np.array([ego_poses[t]["translation"][:3] for t in range(scene_length)]),
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)
    )
    ego_rot = (
        torch.tensor(
            np.array([rot_mat(ego_poses[t]["rotation"]) for t in range(scene_length)]),
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)
    )

    prev_intent = None
    cam_images_sequence = []
    ade1s_list = []
    ade2s_list = []
    ade3s_list = []
    # ade6s_list = []
    for i in range(scene_length - TTL_LEN):
        obs_images = front_camera_images[i : i + OBS_LEN].to(device)
        obs_world_rot = ego_rot[:, :, i : i + OBS_LEN].to(device)
        fut_world_xyz_2hz = ego_xyz[:, :, i + OBS_LEN : i + TTL_LEN].to(device)
        obs_world_xyz_2hz = ego_xyz[:, :, i : i + OBS_LEN].to(device)

        # Anchor pose (last observation frame) in world coordinates
        anchor_T = ego_xyz[:, :, i + OBS_LEN - 1, :].unsqueeze(2).to(device)
        anchor_R = ego_rot[:, :, i + OBS_LEN - 1, :, :].unsqueeze(2).to(device)

        # Transform history and future to ego frame using LAST observation frame as anchor
        obs_ego_xyz_2Hz = world_to_ego_frame(obs_world_xyz_2hz, anchor_T, anchor_R)
        fut_ego_xyz_2hz = world_to_ego_frame(fut_world_xyz_2hz, anchor_T, anchor_R)

        # Generate interpolated future trajectory at 10Hz
        gt_points_2hz = fut_ego_xyz_2hz[0, 0].cpu().numpy()
        t_gt_src = np.arange(14) * 0.5
        gt_points_full = np.vstack(([0.0, 0.0, 0.0], gt_points_2hz))
        t_target = np.arange(1, 65) * 0.1  # Target evaluation grid: 10Hz up to 6.4s (64 steps)
        interpolator = CubicSpline(t_gt_src, gt_points_full, axis=0)
        gt_xyz_10hz = interpolator(t_target)
        fut_ego_xyz = torch.tensor(gt_xyz_10hz, device=device).reshape(1, 1, 64, 3)

        # generate interpolated history trajectory at 10Hz
        obs_points_2hz = obs_ego_xyz_2Hz[0, 0].cpu().numpy()
        t_hist_src = np.array([-1.5, -1.0, -0.5, 0.0])
        t_hist_tgt = np.linspace(-1.5, 0.0, 16)
        interpolator = CubicSpline(t_hist_src, obs_points_2hz, axis=0)
        obs_xyz_10hz = interpolator(t_hist_tgt)
        obs_ego_xyz = torch.tensor(obs_xyz_10hz, device=device).reshape(1, 1, 16, 3)

        # compute relative rotations to the anchor ego frame: R_rel_t = R_anchor^T @ R_t
        anchor_R_exp = anchor_R.expand(-1, -1, OBS_LEN, -1, -1)
        obs_rel_rot_2hz = torch.matmul(anchor_R_exp.transpose(-1, -2), obs_world_rot)
        obs_rot_2hz_np = obs_rel_rot_2hz[0, 0].cpu().numpy()

        # generate interpolated history rotations at 10Hz (anchor-relative) using Slerp
        rotations = R.from_matrix(obs_rot_2hz_np)
        slerp = Slerp(t_hist_src, rotations)
        obs_rot_10hz = slerp(t_hist_tgt).as_matrix()
        obs_ego_rot = torch.tensor(obs_rot_10hz, device=device).reshape(1, 1, 16, 3, 3)

        # Interpolate images to 10Hz (16 frames)
        obs_images_10hz = (
            torch.nn.functional.interpolate(
                obs_images.transpose(0, 1).unsqueeze(0),
                size=(16, obs_images.shape[2], obs_images.shape[3]),
                mode="nearest",
            ).squeeze(0).transpose(0, 1)
        ) 

        messages = helper.create_message(obs_images_10hz)
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
            "ego_history_rot": obs_ego_rot,
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
            print("Chain-of-Causation (per trajectory):\n", extra["cot"])
            print("Predicted trajectory shape:", pred_xyz.shape)
            print("Ground truth future shape:", fut_ego_xyz.shape)
            print(
                f"Predicted temporal dimension: {pred_T}, Future temporal dimension: {fut_ego_xyz.shape[2]}"
            )
            print("\nDEBUG: Full prediction trajectory (first 5 waypoints at 10Hz):")
            print("pred_xyz[0, 0, 0, :5, :2]:\n", pred_xyz[0, 0, 0, :5, :2].cpu().numpy())
            print("\nDEBUG: Consecutive distances in prediction (first 5 steps):")
            for t in range(min(5, pred_T - 1)):
                dist = np.linalg.norm(
                    pred_xyz[0, 0, 0, t + 1, :2].cpu().numpy()
                    - pred_xyz[0, 0, 0, t, :2].cpu().numpy()
                )
                print(f"  Step {t} to {t + 1} (0.1s): {dist:.4f}m")
            print("\nDEBUG: Ground truth trajectory (first 5 timesteps at 2Hz):")
            print("fut_ego_xyz[:5, :2]:\n", fut_ego_xyz[0, 0, :5, :2].cpu().numpy())
            print(f"\nGT interpolated from {FUT_LEN} points (2Hz) to 64 points (10Hz)")
            print("Comparing both at 10Hz native model frequency")
            print(f"pred_xyz shape: {pred_xyz.shape}, fut_ego_xyz shape: {fut_ego_xyz.shape}")
            print("\nDEBUG: Velocity comparison at 10Hz (0.1s intervals):")
            print("Prediction trajectory (first 5 waypoints at 10Hz):")
            print("pred_xyz[0, 0, 0, :5, :2]:\n", pred_xyz[0, 0, 0, :5, :2].cpu().numpy())
            print("  Prediction velocities (at 10Hz = 0.1s apart):")
            for idx in range(min(5, 11)):
                vel = np.linalg.norm(
                    pred_xyz[0, 0, 0, idx + 1, :2].cpu().numpy()
                    - pred_xyz[0, 0, 0, idx, :2].cpu().numpy()
                )
                print(
                    f"    Waypoint {idx} to {idx + 1} (0.1s): {vel:.4f}m (speed: {vel / 0.1:.2f} m/s)"
                )

            print("\nGround truth trajectory (interpolated to 10Hz):")
            print("fut_ego_xyz[0, 0, :5, :2]:\n", fut_ego_xyz[0, 0, :5, :2].cpu().numpy())
            print("  Ground truth velocities (at 10Hz = 0.1s apart):")
            for idx in range(min(5, 11)):
                vel = np.linalg.norm(
                    fut_ego_xyz[0, 0, idx + 1, :2].cpu().numpy()
                    - fut_ego_xyz[0, 0, idx, :2].cpu().numpy()
                )
                print(
                    f"    Waypoint {idx} to {idx + 1} (0.1s): {vel:.4f}m (speed: {vel / 0.1:.2f} m/s)"
                )

        # Prepare trajectories for ADE: use single sample (clean evaluation)
        gt_xy = fut_ego_xyz[0, 0, :, :2].cpu().numpy()  # [64, 2]
        pred_xy = pred_xyz[0, 0, 0, :, :2].detach().cpu().numpy()  # first sample

        if i == 0:
            print(f"\nGT XY shape: {gt_xy.shape}, Pred XY shape: {pred_xy.shape}")
            print(f"GT XY range - min: {gt_xy.min():.4f}, max: {gt_xy.max():.4f}")
            print(f"Pred XY range - min: {pred_xy.min():.4f}, max: {pred_xy.max():.4f}")

        # Align lengths per horizon and compute ADEs
        n_full = min(gt_xy.shape[0], pred_xy.shape[0])
        ade1 = compute_ade(gt_xy[:min(10, n_full)], pred_xy[:min(10, n_full)])
        ade2 = compute_ade(gt_xy[:min(20, n_full)], pred_xy[:min(20, n_full)])
        ade3 = compute_ade(gt_xy[:min(30, n_full)], pred_xy[:min(30, n_full)])
        # ade6 = compute_ade(gt_xy[:min(60, n_full)], pred_xy[:min(60, n_full)])
        if i == 0:
            # Lightweight diagnostic: check if flipping X or Y reduces ADE at full horizon
            pred_xy_flipx = pred_xy.copy(); pred_xy_flipx[:,0] *= -1
            pred_xy_flipy = pred_xy.copy(); pred_xy_flipy[:,1] *= -1
            ade6_flipx = compute_ade(gt_xy[:min(60, n_full)], pred_xy_flipx[:min(60, n_full)])
            ade6_flipy = compute_ade(gt_xy[:min(60, n_full)], pred_xy_flipy[:min(60, n_full)])
            print(f"Diag ADE@6.4s flipX: {ade6_flipx:.4f}m, flipY: {ade6_flipy:.4f}m")
        # print(f"ADE@1s: {ade1:.4f}m, ADE@3s: {ade3:.4f}m, ADE@6.4s: {ade6:.4f}m")
        print(f"ADE@1s: {ade1:.4f}m, ADE@2s: {ade2:.4f}m, ADE@3s: {ade3:.4f}m")

        ade1s_list.append(ade1)
        ade2s_list.append(ade2)
        ade3s_list.append(ade3)
        # ade6s_list.append(ade6)

    mean_ade1s = np.mean(ade1s_list)
    mean_ade2s = np.mean(ade2s_list)
    mean_ade3s = np.mean(ade3s_list)
    # mean_ade6s = np.mean(ade6s_list)
    aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])
    inference_time = time.time() - start_scene_time

    global_ade1s.append(mean_ade1s)
    global_ade2s.append(mean_ade2s)
    global_ade3s.append(mean_ade3s)
    # global_ade6s.append(mean_ade6s)
    global_aveg_ades.append(aveg_ade)
    global_inference_times.append(inference_time)

    result = {
        "name": name,
        "token": token,
        "ade1s": float(mean_ade1s),
        "ade2s": float(mean_ade2s),
        "ade3s": float(mean_ade3s),
        # "ade6s": float(mean_ade6s),
        "avgade": float(aveg_ade),
        "inference_time": float(inference_time),
    }

    with open(f"{timestamp}/ade_results.jsonl", "a") as f:
        f.write(json.dumps(result))
        f.write("\n")

    print(
        f"Scene {name} done in {time.time() - start_scene_time} seconds. ADE1s: {mean_ade1s}, ADE2s: {mean_ade2s}, ADE3s: {mean_ade3s}, AvgADE: {aveg_ade}"
    )


global_result = {
    "name": "OVERALL",
    "overall_ade1s": np.mean(global_ade1s),
    "overall_ade2s": np.mean(global_ade2s),
    "overall_ade3s": np.mean(global_ade3s),
    # "overall_ade6s": np.mean(global_ade6s),
    "overall_avgade": np.mean(global_aveg_ades),
    "overall_inference_time_per_scene": np.mean(global_inference_times),
}
with open(f"{timestamp}/ade_results.jsonl", "a") as f:
    f.write(json.dumps(global_result))
    f.write("\n")

print(
    f"Overall ADE1s: {np.mean(global_ade1s)}, ADE2s: {np.mean(global_ade2s)}, ADE3s: {np.mean(global_ade3s)}, AvgADE: {np.mean(global_aveg_ades)}"
)
print(f"Overall Inference Time per Scene: {np.mean(global_inference_times)} seconds")

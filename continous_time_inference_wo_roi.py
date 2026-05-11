import rppg
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.stats import pearsonr

models = [
    'FacePhys.rlap',
    # 'ME-chunk.rlap',
    # 'ME-flow.rlap',
    # 'ME-chunk.pure',
    # 'ME-flow.pure',
    # 'PhysMamba.pure',
    # 'PhysMamba.rlap',
    # 'RhythmMamba.rlap',
    # 'RhythmMamba.pure',
    'PhysFormer.pure',
    'PhysFormer.rlap',
    # 'TSCAN.rlap',
    # 'TSCAN.pure',
    # 'PhysNet.rlap',
    # 'PhysNet.pure',
    'EfficientPhys.pure',
    'EfficientPhys.rlap'
]

video_name = ""
video_path = f"video/{video_name}.avi"
print(f"Video_name: {video_name}")

# --- Split video into 30s sub-videos ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
segment_duration = 15  # seconds
frames_per_segment = int(segment_duration * fps)

print("Starting video frames splitting")
segments = []
start_frame = 0
while start_frame < total_frames:
    end_frame = min(start_frame + frames_per_segment, total_frames)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if frames:
        segments.append(frames)
    start_frame = end_frame
cap.release()

# --- For each segment, estimate HR ---
for model_name in models:
    print(f"\nRunning inference for model: {model_name}")
    model = rppg.Model(model_name)

    print("Starting to estimate HR for each video frame")
    hr_array = []
    time_array = []
    segment_times = []  # To store time for each segment
    start_total = time.time()  # Start total timer

    for idx, frames in enumerate(segments):
        start_segment = time.time()  # Start timer for this segment
        temp_video = f"temp_segment_{idx}.mp4"
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        
        results = model.process_video(temp_video)
        print(f"Estimated Heart Rate: {results['hr']} BPM\n")
        
        hr_array.append(results['hr'])
        time_array.append(idx * segment_duration)

        os.remove(temp_video)
        
        end_segment = time.time()
        segment_times.append(end_segment - start_segment)

    end_total = time.time()
    total_time = end_total - start_total

    # Print timing results
    print(f"Total HR estimation time: {total_time:.2f} seconds")
    print("Time taken for each segment:")
    for i, t in enumerate(segment_times):
        print(f"  Segment {i}: {t:.2f} seconds")

    # --- Plot HR over time ---
    # plt.figure(figsize=(10, 5))
    # plt.plot(time_array, hr_array, marker='o')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Estimated HR (BPM)')
    # plt.title(f'Estimated HR over time for {video_name}')
    # plt.grid(True)
    # plt.show()

    # --- Plot estimated HR and ground truth HR together ---
    gt_file = 'ground_truth/sp02wave_L9v2_16-04-13_16-06-12.txt'
    with open(gt_file, 'r') as f:
        _ = f.readline()  # skip PPG line
        gt_hr_line = f.readline()
        gt_ts_line = f.readline()

    gt_hr = np.array([float(x) for x in gt_hr_line.strip().split()])
    gt_ts = np.array([float(x) for x in gt_ts_line.strip().split()])

    # Interpolate ground truth HR to match the time_array points
    if len(gt_ts) > 1:
        gt_hr_interp = np.interp(time_array, gt_ts, gt_hr)
    else:
        gt_hr_interp = np.full_like(time_array, gt_hr[0])

    # --- Calculate mean error between estimated HR and ground truth HR ---
    mean_error_array = np.abs(np.array(hr_array) - gt_hr_interp)
    avg_mean_error = np.mean(mean_error_array)
    std_mean_error = np.std(mean_error_array)

    print("Mean error array:", mean_error_array)
    print(f"AVG MEAN ERROR: {avg_mean_error:.2f} ± {std_mean_error:.2f} BPM")

    # --- Calculate Pearson correlation between estimated HR and ground truth HR ---
    pearson_corr, pearson_p = pearsonr(hr_array, gt_hr_interp)
    print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3g})")

    # # --- Plot estimated HR and ground truth HR together ---
    # plt.figure(figsize=(10, 5))
    # plt.plot(time_array, hr_array, marker='o', label='Estimated HR (model)')
    # plt.plot(time_array, gt_hr_interp, marker='x', linestyle='--', label='Ground Truth HR')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Heart Rate (BPM)')
    # plt.title(f'Estimated HR vs. Ground Truth HR for {video_name}')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # --- Save results to text file ---
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_txt_path = os.path.join(results_dir, f"{model_name}_{video_name}_result.txt")

    with open(results_txt_path, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Total HR estimation time: {total_time:.2f} seconds\n")
        f.write("Time taken for each segment:\n")
        for i, t in enumerate(segment_times):
            f.write(f"  Segment {i}: {t:.2f} seconds\n")
        f.write("\n")
        f.write(f"Mean error array: {mean_error_array.tolist()}\n")
        f.write(f"AVG MEAN ERROR: {avg_mean_error:.2f} ± {std_mean_error:.2f} BPM\n")
        f.write(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3g})\n")

    # --- Save plot to file ---
    plot_path = os.path.join(results_dir, f"{model_name}_{video_name}_result.png")
    plt.figure(figsize=(10, 5))
    plt.plot(time_array, hr_array, marker='o', label='Estimated HR (model)')
    plt.plot(time_array, gt_hr_interp, marker='x', linestyle='--', label='Ground Truth HR')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title(f'Estimated HR vs. Ground Truth HR for {video_name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    # plt.show()

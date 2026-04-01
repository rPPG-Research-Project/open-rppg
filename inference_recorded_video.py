import rppg

# List of model names and their corresponding loader functions
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
    'PhysNet.rlap',
    'PhysNet.pure',
    'EfficientPhys.pure',
    'EfficientPhys.rlap'
]

video_name = ""
video_path = f"video/{video_name}.avi"
print(f"Video_name: {video_name}")

with open(f"results/{video_name}_results.log", "w") as log_file:
    for model_name in models:
        model = rppg.Model(model_name)
        results = model.process_video(video_path)
        output = f"Estimated Heart Rate: {results['hr']} BPM\n"
        
        print("---------------------------------------------------------------")
        print(f"Model: {model_name}")

        log_file.write("---------------------------------------------------------------\n")
        log_file.write(f"Model: {model_name}\n")
        
        print(output)
        log_file.write(output + "\n")
    
log_file.close()
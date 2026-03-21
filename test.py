import rppg

# Initialize the model
print("-------- Initializing the model --------")
model = rppg.Model()

# Process a video file
print("-------- Processing video --------")

results = model.process_video("video/pilot.mp4")

# Display the heart rate
print("-------- Results --------")

print(f"Estimated Heart Rate: {results['hr']} BPM")
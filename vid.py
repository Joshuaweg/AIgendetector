import cv2
import os

# Define the path where your frames are stored
image_folder = r'C:\Users\joshu\OneDrive\Desktop\AIGenClassifier\plots'

# Set the output video file name and format
output_video = 'output_video_real.mp4'

# Get all image file names in the folder (assuming PNG files)
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  # Optional: Sort to ensure correct frame order

# Read the first image to get dimensions (assuming all images have the same size)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or other codecs
fps = 30  # Set frames per second
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Loop through all images and write them into the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release the video writer
video.release()

print(f"Video saved as {output_video}")
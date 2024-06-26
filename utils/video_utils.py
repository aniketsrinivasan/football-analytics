# Reading in and saving the video.

# Importing CV2 to use computer vision for the task:
import cv2
import os


def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k "
             "-vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input=avi_file_path,
                                                                                      output=output_name))
    return True


# Function to "read" a video given a video path, returning a
#    list of frames:
def read_video(video_path) -> list:
    # Creating a VideoCapture object:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        # Using the .read() method on VideoCapture object:
        #   (returns tuple[bool, Union])
        ret, frame = cap.read()
        if not ret:
            break
        # Appending current frame read:
        frames.append(frame)
    return frames


def save_video(frames, save_path):
    # Storing the video format in a FourCC type:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # Frame output:
    #   arguments are (save_path, format, FPS, width, height)
    out = cv2.VideoWriter(save_path, fourcc, 24,
                          (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        # Writes the frame to the VideoWriter out:
        out.write(frame)
    out.release()

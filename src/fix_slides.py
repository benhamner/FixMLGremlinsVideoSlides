import csv
import cv2
import cv2.cv as cv
from matplotlib import pyplot as plt
import numpy as np

# I/O paths
video_path  =  "../data/MachineLearningGremlins.mp4"
output_path = "../data/out.avi"
slide_paths = ["../data/Slides/Slide%d.PNG" % i for i in range(1,148)]
log_file    = open("../data/log.txt", "w")
log_writer  = csv.writer(log_file, lineterminator="\n")
log_writer.writerow(["Frame", "Slide", "SlideCorrelation"])

# Parameters
video_width        = 1280
video_height       = 720
video_slide_left   = 244
video_slide_lower  = 66
video_slide_upper  = 654
video_slide_right  = 1279
video_slide_width  = video_slide_right - video_slide_left  + 1
video_slide_height = video_slide_upper - video_slide_lower + 1
video_slide_size   = (video_slide_width, video_slide_height)
video_slide_length = video_slide_height * video_slide_width

# Read in slides
slides_original = [cv2.imread(path) for path in slide_paths]
slides_resized  = [cv2.resize(slide[2:-1,3:,:], video_slide_size) for slide in slides_original]
slides_gray     = [np.reshape(cv2.cvtColor(slide, cv2.COLOR_BGR2GRAY), video_slide_length) for slide in slides_resized]

# Set up video I/O
cap    = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv.CV_CAP_PROP_FPS)
print("Frame Rate: \t%f" % cap.get(cv.CV_CAP_PROP_FPS))
print("Frame Count:\t%d" % cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
xvid = cv.CV_FOURCC('X','V','I','D')
video_writer = cv2.VideoWriter(output_path, xvid, frame_rate, (video_width, video_height), True)

# Initialization Parameters
last_cropped_gray = None
i = 0
transition_corr      = 0.0
last_transition_corr = 0.0
max_corr = 0.0
min_slide = 0
slide_width = 10
frame = 1
last_slide_shown = -2
corr_match_cutoff = 0.45

while True:
    i += 1
    ret,frame = cap.read()
    if frame is None:
        break
    cropped = frame[video_slide_lower:video_slide_upper+1, video_slide_left:video_slide_right+1, :]
    cropped_gray = np.reshape(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), video_slide_length)
    if last_cropped_gray is not None:
        transition_corr = np.corrcoef(last_cropped_gray, cropped_gray)[0,1]
    last_cropped_gray = cropped_gray
    if (transition_corr < 0.995) != (last_transition_corr < 0.995): # only change slide on transitions from/to stable state
        slide_correlations = np.array([np.abs(np.corrcoef(cropped_gray, slide)[0,1]) for slide in slides_gray[min_slide:min_slide+slide_width]])
        slide = slide_correlations.argmax() + min_slide
        max_corr = slide_correlations.max()
        slide_shown = slide if max_corr > corr_match_cutoff else -1
        if slide_shown != last_slide_shown:
            log_writer.writerow([i, slide_shown, max_corr])
            log_file.flush()
            cv2.imwrite("../data/Frames/%d-slide-%f.png" % (i, max_corr), np.concatenate((cropped, slides_resized[slide]), 0))
            print(i, max_corr, slide, transition_corr, last_transition_corr)
        last_slide_shown = slide_shown
    if max_corr > corr_match_cutoff:
        min_slide = max(slide - int(slide_width/2), min_slide)
        frame[video_slide_lower:video_slide_upper+1, video_slide_left:video_slide_right+1, :] = slides_resized[slide]
    last_transition_corr = transition_corr
    video_writer.write(frame)
video_writer.release()
log_file.close()

# .\ffmpeg.exe -i MachineLearningGremlins.mp4 -acodec copy -vn audio.mp4
# .\ffmpeg.exe -i out.avi -i audio.mp4 -vcodec copy -acodec copy -map 0:0 -map 1:0 MachineLearningGremlinsCorrected2.mp4
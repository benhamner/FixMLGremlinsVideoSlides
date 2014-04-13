import cv2
import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt

video_path =  "../data/MachineLearningGremlins.mp4"
output_path = "../data/out.avi"
slide_paths = ["../data/Slides/Slide%d.PNG" % i for i in range(1,148)]

video_slide_left   = 243
video_slide_lower  = 65
video_slide_upper  = 655
video_slide_right  = 1279
video_slide_width  = video_slide_right - video_slide_left  + 1
video_slide_height = video_slide_upper - video_slide_lower + 1
video_slide_size   = (video_slide_width, video_slide_height)
video_slide_length = video_slide_height * video_slide_width

# read in slides
slides_original = [cv2.imread(path) for path in slide_paths]
slides_resized  = [cv2.resize(slide, video_slide_size) for slide in slides_original]
slides_gray     = [np.reshape(cv2.cvtColor(slide, cv2.COLOR_BGR2GRAY), video_slide_length) for slide in slides_resized]

cap    = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv.CV_CAP_PROP_FPS)
fourcc     = cap.get(cv.CV_CAP_PROP_FOURCC)
print("Frame Rate: \t%f" % cap.get(cv.CV_CAP_PROP_FPS))
print("Frame Count:\t%d" % cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
print("FOURCC:     \t%d" % cap.get(cv.CV_CAP_PROP_FOURCC))
xvid = cv.CV_FOURCC('X','V','I','D')

writer = cv2.VideoWriter(output_path, xvid, frame_rate, (1280,720), True)
last_cropped_gray = None

i = 0
ret = True
transition_corr      = 0.0
last_transition_corr = 0.0
max_corr = 0.0
min_slide = 0
slide_width = 10
frame = 1

while frame is not None:
    i += 1
    ret,frame = cap.read()
    cropped = frame[video_slide_lower:video_slide_upper+1, video_slide_left:video_slide_right+1, :]
    cropped_gray = np.reshape(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), video_slide_length)
    if last_cropped_gray is not None:
        transition_corr = np.corrcoef(last_cropped_gray, cropped_gray)[0,1]
    last_cropped_gray = cropped_gray
    if (transition_corr < 0.99) != (last_transition_corr < 0.99): # only change slide on transitions from/to stable state
        slide_correlations = np.array([np.corrcoef(cropped_gray, slide)[0,1] for slide in slides_gray[min_slide:min_slide+slide_width]])
        slide = slide_correlations.argmax() + min_slide
        max_corr = slide_correlations.max()
        if max_corr>0.25:
            cv2.imwrite("../data/Frames/%d-slide-%f.png" % (i, max_corr), slides_resized[slide])
            cv2.imwrite("../data/Frames/%d-cropped.png" % i, cropped)
            print(i, max_corr, slide, transition_corr, last_transition_corr)
    if max_corr > 0.5:
        min_slide = max(slide - int(slide_width/2), 0)
        frame[video_slide_lower:video_slide_upper+1, video_slide_left:video_slide_right+1, :] = slides_resized[slide]
    last_transition_corr = transition_corr
    writer.write(frame)
writer.release()

# .\ffmpeg.exe -i MachineLearningGremlins.mp4 -acodec copy -vn audio.mp4
# .\ffmpeg.exe -i out.avi -i audio.mp4 -vcodec copy -acodec copy -map 0:0 -map 1:0 final.mp4
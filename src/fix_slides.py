import cv2
import numpy as np
from matplotlib import pyplot as plt

video_path =  "../data/MachineLearningGremlins.mp4"
output_path = "../data/out.mp4"
slide_paths = ["../data/Slides/Slide%d.PNG" % i for i in range(1,148)]

video_slide_left   = 244
video_slide_lower  = 67
video_slide_upper  = 653
video_slide_right  = 1278
video_slide_width  = video_slide_right - video_slide_left  + 1
video_slide_height = video_slide_upper - video_slide_lower + 1
video_slide_size   = (video_slide_width, video_slide_height)
video_slide_length = video_slide_height * video_slide_width

# read in slides
slides_original = [cv2.imread(path) for path in slide_paths]
slides_resized  = [cv2.resize(slide, video_slide_size) for slide in slides_original]
slides_gray     = [np.reshape(cv2.cvtColor(slide, cv2.COLOR_BGR2GRAY), video_slide_length) for slide in slides_resized]
print(np.shape(slides_gray[0]))

cap    = cv2.VideoCapture(video_path)
writer = cv2.VideoWriter(output_path, ['M','P','4','V'], 29, (1280,720), True)
last_cropped_gray = None

for i in range(2000):
    ret,frame = cap.read()
    cropped = frame[video_slide_lower:video_slide_upper+1, video_slide_left:video_slide_right+1, :]
    cropped_gray = np.reshape(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), video_slide_length)
    if last_cropped_gray is not None:
        last_coef = np.corrcoef(last_cropped_gray, cropped_gray)[0,1]
    else:
        last_coef = 0.0
    last_cropped_gray = cropped_gray
    if last_coef < 0.99:
        slide_correlations = np.array([np.corrcoef(cropped_gray, slide)[0,1] for slide in slides_gray])
        print(i, slide_correlations.max(), slide_correlations.argmax())
        slide = slide_correlations.argmax()
        max_corr = slide_correlations.max()
        plt.subplot(121),plt.imshow(cropped)
        plt.subplot(122),plt.imshow(slides_resized[slide_correlations.argmax()])
        plt.show(block=False)
        cv2.waitKey(0)
    if max_corr > 0.5:
        frame[video_slide_lower:video_slide_upper+1, video_slide_left:video_slide_right+1, :] = slides_resized[slide]
    writer.write(frame)
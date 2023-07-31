import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

model = load_model('./assets/attn-unet_carlo.h5')

#set color to image
def give_color_to_seg_img(seg, n_classes=13):
    seg_img = np.zeros( (seg.shape[0],seg.shape[1], 3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)

def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = img / 255.
    return img

video_capture = cv2.VideoCapture('E:\\output_video.mp4')

if not video_capture.isOpened():
    print("Failed to open video source")
    exit()

# Read the first frame
ret, frame = video_capture.read()

while ret:
    # Preprocess the frame
    image = preprocess(frame)
    
    # Perform segmentation using the model
    preds = model.predict(np.expand_dims(image, axis=0))
    pred_mask = np.argmax(preds, axis=3)

    # Convert the segmentation mask to the appropriate data type
    #pred_mask = pred_mask.astype(np.uint8) * 255

    # Display the segmented frame
    pred_mask_color = give_color_to_seg_img(pred_mask[0])
    cv2.imshow('Segmented Image', image * 0.5 + pred_mask_color * 0.5)

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = video_capture.read()

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
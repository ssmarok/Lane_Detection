#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
from arg_parser import args

# Setup video playback
video_dir = args.video_dir
video_rot = args.video_rot if (args.video_rot % 90 == 0) else 270
video_str = str('video_data_' + str(args.video_num))
cap = cv2.VideoCapture(video_dir + video_str + '.mov')
# Get frame size (out of typical order because image gets rotated)
frame_height = int(cap.get(3))
frame_width = int(cap.get(4))
# Define the codec and create VideoWriter object for output
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'MPEG')
# Last parameter is 0 for grayscale video output
out_dir = 'out/'
out = cv2.VideoWriter(out_dir + "processed_"+ video_str + ".mov", fourcc, 30, (frame_width, frame_height), 0)

# Apply Gaussian Blurring
def apply_smoothing(image, kernel_size=3):
    """
    kernel_size must be postivie and odd.
	Larger kernel_size - - > More processing time and more blur
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Apply Canny Edge Detection
def detect_edges(image, low_threshold=50, high_threshold=150):
    """ If pixel gradient > high_threshold:
            pixel accepted as edge
        If pixel gradient < low_threshold:
            pixel rejected as edge
        If pixel gradient > low and pixel gradient < high:
            ccepted as edge, only if connected to pixel above high_threshold
      Recommended ratio upper:lower is 2:1 or 3:1
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.05, rows*0.95]
    top_left     = [cols*0.3, rows*0.55]
    bottom_right = [cols*0.95, rows*0.95]
    top_right    = [cols*0.7, rows*0.55]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

# Process a single image frame
def process_frame(frame):
    processed = frame
    # Rotate Image
    processed = imutils.rotate_bound(frame, video_rot) 
    # Convert image to grayscale
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blurring with specififed kernel size (k x k)
    processed = apply_smoothing(processed, kernel_size = 3)
    # Select Region of Interest
    processed = select_region(processed)
    # Apply Canny Edge Detection
    processed = detect_edges(processed)  # Apply canny

    # Return processed image
    return processed

# Main function
def main():
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video file")
    # Continuously process video frames
    while(cap.isOpened()):
        # Capture video frame
        ret, frame = cap.read()
        if ret == False:
            print("Video Complete")
            exit()
        
        # Apply image processing
        processed_frame = process_frame(frame)
        # Write the processed frame
        out.write(processed_frame)
        # Display processed image
        cv2.imshow('frame', processed_frame)
        # Delay for key press and frame rate
        if cv2.waitKey(args.video_delay) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # This script has been tested with python3 and OpenCV version 3.2.0
    print("-------------------------------------------")
    print("            Stay in Yo' Lane               ")
    print("           -------------------             ")
    print("             Lane Detection                ")
    print("-------------------------------------------")
    # Display arguments and help
    print("Arguments: ")
    print("\tVideo Number: ", args.video_num)
    print("\tVideo Frame Delay: ", args.video_delay,"ms")
    print("Press 'q' to quit")

    # Run main
    main()
    

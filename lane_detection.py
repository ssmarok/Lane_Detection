#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
from arg_parser import args

# Debug options
DEBUG_IMAGES = args.debug
# Setup video playback
video_rot = args.video_rot if (args.video_rot % 90 == 0) else 270
video_str = str('video_data_' + str(args.video_num))
cap = cv2.VideoCapture(args.video_in_dir + video_str + '.mov')
# Get frame size (out of typical order because image gets rotated)
frame_height = int(cap.get(3))
frame_width = int(cap.get(4))
# Define a codec and create VideoWriter object for output
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc(*'MPEG')
# Output directory
# Last parameter is 0 for grayscale video output
out = cv2.VideoWriter(str(args.video_out_dir) + 'processed_'+ video_str + '.mov', fourcc, 30, (frame_width, frame_height), 1)


# Add two images/masks
def add_image_mask(image, image_mask):
    """
    Add two images. (e.g. original + lines_image)
    image_mask is given more weight (e.g. lines_image)
    """
    try:
        summed = cv2.addWeighted(image, 0.8, image_mask, 1, 0)
        return summed
    except:
        print("Unable to apply mask, Check the shape of mask and image")

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

# Filter region given vertices
def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image.
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2]) # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


# Select ROI
def select_region(image):
    """
    Create a trapezoidal mask on road as region of interest.
    """
    # Define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.05, rows*0.95]
    top_left     = [cols*0.3, rows*0.55]
    bottom_right = [cols*0.95, rows*0.95]
    top_right    = [cols*0.7, rows*0.55]
    # Vertices are an array of polygons (i.e array of arrays) and the data type must be integer.
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

# Apply Probabilistic Hough Lines algorithm
def get_hough_lines(image):
    """
    'image' should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    rho: Distance resolution of the accumulator in pixels.
    theta: Angle resolution of the accumulator in radians.
    threshold: Accumulator threshold parameter. Only those lines are returned
                that get enough votes (> threshold).
    minLineLength: Minimum line length. Line segments shorter than that are rejected.
    maxLineGap: Maximum allowed gap between points on the same line to link them.
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20,
                           minLineLength=20, maxLineGap=300)

# Draw Hough Lines
def draw_hough_lines(image, lines):
    """
    Draw hough lines on image frame.
    """
    # Create new empty image
    line_image = np.zeros_like(image)
    if len(lines) > 0:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return line_image

# Process Hough Lines
def process_lines(lines, min_slope=0.4, max_slope =0.7):
    """ 
    Apply post-processing to hough lines returned.
    """
    processed_lines = []
    if len(lines) > 0:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Skip for vertical line
                if(x2 == x1):
                    continue
                # Check slope of lines
                calc_slope = abs((y2 - y1) / (x2 - x1))
                if calc_slope > min_slope and calc_slope < max_slope:
                    processed_lines.append(line)

    return processed_lines

# Process a single image frame
def process_frame(frame):
    # Rotate Image
    rotated_img = imutils.rotate_bound(frame, video_rot) 
    # Convert image to grayscale
    processed = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blurring with specififed kernel size (k x k)
    processed = apply_smoothing(processed, kernel_size = 3)
    # Select Region of Interest
    processed = select_region(processed)
    # Apply Canny Edge Detection
    processed = detect_edges(processed)
    if DEBUG_IMAGES:
        cv2.imshow('canny', processed)
    # Get Hough Lines
    raw_lines = get_hough_lines(processed)
    # Get processed Hough Lines
    processed_lines = process_lines(raw_lines, min_slope=.4, max_slope=0.8)
    # Draw Hough Lines on original image
    raw_line_image = draw_hough_lines(rotated_img, raw_lines) # For Debugging
    processed_line_image = draw_hough_lines(rotated_img, processed_lines)
    if DEBUG_IMAGES:
        cv2.imshow('raw', raw_line_image)
        cv2.imshow('processed', processed_line_image)
    # Show lines on original image
    summed_image = add_image_mask(rotated_img, processed_line_image)

    # Return processed image
    return summed_image

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
    

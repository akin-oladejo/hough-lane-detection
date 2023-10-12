import streamlit as st

import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import cv2
import math
import streamlit.components.v1 as components

DEFAULT_VIDEO = "highway.mp4"

# tab title and favicon
st.set_page_config(
    page_title="Lane Detection",
    page_icon="🚗"
)

# session state
st.session_state.params = {}
st.session_state.fps = 30


# define function to detect lanes
def plot_lanes(image):
    """
    image. image array

    Canny Edge Detection Parameters
    ====
    canny_low.
    canny_high.

    Hough Transform Parameters
    ====
    rho. distance resolution in pixels of the Hough grid
    theta. angular resolution in radians of the Hough grid
    threshold. minimum number of votes (intersections in Hough grid cell)
    min_line_length. minimum number of pixels making up a line
    max_line_gap. maximum gap in pixels between connectable line segments
    """

    if st.session_state.get('hide_canny') and st.session_state.hide_canny:
        cv2.destroyAllWindows() # close all windows
        return image

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # convert image to gray

    # Define a kernel size and apply Gaussian smoothing to remove noise
    blur_gray = cv2.GaussianBlur(gray,(st.session_state.params['kernel_size'], st.session_state.params['kernel_size']),0)

    # Define our parameters for Canny and apply
    edges = cv2.Canny(blur_gray, st.session_state.params['canny_low'], st.session_state.params['canny_high'])

    # return edges
    if st.session_state.get('hide_hough') and st.session_state.hide_hough:
        cv2.destroyAllWindows() # close all windows
        return edges


    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Make a blank the same size as our image to draw on
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # # Run Hough on edge detected image
    # # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, 
                            st.session_state.params['rho'], 
                            st.session_state.params['theta'], 
                            st.session_state.params['hough_threshold'],
                            np.array([]),
                            st.session_state.params['min_line_length'], 
                            st.session_state.params['max_line_gap'])

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    cv2.destroyAllWindows() # close all windows
    return lines_edges

def play_output(frames):
    st.subheader("Preview")
    annotated_frames = np.array([plot_lanes(img) for img in frames])

    # Create a figure
    fig = plt.figure()
    plt.axis('off')

    # Define a function to update the image in the figure
    def update_image(n):
        plt.imshow(annotated_frames[n])

    # Create an animation object
    anim = animation.FuncAnimation(fig, update_image, frames=15, interval=100)

    components.html(anim.to_jshtml(), height=800) # return html frame


# Side bar for adjusting parameters
with st.sidebar:
    st.subheader("Canny Edge Detection Parameters") 
    st.toggle('Hide Canny Edge Detection', key='hide_canny')

    if not st.session_state.hide_canny:
        st.session_state.params['canny_low'], st.session_state.params['canny_high'] = st.slider('threshold range', 0, 200, (80, 180), help="threshold limits for the hysteresis procedure")
        st.session_state.params['kernel_size'] = st.number_input('kernel size (odd number)', min_value=1, max_value=11, value=5, step=2, help='kernel size for supplementary gaussian smoothing')
        "---" # divider
        st.subheader("Hough Transform Parameters") 
        st.toggle('Hide Hough Transform', key='hide_hough')
        if not st.session_state.hide_hough:
            st.session_state.params['rho'] = st.slider('rho', 0, 100, 5, help='distance resolution in pixels of the Hough grid')
            st.session_state.params['theta'] = math.radians(min(1, st.slider('theta (will be converted to radians)', 0, 360, 60, step=1, help='angular resolution in pixels of the Hough grid')))
            st.session_state.params['hough_threshold'] = st.slider('threshold', value=1, help='minimum number of votes (intersections in Hough grid cell)')
            st.session_state.params['min_line_length'] = st.number_input('minimum line length', value=25, help='minimum number of pixels making up a line')
            st.session_state.params['max_line_gap'] = st.number_input('maximum line gap', value=3.0, help='maximum gap in pixels between connectable line segments')

# main section

# title and description
st.title('Traditional Lane Detection')
"""With [Canny Edge detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) + [Hough Transform](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)"""

# two tabs for default and uploaded videos
default, upload_tab = st.tabs(['Default video', 'Upload video'])

with default:
    frames = iio.imread(DEFAULT_VIDEO, plugin="pyav")[:15]
    play_output(frames)

        
with upload_tab:
    video = st.file_uploader("Select a video from your files", accept_multiple_files=False)
    if video is not None:
        uploaded_frames = iio.imread(video.getvalue(), plugin="pyav")[:15]
        play_output(uploaded_frames)

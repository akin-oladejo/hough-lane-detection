import streamlit as st

import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import cv2
import math
from array2gif import write_gif


# tab title and favicon
st.set_page_config(
    page_title="Lane Detection",
    page_icon="🚗"
)

# detection_parameters
# params = {
#     'canny_low': 24,
#     'canny_high': 113,
#     'kernel_size': 5,
#     'rho': 1,
#     'theta': np.pi/180,
#     'hough_threshold': 1,
#     'min_line_length': 5,
#     'max_line_gap': 1,
#     'hide_canny':False,
#     'hide_hough':False
# }

st.session_state.params = {}
st.session_state.video_ready = False


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

    if st.session_state.hide_canny:
        return image

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # convert image to gray

    # return gray

    # Define a kernel size and apply Gaussian smoothing to remove noise
    blur_gray = cv2.GaussianBlur(gray,(st.session_state.params['kernel_size'], st.session_state.params['kernel_size']),0)

    # return blur_gray

    # Define our parameters for Canny and apply
    edges = cv2.Canny(blur_gray, st.session_state.params['canny_low'], st.session_state.params['canny_high'])

    # return edges

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    if st.session_state.hide_hough:
        return masked_edges


    # return edges

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
    return lines_edges
    # plt.imshow(lines_edges)


@st.cache_data
def annotate_video(frame_arr, fps):
    """annotate lanes in video"""
    annotated_frames = np.array([plot_lanes(img) for img in frame_arr], dtype=np.int32)
     
    # # Create a figure
    # fig = plt.figure()

    # # Define a function to update the image in the figure
    # def update_image(n):
    #     plt.imshow(annotated_frames[n])

    # # Create an animation object
    # anim = animation.FuncAnimation(fig, update_image, frames=len(annotated_frames))

    # anim.save('output.mp4', writer='ffmpeg', fps=30)

    write_gif(annotated_frames, 'output.gif', fps=fps)

    # update state
    st.session_state.video_ready = True

# display preview
def preview(frame_arr, index:int=10):
    st.subheader("Preview")
    fig, ax = plt.subplots()
    ax.imshow(plot_lanes(frame_arr[index]))
    plt.axis('off')

    st.pyplot(fig)

st.title('Simple Lane Detection')
"""With Canny Edge detection + Hough Transform"""



# if 'is_default' not in st.session_state:
#     st.session_state.is_default = True

# if st.session_state.is_default:
#     myBtn = st.button('Choose your video')
#     st.session_state.is_default = False

#     st.subheader("Preview (default video)")
#     fig, ax = plt.subplots()
#     ax.imshow(plot_lanes(st.session_state.frames[0]))
#     plt.axis('off')

#     st.pyplot(fig)

# else:
#     myBtn = st.button('Use default video')
#     st.session_state.is_default = True

#     video = st.file_uploader("Select a video from your files", accept_multiple_files=False)
#     if video is not None:
#         st.session_state.frames = iio.imread(video, plugin="pyav")

#         st.subheader("Preview (your video)")
#         fig, ax = plt.subplots()
#         ax.imshow(plot_lanes(st.session_state.frames[0]))
#         plt.axis('off')

#         st.pyplot(fig)

with st.sidebar:
    st.subheader("Canny Edge Detection Parameters") 
    
    if not st.toggle('Hide Canny Edge Detection'):
        st.session_state.params['canny_low'], st.session_state.params['canny_high'] = st.slider('threshold range', 0, 200, (24, 113), help="")
        "---" # divider
        st.subheader("Hough Transform Parameters") 
        if not st.toggle('Hide Hough Transform'):
            st.session_state.params['kernel_size'] = st.number_input('kernel size', min_value=1, max_value=50, value=5, step=1, help='')
            st.session_state.params['rho'] = st.slider('rho', 0, 100, 12, help='distance resolution in pixels of the Hough grid')
            st.session_state.params['theta'] = math.radians(st.slider('theta (will be converted to radians)', 0, 360, 60, step=1, help='angular resolution in pixels of the Hough grid'))
            st.session_state.params['hough_threshold'] = st.slider('threshold', value=1, help='minimum number of votes (intersections in Hough grid cell)')
            st.session_state.params['min_line_length'] = st.number_input('minimum line length', help='minimum number of pixels making up a line')
            st.session_state.params['max_line_gap'] = st.number_input('maximum line gap', help='maximum gap in pixels between connectable line segments')


default, upload_tab = st.tabs(['default', 'upload video'])


with default:
    frames = iio.imread("highway.mp4", plugin="pyav")
    fps = cv2.VideoCapture("lane_driving.mp4").get(cv2.CAP_PROP_FPS) # get fps in original video

    preview(frames) # display preview
    if st.button('Process full video', key='process_default'):
        annotate_video(frames, fps)
        if st.session_state.video_ready:
            st.download_button('Download video', 'output.gif')

with upload_tab:
    video = st.file_uploader("Select a video from your files", accept_multiple_files=False)
    if video is not None:
        uploaded_frames = iio.imread(video.getvalue(), plugin="pyav") 
        fps = cv2.VideoCapture("lane_driving.mp4").get(cv2.CAP_PROP_FPS) # get fps in original video
        
        preview(uploaded_frames) # display preview

        if st.button('Process full video', key='process_upload'):
            annotate_video(uploaded_frames, fps)
            if st.session_state.video_ready:
                st.download_button('Download video', 'output.gif')

# # toggle button to load/download video
# proc_button = st.empty()
# select_video = proc_button.button('Process full video', key='select_new')
# if select_video:
#     proc_button.button('Download')
    
# # show preview on chosen video
# if st.session_state["select_new"]:
#     # choose video
#     video = st.file_uploader("Select a video from your files", accept_multiple_files=False)
#     if video is not None:
#         st.session_state.frames = iio.imread(video.name, plugin="pyav")
#         preview()
# else:
#     st.session_state.frames = iio.imread("lane_driving.mp4", plugin="pyav")
#     preview()  # display preview







# frames = iio.imread("lane_driving.mp4", plugin="pyav") # set default video
# preview(frames)

# def upload():
#     video = st.file_uploader("Select a video from your files", type=["mp4", "mpeg"])
#     video
#     # if video is not None:
#         # frames = iio.imread(video.name, plugin="pyav")
#         # preview(frames)
#         # frames.shape

# st.button('Upload a video', on_click=upload)


# st.video(frames)


# # video_path = os.getcwd() + video.filename
# if video is not None:
#     sample = frames
# else:
#     video_path = "lane_driving.mp4"


# plt.imshow();


# # Create a figure
# fig = plt.figure()


# # Define a function to update the image in the figure
# def update_image(n):
#     plt.imshow(annotated_frames[n])


# # Create an animation object
# anim = animation.FuncAnimation(fig, update_image, frames=len(annotated_frames))


# # anim.save('annotated_lanes.mp4', writer='ffmpeg', fps=30)
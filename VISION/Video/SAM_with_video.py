### From https://medium.com/@tenyks_blogger/segment-anything-model-2-sam-2-gpt-4o-cascading-foundation-models-via-visual-prompting-76158ff0b9f4

import os
HOME = os.getcwd()

# Clone the repository
!git clone https://github.com/facebookresearch/segment-anything-2.git
%cd {HOME}/segment-anything-2

# install the python libraries for "segment-anything-2"
!pip install -e . -q
!pip install -e ".[demo]" -q
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P {HOME}/checkpoints

from sam2.build_sam import build_sam2_video_predictor

def refine_mask_with_coordinates(coordinates, ann_frame_idx, ann_obj_id, show_result=True):
    """
    Refine a mask by adding new points using a SAM predictor.

    Args:
    coordinates (list): List of [x, y] coordinates, 
        e.g., [[210, 350], [250, 220]]
    ann_frame_idx (int): The index of the frame being processed
    ann_obj_id (int): A unique identifier for the object being segmented
    show_result (bool): Whether to display the result (default: True)
    """
    # Convert the list of coordinates to a numpy array
    points = np.array(coordinates, dtype=np.float32)
    
    # Create labels array (assuming all points are positive clicks)
    labels = np.ones(len(coordinates), dtype=np.int32)

    # Add new points to the predictor
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    if show_result:
        # Display the results
        plt.figure(figsize=(12, 8))
        plt.title(f"Frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.show()

sam2_checkpoint = f"{HOME}/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# Extract the frames
video_path = f"{HOME}/segment-anything-2/SAM2_gymnastics.mp4"
output_path = f"{HOME}/segment-anything-2/outputs/gymnastics"
!ffmpeg -i {video_path} -q:v 2 -start_number 0 {output_path}/'%05d.jpg'

video_dir = f"{HOME}/segment-anything-2/outputs/gymnastics"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)

refine_mask_with_coordinates([[950, 700], [950, 600], [950, 500]], 0, 1)

###### run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


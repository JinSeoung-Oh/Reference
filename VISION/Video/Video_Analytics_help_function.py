## From https://medium.com/@VK_Venkatkumar/video-analytics-counting-speed-distance-estimation-with-graph-visualization-yolov10-da1c24f7f245

'''
Final code: Video Analytics Specific Object

Guidance

1. User Input: Specific object
2. Specific Object detection, Speed and distance estimation
3. Graph Analytics: Pie, Area, Multi-Class line

!pip install ultralytics

'''

#Helper function
def create_pie_chart(data):
    fig, ax = plt.subplots(figsize=(4, 3))  # Aspect ratio of 4:3
    ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%')
    ax.legend()
    ax.set_title("Total Percentage of Individual Class Perspective")
    plt.close(fig)
    return fig

def create_area_plot(class_counts_over_time):
    fig, ax = plt.subplots(figsize=(4, 3))  # Aspect ratio of 4:3
    sorted_keys = sorted(class_counts_over_time.keys())
    for cls in sorted_keys:
        ax.fill_between(range(len(class_counts_over_time[cls])), class_counts_over_time[cls], label=cls, alpha=0.6)
    ax.legend()
    ax.set_title("Distribution of Each Class Over Time")
    ax.set_xlabel("Frame Count")
    ax.set_ylabel("Count")
    plt.close(fig)
    return fig

def create_multiple_line_plot(speed_data, distance_data, frame_count):
    fig, ax = plt.subplots(figsize=(4, 3))  # Aspect ratio of 4:3
    for track_id in speed_data.keys():
        ax.plot(range(frame_count), speed_data[track_id], label=f"Speed {track_id}")
    for track_id in distance_data.keys():
        ax.plot(range(frame_count), distance_data[track_id], label=f"Distance {track_id}")
    ax.legend()
    ax.set_title("Speed and Distance Identification of Each Class")
    ax.set_xlabel("Frame Count")
    ax.set_ylabel("Value")
    plt.close(fig)
    return fig

def create_scatter_plot(data):
    fig, ax = plt.subplots(figsize=(4, 3))  # Aspect ratio of 4:3
    x = list(data.keys())
    y = list(data.values())
    ax.scatter(x, y)
    ax.set_title("Class Distribution Scatter Plot")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.close(fig)
    return fig

def fig_to_img(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img

def resize_and_place_image(base_image, overlay_image, position):
    overlay_image_resized = cv2.resize(overlay_image, (w // 3, h // 3))
    x, y = position
    base_image[y:y + overlay_image_resized.shape[0], x:x + overlay_image_resized.shape[1]] = overlay_image_resized
    return base_image

def draw_visualizations(frame, data, labels, speed_data, distance_data, class_counts_over_time, frame_count):
    vis_frame = np.zeros((h, w // 3, 3), dtype=np.uint8)

    # Create Pie Chart
    if data:
        pie_chart = create_pie_chart(data)
        pie_chart_img = fig_to_img(pie_chart)
        vis_frame = resize_and_place_image(vis_frame, pie_chart_img, (0, 0))

    # Create Area Plot
    if class_counts_over_time:
        area_plot = create_area_plot(class_counts_over_time)
        area_plot_img = fig_to_img(area_plot)
        vis_frame = resize_and_place_image(vis_frame, area_plot_img, (0, h // 3))

    # Create Multiple Line Plot
    if speed_data or distance_data:
        line_plot = create_multiple_line_plot(speed_data, distance_data, frame_count)
        line_plot_img = fig_to_img(line_plot)
        vis_frame = resize_and_place_image(vis_frame, line_plot_img, (0, 2 * (h // 3)))

    combined_frame = np.hstack((frame, vis_frame))
    return combined_frame

def pad_lists_to_length(data_dict, length, default_value=0):
    for key in data_dict.keys():
        if len(data_dict[key]) < length:
            data_dict[key] += [default_value] * (length - len(data_dict[key]))

'''
Main function:

Specific input based video analytics 
(object count, speed, distance estimation..)

'''

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.solutions import speed_estimation

# Initialize YOLO models
object_detection_model = YOLO("yolov8s.pt")
speed_estimation_model = YOLO("yolov8n.pt")
names = speed_estimation_model.model.names

# Open video file
cap = cv2.VideoCapture("/content/drive/MyDrive/yolo/race.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer
out = cv2.VideoWriter("Distribution_speed_distance_visual_scatter_unique1hor_car_overall.avi", cv2.VideoWriter_fourcc(*"MJPG"),15, (w + w // 3, h))

frame_count = 0
data = {}
labels = []
class_counts_over_time = {}
speed_over_time = {}
distance_over_time = {}

# Center point and pixel per meter for distance calculation
center_point = (0, h)
pixel_per_meter = 10

# Line points for speed estimation
line_pts = [(0, 360), (1280, 360)]

# Initialize speed-estimation object
speed_obj = speed_estimation.SpeedEstimator(names=names, reg_pts=line_pts, view_img=False)

# Colors for text and bounding box
txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

print('Example input: horse:17, person: 0,car: 2, van: 8,bus: 5,tree: 62')
# Allow user to input desired classes
user_input = input("Enter desired classes with their IDs (format: 'class1:id1,class2:id2,...'): ")
# Example input: "person:0,car:2,horse:17"
desired_classes = {}
for item in user_input.split(','):
    cls, cls_id = item.split(':')
    desired_classes[cls.strip()] = int(cls_id.strip())

print("Desired classes:", desired_classes)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Object detection for speed estimation
    speed_tracks = speed_estimation_model.track(frame, persist=True, show=False)
    frame = speed_obj.estimate_speed(frame, speed_tracks)

    # Object detection for distance estimation
    annotator = Annotator(frame, line_width=2)
    results = object_detection_model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            cls_name = object_detection_model.names[int(cls)]
            if cls_name in desired_classes and desired_classes[cls_name] == cls:  # Filter desired classes and IDs
                if cls_name not in labels:
                    labels.append(cls_name)

                if cls_name in data:
                    data[cls_name] += 1
                else:
                    data[cls_name] = 1

                annotator.box_label(box, label=str(track_id), color=bbox_clr)
                annotator.visioneye(box, center_point)

                x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # Bounding box centroid

                distance = (math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)) / pixel_per_meter

                text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), txt_background, -1)
                cv2.putText(frame, f"Distance: {distance:.2f} m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 3)

                if track_id not in distance_over_time:
                    distance_over_time[track_id] = [0] * (frame_count - 1)
                distance_over_time[track_id].append(distance)

                speed = speed_obj.speeds.get(track_id, 0) if hasattr(speed_obj, 'speeds') else 0
                if track_id not in speed_over_time:
                    speed_over_time[track_id] = [0] * (frame_count - 1)
                speed_over_time[track_id].append(speed)

                if cls_name not in class_counts_over_time:
                    class_counts_over_time[cls_name] = [0] * frame_count
                if len(class_counts_over_time[cls_name]) < frame_count:
                    class_counts_over_time[cls_name].extend([0] * (frame_count - len(class_counts_over_time[cls_name])))
                class_counts_over_time[cls_name][-1] += 1

    # Pad lists to current frame count to ensure equal lengths
    pad_lists_to_length(distance_over_time, frame_count)
    pad_lists_to_length(speed_over_time, frame_count)

    # Draw combined visualizations on the frame
    combined_frame = draw_visualizations(frame, data, labels, speed_over_time, distance_over_time, class_counts_over_time, frame_count)

    # Write the frame with visualizations
    out.write(combined_frame)

    # Clear counts for next frame
    data = {}

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Generate and overlay scatter plot on the final frame
final_frame = np.zeros((h, w, 3), dtype=np.uint8)
scatter_plot = create_scatter_plot(class_counts_over_time)
scatter_plot_img = fig_to_img(scatter_plot)
final_frame = resize_and_place_image(final_frame, scatter_plot_img, (0, 0))

# Save the final frame with the scatter plot
cv2.imwrite("final_frame_with_scatter_plot.png", final_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# Print overall analytics
total_counts = sum(sum(counts) for counts in class_counts_over_time.values())
print(f"Overall total count: {total_counts}")
for cls, counts in class_counts_over_time.items():
    print(f"Total count for {cls}: {sum(counts)}")

best_speed = max((max(speeds) for speeds in speed_over_time.values()), default=0)
print(f"Overall best speed: {best_speed} m/s")
best_distance = max((max(distances) for distances in distance_over_time.values()), default=0)
print(f"Overall best distance: {best_distance} meters")

# From https://medium.com/voxel51/how-to-detect-small-objects-cfa569b4d5bd

# Set basic yolov8 model
! pip install -U fiftyone sahi ultralytics huggingface_hub --quiet

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.huggingface as fouh
from fiftyone import ViewField as F

from ultralytics import YOLO

ckpt_path = "yolov8l.pt"
model = YOLO(ckpt_path)

dataset.apply_model(model, label_field="base_model")
session.view = dataset.view()

mapping = {"pedestrians": "person", "people": "person", "van": "car"}
mapped_view = dataset.map_labels("ground_truth", mapping)

def get_label_fields(sample_collection):
    """Get the (detection) label fields of a Dataset or DatasetView."""
    label_fields = list(
        sample_collection.get_field_schema(embedded_doc_type=fo.Detections).keys()
    )
    return label_fields

def filter_all_labels(sample_collection):
    label_fields = get_label_fields(sample_collection)

    filtered_view = sample_collection

    for lf in label_fields:
        filtered_view = filtered_view.filter_labels(
            lf, F("label").is_in(["person", "car", "truck"]), only_matches=False
        )
    return filtered_view

filtered_view = filter_all_labels(mapped_view)
session.view = filtered_view.view()

############################ With SAHI for Hyper Inference #######################################
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=ckpt_path,
    confidence_threshold=0.25, ## same as the default value for our base model
    image_size=640,
    device="cpu", # or 'cuda' if you have access to GPU
)

result = get_prediction(dataset.first().filepath, detection_model)
print(result)
print(result.to_fiftyone_detections())
#
sliced_result = get_sliced_prediction(
    dataset.skip(40).first().filepath,
    detection_model,
    slice_height = 320,
    slice_width = 320,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
)

num_sliced_dets = len(sliced_result.to_fiftyone_detections())
num_orig_dets = len(result.to_fiftyone_detections())

print(f"Detections predicted without slicing: {num_orig_dets}")
print(f"Detections predicted with slicing: {num_sliced_dets}")

def predict_with_slicing(sample, label_field, **kwargs):
    result = get_sliced_prediction(
        sample.filepath, detection_model, verbose=0, **kwargs
    )
    sample[label_field] = fo.Detections(detections=result.to_fiftyone_detections())


kwargs = {"overlap_height_ratio": 0.2, "overlap_width_ratio": 0.2}

for sample in dataset.iter_samples(progress=True, autosave=True):
    predict_with_slicing(sample, label_field="small_slices", slice_height=320, slice_width=320, **kwargs)
    predict_with_slicing(sample, label_field="large_slices", slice_height=480, slice_width=480, **kwargs)

filtered_view = filter_all_labels(mapped_view)
session = fo.launch_app(filtered_view, auto=False)



base_results = filtered_view.evaluate_detections("base_model", gt_field="ground_truth", eval_key="eval_base_model")
large_slice_results = filtered_view.evaluate_detections("large_slices", gt_field="ground_truth", eval_key="eval_large_slices")
small_slice_results = filtered_view.evaluate_detections("small_slices", gt_field="ground_truth", eval_key="eval_small_slices")

########################## Filtering for only small boxes ################################

box_width, box_height = F("bounding_box")[2], F("bounding_box")[3]
rel_bbox_area = box_width * box_height

im_width, im_height = F("$metadata.width"), F("$metadata.height")
abs_area = rel_bbox_area * im_width * im_height

small_boxes_view = filtered_view
for lf in get_label_fields(filtered_view):
    small_boxes_view = small_boxes_view.filter_labels(lf, abs_area < 32**2, only_matches=False)

session.view = small_boxes_view.view()

## Evaluating on only small boxes
small_boxes_base_results = small_boxes_view.evaluate_detections("base_model", gt_field="ground_truth", eval_key="eval_small_boxes_base_model")
small_boxes_large_slice_results = small_boxes_view.evaluate_detections("large_slices", gt_field="ground_truth", eval_key="eval_small_boxes_large_slices")
small_boxes_small_slice_results = small_boxes_view.evaluate_detections("small_slices", gt_field="ground_truth", eval_key="eval_small_boxes_small_slices")

## Printing reports
print("Small Box — Base model results:")
small_boxes_base_results.print_report()

print("-" * 50)
print("Small Box — Large slice results:")
small_boxes_large_slice_results.print_report()

print("-" * 50)
print("Small Box — Small slice results:")
small_boxes_small_slice_results.print_report()

import os
import sys
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import re
import cv2

import fiftyone as fo
import fiftyone.core.fields as fof
from fiftyone import ViewField as F
from fiftyone.utils.yolo import YOLOv5DatasetExporter, YOLOv5DatasetImporter
import fiftyone.utils.iou as foiou
import fiftyone.core.labels as fol
from fiftyone.core.labels import Detections, Detection

from ultralytics import YOLO
import argparse


class_names = ['Ball', 'Goalpost', 'Person', 'LCross', 'TCross', 'XCross', 'PenaltyPoint', 'Opponent', 'BRMarker']

CONF_THRESH_4_FILTERED_UPLOADING = 0.5
IOU_THRESH_4_FILTERED_UPLOADING = 0.5
TARGET_FIELD_4_FILTERED_UPLOADING = "PLACE_HOLDER" #
GT_FIELD_4_FILTERED_UPLOADING = "PLACE_HOLDER"
DEST_FIELD_4_DOWNLOADING = "PLACE_HOLDER"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MERGED_FIELD = f"merged_field_for_upload{timestamp}"  #


class CustomYOLOv8Model():  
    def __init__(self, model_path="yolov8n.pt", iou=0.5, conf=0.25, prelable=False, prelable_class=None):   # constructor
        self.model = YOLO(model_path)
        self.iou = iou
        self.conf = conf
        self.prelabel = prelable
        self.prelabel_class = prelable_class
        self.labels = {0: 'Ball',
                       1: 'Goalpost',
                       2: 'Person',
                       3: 'LCross',
                       4: 'TCross',
                       5: 'XCross',
                       6: 'PenaltyPoint',
                       7: 'Opponent',
                       8: 'BRMarker'}

    def predict(self, filepath):    # make predictions and do fiftyone style formatting 
        results = self.model.predict(source=filepath, iou=self.iou, conf=self.conf)

        detections = []
        result = results[0]
        width, height = result.orig_shape[1], result.orig_shape[0]

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            # label = self.model.names[int(box.cls[0])]
            label = self.labels[int(box.cls[0])]
            if self.prelabel and len(self.prelabel_class) > 0 and not label in self.prelabel_class:
                continue

            detections.append(
                fol.Detection(
                    label=label,
                    bounding_box=[
                        xyxy[0] / width,
                        xyxy[1] / height,
                        (xyxy[2] - xyxy[0]) / width,
                        (xyxy[3] - xyxy[1]) / height,
                    ],
                    confidence=conf,
                )
            )

        return fol.Detections(detections=detections)
    
# def find_first_dir_with_color_images(start_dir, image_exts=(".jpg", ".jpeg", ".png", ".bmp")):
#     files = os.listdir(start_dir)
#     for f in files:
#         if f.lower().startswith("color") and f.lower().endswith(image_exts):
#             return start_dir
#     subdirs = [os.path.join(start_dir, d) for d in files if os.path.isdir(os.path.join(start_dir, d))]
#     for subdir in subdirs:
#         try:
#             return find_first_dir_with_color_images(subdir, image_exts)
#         except ValueError:
#             continue
#     raise ValueError(f"No folder with color images found in {start_dir}")

def find_all_dirs_with_color_images(start_dir, image_exts=(".jpg", ".jpeg", ".png", ".bmp")):
    result_dirs = []

    files = os.listdir(start_dir)
    has_color_image = any(
        f.lower().startswith("color") and f.lower().endswith(image_exts)
        for f in files
    )
    if has_color_image:
        result_dirs.append(start_dir)

    subdirs = [os.path.join(start_dir, d) for d in files if os.path.isdir(os.path.join(start_dir, d))]
    for subdir in subdirs:
        result_dirs.extend(find_all_dirs_with_color_images(subdir, image_exts))

    return result_dirs


def get_predicted_ds(input_dataset, models, prediction_fields):   # predict on the fiftyone dataset (in one-field-on-one-model fashion and save/write predictions in the field) 
    cl_dataset_name = f'cl_{input_dataset.name}'
    if cl_dataset_name in fo.list_datasets():
        dataset_cl = fo.load_dataset(cl_dataset_name)
    else:
        dataset_cl = input_dataset.clone(name = cl_dataset_name) #
        dataset_cl.persistent = True   #
        print(f"Adding clone dataset: {cl_dataset_name}")
    if not len(models) == len(prediction_fields):
        raise Exception('models and prediction_fields not equal length')
    for model, prediction_field in zip(models, prediction_fields):
        if prediction_field not in dataset_cl.get_field_schema():
            print(f"Adding prediction field: {prediction_field}")
        else:
            print(f"Prediction field {prediction_field} already exists, skipping prediction")
            continue
        samples_to_delete = []
        for sample in dataset_cl:
            img_path = sample.filepath
            print(f"Processing: {img_path}")
            if not os.path.exists(img_path):
                print(f"Warning: File not found, deleting from dataset: {img_path}")
                samples_to_delete.append(sample.id)
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Cannot open image or corrupted, deleting from dataset: {img_path}")
                samples_to_delete.append(sample.id)
                continue
            try:
                sample[prediction_field] = model.predict(img_path)
                sample.save()
            except Exception as e:
                print(f"Warning: Prediction failed on {img_path}, deleting from dataset. Error: {e}")
                samples_to_delete.append(sample.id)
                continue
        if samples_to_delete:
            print(f"Deleting {len(samples_to_delete)} bad samples from dataset...")
            dataset_cl.delete_samples(samples_to_delete)
    return dataset_cl

def find_folders_with_images_and_labels(root_dir): 
    if not os.path.isdir(root_dir):
        raise ValueError(f"Directory does not exist or is not a folder: {root_dir}")
    matched_folders = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        try:
            dirnames_lower = set(name.lower() for name in dirnames)

            if 'images' in dirnames_lower and 'labels' in dirnames_lower:
                images_path = os.path.join(dirpath, 'images')
                labels_path = os.path.join(dirpath, 'labels')

                # Check if both 'images' and 'labels' folders are not empty
                if os.listdir(images_path) and os.listdir(labels_path):
                    matched_folders.append(dirpath)

        except PermissionError as e:
            print(f"Permission error when accessing {dirpath}: {e}, skipping this folder.")
        except Exception as e:
            print(f"Error occurred while scanning {dirpath}: {e}, skipping this folder.")

    print(f"Found {len(matched_folders)} folders with 'images' and 'labels':")
    return matched_folders            # a list of paths to the subfolders with images and labels

def generate_dataset_yaml(  # generate a file with multiple pieces of useful information for model training
    train_dir,
    val_dir,
    test_dir=None,
    class_names=None,
    output_path="dataset.yaml"
):
    for d, name in zip([train_dir, val_dir], ['train_dir', 'val_dir']):
        if not os.path.isdir(d):
            raise ValueError(f"{name} does not exist or is not a valid directory: {d}")
    # Auto-generate class names if not provided
    if class_names is None:
        # Example: generate "class0", "class1", ...
        # Replace this with your actual class list if you have one
        num_classes = 80  # set your number of classes
        class_names = [f"class{i}" for i in range(num_classes)]

    data = {
        "train": os.path.abspath(train_dir),
        "val": os.path.abspath(val_dir),
        "nc": len(class_names),
        "names": class_names
    }

    if test_dir:
        data["test"] = os.path.abspath(test_dir)

    with open(output_path, "w") as f:
        yaml.dump(data, f)

    print(f"Generated {output_path}")

def get_dataset_info(dataset_dir):
    if fo.dataset_exists(dataset_dir):
        dataset = fo.load_dataset(dataset_dir)
    else:
        print(f'Creating dataset: {dataset_dir}')
        dataset = fo.Dataset(dataset_dir)
        img_path = dataset_dir + '/images'
        yaml_path = dataset_dir + '/dataset.yaml'
        generate_dataset_yaml(img_path, img_path, output_path=yaml_path, class_names=class_names)
        dataset.add_dir(dataset_dir=dataset_dir, dataset_type=fo.types.YOLOv5Dataset)
    print(f'dataset: {dataset_dir}')
    ball_view = dataset.filter_labels('ground_truth', F("label") == "Ball")
    bbox_area_expr =  F("bounding_box")[2] * F("bounding_box")[3] * 1280 * 720     
    ball_view = ball_view.set_field('ground_truth.detections.bbox_area', bbox_area_expr)    #separating the balls from others and get list of S(bounding_box)

    def categorize_distance(area, bbox):  # Sorting balls into far/medium/near
        width = bbox[2] * 1280
        height = bbox[3] * 720
        w_h_ratio = width / height * 1.0
        near_square = False
        if w_h_ratio >= 0.8 and w_h_ratio <= 1.2:
            near_square = True
        
        if near_square:
            if area <= 500:   # edge 22
                return 'far'
            elif area <= 4500: # edge 67
                return 'medium'
            else:
                return 'near'
        else:
            edge = max(width, height)
            if edge <= 22:
                return 'far'
            elif edge <= 67:
                return 'medium'
            else:
                return 'near'


    for sample in ball_view.iter_samples(autosave=True):   
        for det in sample.ground_truth.detections:
            det['ball_distance'] = categorize_distance(det.bbox_area, det.bounding_box) #attach a detection-level attribute(distance) to every ground_truth ball 
    
    dataset_path = Path(dataset_dir)
    if dataset_path.name == 'train':
        dataset.tag_samples('train')
    elif dataset_path.name == 'val':
        dataset.tag_samples('val')
    elif dataset_path.name == 'test':
        dataset.tag_samples('test')

    near_ball_view = ball_view.filter_labels('ground_truth', F("ball_distance") == "near")
    medium_ball_view = ball_view.filter_labels('ground_truth', F("ball_distance") == "medium")
    far_ball_view = ball_view.filter_labels('ground_truth', F("ball_distance") == "far")

    near_ball_view.tag_samples('near_ball')
    medium_ball_view.tag_samples('medium_ball')
    far_ball_view.tag_samples('far_ball')


def filter_detections_for_uploading(dataset):

    merged_detections = []

    for sample in dataset: #找出每一张图里的各种detections
        gts = []
        if sample.has_field(GT_FIELD_4_FILTERED_UPLOADING) and sample[GT_FIELD_4_FILTERED_UPLOADING] is not None:
            gts = sample[GT_FIELD_4_FILTERED_UPLOADING].detections

        raw_preds = []
        if sample.has_field(TARGET_FIELD_4_FILTERED_UPLOADING) and sample[TARGET_FIELD_4_FILTERED_UPLOADING] is not None:
            raw_preds = sample[TARGET_FIELD_4_FILTERED_UPLOADING].detections

        preds = [
            det for det in raw_preds
            if det.confidence is not None and det.confidence >= CONF_THRESH_4_FILTERED_UPLOADING
        ]

        iou_matrix = foiou.compute_ious(preds, gts, classwise=True) ##

        keep_preds = []
        for i, pred in enumerate(preds):
            overlaps = iou_matrix[i]  # 每一个模型预测框的所有的IoU都在这儿
            max_iou = max(overlaps) if len(overlaps) > 0 else 0
            if max_iou < IOU_THRESH_4_FILTERED_UPLOADING:
                keep_preds.append(pred)

        merged = Detections(detections=gts + keep_preds)
        merged_detections.append(merged)

    dataset.set_values(MERGED_FIELD, merged_detections)
    return dataset

def choose_required_field(prompt):
    while True:
        try:
            idx = int(input(prompt))
            if 0 <= idx < len(label_fields):
                return label_fields[idx]
            else:
                print(f"Invalid index: choose between 0 and {len(label_fields)-1}")
        except ValueError:
            print("Please enter a valid number.")

def choose_optional_field(prompt):
    none_index = len(label_fields)
    while True:
        try:
            idx = int(input(f"{prompt} (or {none_index} for None): "))
            if 0 <= idx < len(label_fields):
                return label_fields[idx]
            elif idx == none_index:
                return None
            else:
                print(f"Invalid index: choose 0 to {len(label_fields)-1} or {none_index} for None")
        except ValueError:
            print("Please enter a valid number.")

def find_first_valid_dir(root, exts):
    for dirpath, dirnames, filenames in os.walk(root):
        if any(file.lower().endswith(exts) for file in filenames):
            return dirpath
    return None

def find_first_color_image_dir(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for file in filenames:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and file.startswith("color_"):
                return dirpath
    return None



if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Process dataset and models")
    parser.add_argument("datasetname", type=str, help="Path to dataset folder") 
    parser.add_argument("model_paths", nargs="*", help="Paths to YOLO model files")
    parser.add_argument("--prelabel", action="store_true", help="Enable prelabling")
    parser.add_argument("--exportdataset", action="store_true", help="Enable exporting")
    parser.add_argument("--prelabel-class", nargs="*", default=None, help="Class name for prelabeling")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation")
    parser.add_argument("--upload", action="store_true", help="Upload dataset to CVAT")
    parser.add_argument("--prelabelupload", action="store_true", help="Upload dataset to CVAT")
    parser.add_argument("--filteredupload", action="store_true", help="Upload dataset to CVAT with filtered boxes")
    parser.add_argument("--filterconf", type=float, default=0.5, help="Confidence threshold for predictions")
    parser.add_argument("--filterIOU", type=float, default=0.5, help="IOU threshold for predictions")
    parser.add_argument("--download", action="store_true", help="Reload dataset from cvat after annotating and validating manually")
    parser.add_argument("--dest", type=str, default="ground_truth", help="Destination of the download content from CVAT") 
    parser.add_argument("--export-bad-case", action="store_true", help="Export bad cases (FN/FP)")
    parser.add_argument("--fplabel", type=str, default="Ball", help="FP label selection") 
    parser.add_argument("--importdataset", type=str, default=None, help="Path to dataset folder") 
    parser.add_argument("--rename", type=str, default=None, help="Rename an imported dataset") 
    parser.add_argument("--project", type=str, default= "detection2507", help="The cvat project to be sent to.") 
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold for predictions")
    args = parser.parse_args()

    print(f"Dataset name: {args.datasetname}")
    print(f"Model paths: {args.model_paths}")
    print(f"Exporting enabled: {args.exportdataset}")
    print(f"Prelabeling enabled: {args.prelabel}")
    print(f"Prelabel class: {args.prelabel_class}")
    print(f"Evaluating enabled: {args.eval}")
    print(f"Export bad cases: {args.export_bad_case}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Import dataset directory: {args.importdataset}")
    print(f"Downloading destination field: {args.dest}")
    print(f"Destination CVAT project: {args.project}")

    datasetname = args.datasetname
    model_path_list = args.model_paths
    confidence = args.confidence
    DEST_FIELD_4_DOWNLOADING = args.dest
    CONF_THRESH_4_FILTERED_UPLOADING = args.filterconf
    IOU_THRESH_4_FILTERED_UPLOADING = args.filterIOU
    FP_LABEL_4_EXPORTING = args.fplabel
    CVAT_DEST_PROJECT = args.project
    RENAME_IMPORTED_DATASET = args.rename

    print(f"Confidence threshold for filtered upload: {CONF_THRESH_4_FILTERED_UPLOADING}")
    print(f"IOU threshold for filtered upload: {IOU_THRESH_4_FILTERED_UPLOADING}")
    print(model_path_list)

    model_list = []
    model_name_list = []
    for model_path in model_path_list:
        model_name = f'{model_path}_conf_{confidence}'.replace('/','_').replace('.','0')
        model_name_list.append(model_name)
        model_list.append(CustomYOLOv8Model(model_path, iou=0.4, conf=confidence, prelable=args.prelabel, prelable_class=args.prelabel_class))

    # import ipdb; ipdb.set_trace()
    # if fo.dataset_exists(datasetname):
    #     dataset = fo.load_dataset(datasetname)
    #     print(f'Loaded existing dataset: {datasetname}')
    # else:
    #     dataset = fo.Dataset(datasetname)
    #     dataset.persistent = True  ###
    #     exported_dataset_folders = find_folders_with_images_and_labels(datasetname)
    #     print(f'Found {len(exported_dataset_folders)} exported dataset folders')
    #     import ipdb; ipdb.set_trace()
    #     if not exported_dataset_folders:
    #         # load dataset from directory
    #         color_paths = [os.path.join(datasetname, d) for d in os.listdir(datasetname) if d.startswith('color_')]

    #         dataset.add_images(color_paths)
    #     else: 
    #         for _dataset_name in exported_dataset_folders:
    #             tmp_dataset = fo.load_dataset(_dataset_name)
    #             dataset.merge_samples(tmp_dataset)

    if args.importdataset:
        datasetname = f"imported_{Path(args.importdataset).stem}"
        if fo.dataset_exists(datasetname):
            dataset = fo.load_dataset(datasetname)
            print(f"Loaded imported dataset: {datasetname}")
        else:
            matched_folders = find_folders_with_images_and_labels(args.importdataset)
            if not matched_folders:
                raise ValueError(f"No 'images' and 'labels' folders found under {args.importdataset}")
            selected_folder = matched_folders[0]
            print(f"Selected Folder: {selected_folder}")
            yaml_path = os.path.join(selected_folder, "dataset.yaml")
            if not os.path.exists(yaml_path):
                print(f"dataset.yaml not found in {selected_folder}, auto-generating...")
                images_root = os.path.join(selected_folder, "images")
                labels_root = os.path.join(selected_folder, "labels")
                image_dir = find_first_color_image_dir(images_root)
                label_dir = find_first_valid_dir(labels_root, ('.txt',))
                if image_dir is not None and label_dir is not None:
                    print(f"Found images in: {image_dir}")
                    print(f"Found labels in: {label_dir}")
                    generate_dataset_yaml(
                        train_dir=image_dir,
                        val_dir=image_dir,
                        class_names=class_names,
                        output_path=yaml_path
                    )
                else:
                    raise ValueError("No valid color images or labels found under images/ and labels/")
            else:
                print(f"Found existing dataset.yaml in {selected_folder}, using it directly.")
            importer = YOLOv5DatasetImporter(
                dataset_dir=selected_folder,
                yaml_path="dataset.yaml",
                split="val",
                label_type="detections",
                include_all_data=True
            )
            final_datasetname = datasetname if RENAME_IMPORTED_DATASET is None else f"imported_{RENAME_IMPORTED_DATASET}"
            dataset = fo.Dataset.from_importer(importer, name=final_datasetname)
            dataset.persistent = True
            print(f"Imported YOLOv5 dataset: {final_datasetname}")
            print(f"Imported {len(dataset)} samples into FiftyOne") 
    else:
        if fo.dataset_exists(datasetname):
            dataset = fo.load_dataset(datasetname)
            print(f'Loaded existing dataset: {datasetname}')
        else:
            print(f"datasetname: {datasetname}")
            dataset = fo.Dataset(datasetname)
            dataset.persistent = True
            try:
                target_dirs = find_all_dirs_with_color_images(datasetname)
                if not target_dirs:
                    raise ValueError(f"No directories with color images found under {datasetname}")
                color_paths = []
                for target_dir in target_dirs:
                    for fname in os.listdir(target_dir):
                        if fname.lower().startswith('color') and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            color_paths.append(os.path.join(target_dir, fname))
                if not color_paths:
                    raise ValueError(f"No color images found in the matched directories under {datasetname}")
                dataset.add_images(color_paths)
                print(f"Added {len(color_paths)} images from {len(target_dirs)} folders to dataset {datasetname}")
            except ValueError as e:
                print(f"Error: {e}")
                raise e

    dataset_cl = get_predicted_ds(dataset, model_list, model_name_list)

    if args.download:
        runs = dataset_cl.list_annotation_runs()
        if runs:
            annotation_key = runs[len(runs)-1]
            dataset_cl.load_annotations(annotation_key, dest_field= str(DEST_FIELD_4_DOWNLOADING))
        else:
            print("No annotation runs found.")

    if args.exportdataset:
        export_dir = f'fiftyone_export/prelable/{dataset_cl.name}_{timestamp}'
        exporter = YOLOv5DatasetExporter(export_dir, classes=class_names)
        schema = dataset_cl.get_field_schema()
        label_fields = [
            name
            for name, field in schema.items()
            if isinstance(field, fof.EmbeddedDocumentField)
            and issubclass(field.document_type, (fol.Detection, fol.Detections))
        ]           

        if not label_fields:
            print("No detection-type label fields found.")
            exit(1)
        print("\n")
        print("Available detection label fields:")
        for i, name in enumerate(label_fields):
            print(f"[{i}] {name}")
        FIELD_4_EXPORTING = choose_optional_field(
            "Select the number for the field to export"
        )
        dataset_cl.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            dataset_exporter=exporter,
            label_field=FIELD_4_EXPORTING  # export predictions instead of 'ground_truth'
        )
        print(f"Export directory: {export_dir}")
        
    if args.eval:
        schema = dataset_cl.get_field_schema()
        label_fields = [
            name
            for name, field in schema.items()
            if isinstance(field, fof.EmbeddedDocumentField)
            and issubclass(field.document_type, (fol.Detection, fol.Detections))
        ]           

        if not label_fields:
            print("No detection-type label fields found.")
            exit(1)
        print("\n")
        print("Available detection label fields:")
        for i, name in enumerate(label_fields):
            print(f"[{i}] {name}")
        EVAL_FIELD_AS_TARGET = choose_required_field(
            "Select the number of the field serving as the target for evaluating: "
        )
        EVAL_FIELD_AS_GROUND_TRUTH = choose_required_field(
            "Select the number of the field serving as the ground truth for evaluating: "
        )
        eval_key = f'eval_{EVAL_FIELD_AS_TARGET}_{timestamp}'
        if EVAL_FIELD_AS_GROUND_TRUTH in dataset_cl.get_field_schema():
            if EVAL_FIELD_AS_TARGET in dataset_cl.get_field_schema():        
                eval_res = dataset_cl.evaluate_detections(
                    EVAL_FIELD_AS_TARGET, EVAL_FIELD_AS_GROUND_TRUTH, eval_key=eval_key, iou=0.5, compute_mAP=True
                )
        
        if args.export_bad_case:
            fn_view = dataset_cl.filter_labels(EVAL_FIELD_AS_GROUND_TRUTH, F(f'{eval_key}') == 'fn')
            fp_view = dataset_cl.filter_labels(EVAL_FIELD_AS_TARGET, (F(f'{eval_key}') == 'fp') & (F('label') == str(FP_LABEL_4_EXPORTING)))
            print(f'len of fn_view: {len(fn_view)}')
            print(f'len of fp_view: {len(fp_view)}')
            merged = fo.Dataset(name=f"merged_dataset_{timestamp}")

            merged.merge_samples(fn_view)
            merged.merge_samples(fp_view)

            print(f'len of bad_case_view: {len(merged)}')
            export_dir = f'fiftyone_export/bad_case/{dataset_cl.name}_{timestamp}'
            exporter = YOLOv5DatasetExporter(export_dir, classes=class_names)
            print(f"exporting the field {EVAL_FIELD_AS_GROUND_TRUTH} to {export_dir}")
            merged.export(dataset_exporter=exporter,
                        dataset_type=fo.types.YOLOv5Dataset,
                        label_field=EVAL_FIELD_AS_GROUND_TRUTH)  # export ground truth labels
    
    if args.upload:
        headers = {"X-Organization": "BVG"}
        options = {
            "label_field": f"new_field_{timestamp}",
            "classes": class_names,
            "label_type": "detections",
            "backend": "cvat",
            "launch_editor": False,
            "task_size":1000,
            "segment_size" : 500,
            "task_name": f'{dataset_cl.name}_task',
            "project_name": CVAT_DEST_PROJECT,  
            "headers": headers,
        }
        annotation_key = f'anno_{dataset_cl.name}_{timestamp}'.replace('/', '-').replace('-', '_')
        results = dataset_cl.annotate(anno_key=annotation_key, **options)
        results.print_status()

    if args.prelabelupload:
        model_name = model_name_list[0]
        headers = {"X-Organization": "BVG"}

        schema = dataset_cl.get_field_schema()
        label_fields = [
            name
            for name, field in schema.items()
            if isinstance(field, fof.EmbeddedDocumentField)
            and issubclass(field.document_type, (fol.Detection, fol.Detections))
        ]           

        if not label_fields:
            print("No detection-type label fields found.")
            exit(1)

        print("\n")
        print("Available detection label fields:")
        for i, name in enumerate(label_fields):
            print(f"[{i}] {name}")
        
        TARGET_FIELD_4_PRELABEL_UPLOADING = choose_required_field(
            "Select the number for the field to upload (TARGET): "
        )
        options = {
            "label_field": TARGET_FIELD_4_PRELABEL_UPLOADING,
            "classes": class_names,
            "label_type": "detections",
            "backend": "cvat",
            "launch_editor": False,
            "task_size":1000,
            "segment_size" : 500,
            "task_name": f'{dataset_cl.name}_task',
            "project_name": CVAT_DEST_PROJECT,  # optional: assign to existing or new CVAT project
            "headers": headers,
        }
        annotation_key = f'anno_{dataset_cl.name}_{timestamp}'.replace('/', '-').replace('-', '_')
        results = dataset_cl.annotate(anno_key=annotation_key, **options)
        results.print_status()

    if args.filteredupload:
        headers = {"X-Organization": "BVG"}
        schema = dataset_cl.get_field_schema()
        label_fields = [
            name
            for name, field in schema.items()
            if isinstance(field, fof.EmbeddedDocumentField)
            and issubclass(field.document_type, (fol.Detection, fol.Detections))
        ]           

        if not label_fields:
            print("No detection-type label fields found.")
            exit(1)

        print("\n")
        print("Available detection label fields:")
        for i, name in enumerate(label_fields):
            print(f"[{i}] {name}")

        TARGET_FIELD_4_FILTERED_UPLOADING = choose_required_field(
            "Select the number for the field to upload (TARGET): "
        )
        print(f"[{len(label_fields)}] None (skip ground truth filtering)")
        GT_FIELD_4_FILTERED_UPLOADING = choose_optional_field(
            "Select the number for the field to use as ground truth (GT)"
        )
        print("\nFinal selection:")
        print(f"TARGET_FIELD_4_FILTERED_UPLOADING: {TARGET_FIELD_4_FILTERED_UPLOADING}")
        print(f"GT_FIELD_4_FILTERED_UPLOADING: {GT_FIELD_4_FILTERED_UPLOADING or 'None'}\n")

        options = {
            "backend": "cvat",
            "label_field": MERGED_FIELD,
            "label_type": "detections",
            "classes": class_names,
            "launch_editor": False,
            "task_size":1000,
            "segment_size" : 500,
            "task_name": f'{dataset_cl.name}_task',
            "project_name": CVAT_DEST_PROJECT,  
            "headers": headers,
        }
        annotation_key = f'anno_{dataset_cl.name}_{timestamp}'.replace('/', '-').replace('-', '_')
        dataset_cl = filter_detections_for_uploading(dataset_cl)
        results = dataset_cl.annotate(anno_key=annotation_key, **options)
        results.print_status()


        # dataset_cl.load_annotations(annotation_key)

        # # Cleanup

        # # Delete tasks from CVAT
        # results = dataset.load_annotation_results(annotation_key)
        # results.cleanup()

        # # Delete run record (not the labels) from FiftyOne
        # dataset.delete_annotation_run(annotation_key)
    





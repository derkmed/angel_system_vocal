from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
from tokenize import Double
import numpy as np
import PIL.Image
import os
import tqdm
import pickle
import pandas as pd
import logging
import re
import gc

gc.collect()

from angel_system.impls.detect_activities.swinb.swinb_detect_activities import SwinBTransformer
from angel_system.eval.support_functions import time_from_name
from angel_system.eval.visualization import EvalVisualization, plot_activities_confidence
from angel_system.eval.compute_scores import EvalMetrics


logging.basicConfig(level = logging.INFO)
log = logging.getLogger("ptg_eval")


def run_eval(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # Load truth annotations
    # ============================
    #gt_label_to_ts_ranges = defaultdict(list)
    gt_f = pd.read_feather(args.activity_gt)
    # Keys: class, start_frame,  end_frame, exploded_ros_bag_path

    gt = []
    RE_FILENAME_TIME = re.compile(r"frame_(\d+)_\d+_\d+.\w+")
    for i, row in gt_f.iterrows():
        g = {
            'class': row["class"].lower().strip(), 
            #'start': float(RE_FILENAME_TIME.match(row["start_frame"]).groups()[0]),
            'start': time_from_name(row["start_frame"]),
            #'end': float(RE_FILENAME_TIME.match(row["end_frame"]).groups()[0])
            'end': time_from_name(row["end_frame"])
        }
        gt.append(g)

    log.info(f"Loaded ground truth from {args.activity_gt}")
    gt = pd.DataFrame(gt)
    for i, row in gt.iterrows():
        print(row)

    # ============================
    # Load detections from
    # extracted ros bag
    # ============================
    detections_input = [det for det in (literal_eval(s) for s in open(args.extracted_activity_detections))][0]
    detections = []

    for dets in detections_input:
        good_dets = {}
        for l, conf in zip(dets["label_vec"], dets["conf_vec"]):
            good_dets[l] = conf

        for l in dets["label_vec"]:
            d = {
                'class': l.lower().strip(), 
                'start': dets["source_stamp_start_frame"], 
                'end': dets["source_stamp_end_frame"],
                'conf': good_dets[l],
                'detect_intersection': np.nan
            }
            detections.append(d)
    detections = pd.DataFrame(detections)
    log.info(f"Loaded detections from {args.extracted_activity_detections}")

    # ============================
    # Load labels
    # ============================
    # Grab all labels based on gt file version
    label_re = re.compile(r'labels_test_(?P<version>v\d+\.\d+)(?P<class>(\w+)?).feather')
    label_ver = label_re.match(os.path.basename(args.activity_gt)).group('version')

    vlabels = pd.read_excel(args.labels, sheet_name=label_ver)
    print(vlabels)
    vlabels['class'] = vlabels['class'].str.lower()
    
    # grab all labels present in data
    dlabels = list(set([l.lower().strip().rstrip('.') for l in gt['class'].unique()] + [l.lower().strip().rstrip('.') for l in detections['class'].unique()]))
    print(dlabels)
    # Remove any labels that we don't actually have
    missing_labels = []
    for i, row in vlabels.iterrows():
        if row['class'] not in dlabels:
            missing_labels.append(i)
    labels = vlabels.drop(missing_labels)

    log.debug(f"Labels v{label_ver}: {labels}")

    # ============================
    # Split by time window
    # ============================
    # Get time ranges
    min_start_time = min(gt['start'].min(), detections['start'].min())
    max_end_time = max(gt['end'].max(), detections['end'].max())
    time_windows = np.arange(min_start_time, max_end_time, args.time_window)

    if time_windows[-1] < max_end_time:
        time_windows = np.append(time_windows, time_windows[-1] + args.time_window)
    time_windows = list(zip(time_windows[:-1], time_windows[1:]))

    # Create masked matrix of detections
    dets_per_time_w = np.full((len(time_windows), len(labels)), None)
    gt_true_pos_mask = np.full((len(time_windows), len(labels)), None)
    uncertain_pad = 1
    time_idx = 0
    print(labels)
    for i, row in labels.iterrows():
        print(row)

    for time in time_windows:
        print('#############################', time)
 
        # Determine what detections we have
        det_overlap = detections.query(f'not (end < {time[0]} or {time[1]} < start)')
        if det_overlap.empty:
            print('empty det')
            time_idx += 1
            continue

        # Determine the highest conf for each class
        for i, row in labels.iterrows():
            class_overlap = det_overlap.loc[det_overlap['class'] == row['class']]
            dets_per_time_w[time_idx][row['id']] = class_overlap['conf'].max()

        # ground truth mask
        gt_overlap = gt.query(f'not (end < {time[0]} or {time[1]} < start)')
        if gt_overlap.empty:
            print('empty gt')
            for i in range(len(labels)):
                gt_true_pos_mask[time_idx][i] = False
            time_idx += 1
            continue

        for i, row in gt_overlap.iterrows():
            correct_label = row['class'].strip().rstrip('.')
            print(correct_label)
            correct_class_idx = labels.loc[labels['class'] == correct_label].iloc[0]['id']
            print(correct_class_idx)

            # if there is overlap, make sure we want to count it first
            """
            u_window_left = [row['start'] - uncertain_pad,  row['start'] + uncertain_pad]
            u_window_right = [row['end'] - uncertain_pad,  row['end'] + uncertain_pad]

            in_left_window = (u_window_left[0] <= time[0] <= u_window_left[1]) and \
                            (u_window_left[0] <= time[1] <= u_window_left[1])

            in_right_window = (u_window_right[0] <= time[0] <= u_window_right[1]) and \
                            (u_window_right[0] <= time[1] <= u_window_right[1])

            invalid = in_left_window or in_right_window
            if invalid:
                gt_true_pos_mask[time_idx][correct_class_idx] = None
            """
            if True:
                # mark the correct class as a true positive
                for i in range(len(labels)):
                    if i == correct_class_idx:
                        gt_true_pos_mask[time_idx][i] = True
                    else: 
                        gt_true_pos_mask[time_idx][i] = False
        time_idx += 1
    
    with open('debug-mat.txt', "w") as f:
        i = 0
        for row in dets_per_time_w:
            f.write(f"{time_windows[i]}: {row}, {gt_true_pos_mask[i]}\n")
            i+=1

    plot_activities_confidence(labels=labels, gt=gt, dets=detections, output_dir=output_dir)
    del gt
    del detections

    gc.collect()
    
    # ============================
    # Metrics
    # ============================
    metrics =  EvalMetrics(labels=labels, gt_true_pos_mask=gt_true_pos_mask, dets_per_time_w=dets_per_time_w, output_fn=f"{output_dir}/metrics.txt")
    #metrics.detect_intersection_per_activity_label()
    metrics.precision_recall_f1()

    log.info(f"Saved metrics to {output_dir}/metrics.txt")
    
    # ============================
    # Plot
    # ============================
    vis = EvalVisualization(labels=labels, gt_true_pos_mask=gt_true_pos_mask, dets_per_time_w=dets_per_time_w, output_dir=output_dir)
    vis.plot_pr_curve()
    #vis.plot_roc_curve()
    #vis.plot_confusion_matrix()

    log.info(f"Saved plots to {output_dir}/plots/")

    del gt_true_pos_mask
    del dets_per_time_w

    gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        default="model_files/activity_detector_annotation_labels.xlsx",
        help="Multi-sheet Excel file of class ids and labels where each sheet is titled after the label version. \
            The sheet used for evaluation is specified by the version number in the ground truth filename"
    )
    parser.add_argument(
        "--activity_gt",
        type=str,
        default="data/ros_bags/Annotated_folding_filter/labels_test_v1.2.feather",
        help="Feather file containing the ground truth annotations in the PTG-LEARN format. \
            The expected filename format is \'labels_test_v<label version>.feather\'"
    )
    parser.add_argument(
        "--extracted_activity_detections",
        type=str,
        default="data/ros_bags/_extracted/activity_detection_data.json",
        help="Text file containing the activity detections from an extracted ROS2 bag"
    )
    parser.add_argument(
        "--time_window",
        type=float,
        default=1,
        help="Time window in seconds to evaluate results on."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval",
        help="Folder to save results and plots to"
    )

    args = parser.parse_args()
    run_eval(args)

if __name__ == '__main__':
    main()

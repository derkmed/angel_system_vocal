from pathlib import Path
from threading import Event, Lock, Thread
from typing import Union

from cv_bridge import CvBridge
import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node, ParameterDescriptor, Parameter
from sensor_msgs.msg import Image

# from yolov7.detect_ptg import load_model, predict_image, predict_hands
from tcn_hpl.data.utils.pose_generation.generate_pose_data import predict_single
# from yolov7.models.experimental import attempt_load
# import yolov7.models.yolo
# from yolov7.utils.torch_utils import TracedModel

from angel_system.utils.event import WaitAndClearEvent
from angel_system.utils.simple_timer import SimpleTimer

from angel_msgs.msg import ObjectDetection2dSet #, JointKeypoints
from angel_utils import declare_and_get_parameters, RateTracker#, DYNAMIC_TYPE
from angel_utils import make_default_main

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from tcn_hpl.data.utils.pose_generation.predictor import VisualizationDemo
import argparse


BRIDGE = CvBridge()

# def get_parser(config_file):
#     parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
#     parser.add_argument(
#         "--config-file",
#         default=config_file,
#         metavar="FILE",
#         help="path to config file",
#     )
#     parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
#     parser.add_argument("--video-input", help="Path to video file."
#                         # , default='/shared/niudt/detectron2/images/Videos/k2/4.MP4'
#     )
#     parser.add_argument(
#         "--input",
#         nargs="+",
#         help="A list of space separated input images; "
#         "or a single glob pattern such as 'directory/*.jpg'"
#         # , default= ['/shared/niudt/DATASET/Medical/Maydemo/2023-4-25/selected_videos/new/M2-16/*.jpg'] # please change here to the path where you put the images
#     )
#     # parser.add_argument(
#     #     "--output",
#     #     help="A file or directory to save output visualizations. "
#     #     "If not given, will show output in an OpenCV window."
#     #     , default='./bbox_detection_results'
#     # )

#     parser.add_argument(
#         "--confidence-threshold",
#         type=float,
#         default=0.8,
#         help="Minimum score for instance predictions to be shown",
#     )
#     parser.add_argument(
#         "--opts",
#         help="Modify config options using the command-line 'KEY VALUE' pairs",
#         default=[],
#         nargs=argparse.REMAINDER,
#     )
#     return parser


def setup_detectron_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.8
    cfg.freeze()
    return cfg

class PoseEstimator(Node):
    """
    ROS node that runs the yolov7 object detector model and outputs
    `ObjectDetection2dSet` messages.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        
        print("getting logger")
        log = self.get_logger()

        # Inputs
        print("getting params")
        param_values = declare_and_get_parameters(
            self,
            [
                ##################################
                # Required parameter (no defaults)
                ("image_topic",),
                ("det_topic",),
                ("det_net_checkpoint",),
                ("pose_net_checkpoint",),
                ("det_config",),
                ("pose_config",),
                ##################################
                # Defaulted parameters
                ("inference_img_size", 1280),  # inference size (pixels)
                ("det_conf_threshold", 0.7),  # object confidence threshold
                ("iou_threshold", 0.45),  # IOU threshold for NMS
                ("cuda_device_id", 0),  # cuda device: ID int or CPU
                ("no_trace", True),  # don`t trace model
                ("agnostic_nms", False),  # class-agnostic NMS
                # Runtime thread checkin heartbeat interval in seconds.
                ("rt_thread_heartbeat", 0.1),
                # If we should enable additional logging to the info level
                # about when we receive and process data.
                ("enable_time_trace_logging", False),
            ],
        )
        self._image_topic = param_values["image_topic"]
        self._det_topic = param_values["det_topic"]
        self.det_model_ckpt_fp = Path(param_values["det_net_checkpoint"])
        self.pose_model_ckpt_fp = Path(param_values["pose_net_checkpoint"])
        self.det_config = Path(param_values["det_config"])
        self.pose_config = Path(param_values["pose_config"])
        
        self._inference_img_size = param_values["inference_img_size"]
        self._det_conf_thresh = param_values["det_conf_threshold"]
        self._iou_thr = param_values["iou_threshold"]
        self._cuda_device_id = param_values["cuda_device_id"]
        self._no_trace = param_values["no_trace"]
        self._agnostic_nms = param_values["agnostic_nms"]
        
        print("finished setting params")
        
        print("loading detectron model")
        # Detectron Model
        # self.args = get_parser(self.det_config).parse_args()
        # print(self.args)
        detecron_cfg = setup_detectron_cfg(self.det_config)
        self.det_model = VisualizationDemo(detecron_cfg)
        
        
        print("loading pose model")
        # Pose model
        self.pose_model = init_pose_model(self.pose_config, 
                                        self.pose_model_ckpt_fp, 
                                        device=self._cuda_device_id)


        self._enable_trace_logging = param_values["enable_time_trace_logging"]


        # Single slot for latest image message to process detection over.
        self._cur_image_msg: Image = None
        self._cur_image_msg_lock = Lock()

        print("creating subscription to image topic")
        # Initialize ROS hooks
        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        
        print("creating publisher to detections")
        self._det_publisher = self.create_publisher(
            ObjectDetection2dSet,
            self._det_topic,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        
        # self._pose_publisher = self.create_publisher(
        #     JointKeypoints,
        #     self._det_topic,
        #     1,
        #     callback_group=MutuallyExclusiveCallbackGroup(),
        # )

        # if not self._no_trace:
        #     self.model = TracedModel(self.model, self.device, self._inference_img_size)

        self.half = half = (
            self.device.type != "cpu"
        )  # half precision only supported on CUDA
        if half:
            self.model.half()  # to FP16

        self._rate_tracker = RateTracker()
        log.info("Detector initialized")

        # Create and start detection runtime thread and loop.
        log.info("Starting runtime thread...")
        # On/Off Switch for runtime loop
        self._rt_active = Event()
        self._rt_active.set()
        # seconds to occasionally time out of the wait condition for the loop
        # to check if it is supposed to still be alive.
        self._rt_active_heartbeat = param_values["rt_thread_heartbeat"]
        # Condition that the runtime should perform processing
        self._rt_awake_evt = WaitAndClearEvent()
        self._rt_thread = Thread(target=self.rt_loop, name="prediction_runtime")
        self._rt_thread.daemon = True
        self._rt_thread.start()

    def listener_callback(self, image: Image):
        """
        Callback function for image messages. Runs the berkeley object detector
        on the image and publishes an ObjectDetectionSet2d message for the image.
        """
        log = self.get_logger()
        if self._enable_trace_logging:
            log.info(f"Received image with TS: {image.header.stamp}")
        with self._cur_image_msg_lock:
            self._cur_image_msg = image
            self._rt_awake_evt.set()

    def rt_alive(self) -> bool:
        """
        Check that the prediction runtime is still alive and raise an exception
        if it is not.
        """
        alive = self._rt_thread.is_alive()
        if not alive:
            self.get_logger().warn("Runtime thread no longer alive.")
            self._rt_thread.join()
        return alive

    def rt_stop(self) -> None:
        """
        Indicate that the runtime loop should cease.
        """
        self._rt_active.clear()

    def rt_loop(self):
        log = self.get_logger()
        log.info("Runtime loop starting")
        enable_trace_logging = self._enable_trace_logging

        while self._rt_active.wait(0):  # will quickly return false if cleared.
            if self._rt_awake_evt.wait_and_clear(self._rt_active_heartbeat):
                with self._cur_image_msg_lock:
                    if self._cur_image_msg is None:
                        continue
                    image = self._cur_image_msg
                    self._cur_image_msg = None

                if enable_trace_logging:
                    log.info(f"[rt-loop] Processing image TS={image.header.stamp}")
                # Convert ROS img msg to CV2 image
                img0 = BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8")
                
                
                # print()
                
                print(f"img0: {img0.shape}")
                width, height = self._inference_img_size

                det_msg = ObjectDetection2dSet()
                det_msg.header.stamp = self.get_clock().now().to_msg()
                det_msg.header.frame_id = image.header.frame_id
                det_msg.source_stamp = image.header.stamp
                det_msg.label_vec[:] = self.model.names
                
                # pose_msg = JointKeypoints()
                # pose_msg.header.stamp = self.get_clock().now().to_msg()
                # pose_msg.header.frame_id = image.header.frame_id
                # pose_msg.source_stamp = image.header.stamp
                # pose_msg.label_vec[:] = self.model.names

                boxes, labels, keypoints = predict_single(det_model=self.det_model,
                                                          pose_model=self.pose_model,
                                                          image=img0)
                
                # at most, we have 1 set of keypoints for 1 patient
                # for keypoints_ in keypoints:
                #     pose_msg.keypoints = keypoints_
                #     self._pose_publisher.publish(pose_msg)
                
                n_dets = 0
                for xyxy, labels in zip(boxes, labels):
                    n_dets += 1
                    det_msg.left.append(xyxy[0])
                    det_msg.top.append(xyxy[1])
                    det_msg.right.append(xyxy[2])
                    det_msg.bottom.append(xyxy[3])

                    # dflt_conf_vec[cls_id] = conf
                    # copies data into array
                    det_msg.label_confidences.extend(1.0)
                    # reset before next passthrough
                    # dflt_conf_vec[cls_id] = 0.0

                det_msg.num_detections = n_dets

                self._det_publisher.publish(det_msg)

                self._rate_tracker.tick()
                log.info(
                    f"Objects Detection Rate: {self._rate_tracker.get_rate_avg()} Hz",
                )

    def destroy_node(self):
        print("Stopping runtime")
        self.rt_stop()
        print("Shutting down runtime thread...")
        self._rt_active.clear()  # make RT active flag "False"
        self._rt_thread.join()
        print("Shutting down runtime thread... Done")
        super().destroy_node()


# Don't really want to use *all* available threads...
# 3 threads because:
# - 1 known subscriber which has their own group
# - 1 for default group
# - 1 for publishers
print("executing make_default_main")
main = make_default_main(PoseEstimator, multithreaded_executor=3)


if __name__ == "__main__":
    print("executing main")
    # node = PoseEstimator()
    print(f"before main: {node}")
    main()

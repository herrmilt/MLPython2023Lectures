from detectors.openvino_generic_detector import OpenvinoGenericDetector
from pipelines.camera_tracker_processor import CameraTrackerProcessor
from pipelines.common_tools import DictWrapper
from pipelines.detection_sync_processor import GetDetectionProcessor
from pipelines.image_drawers_processors import PaintFrameId, PaintTrackers, CopyImageProcessor, PaintDetectionsProcessor
from pipelines.pipeline import Pipeline
from pipelines.simple_object_tracker import SimpleObjectTracker
from pipelines.sinks import WaitForKeySink, ShowInWindowsSink
from pipelines.sources import OpencvVideoSource
from detectors.yolo8_detector import Yolo8Detector, PaintYoloDetectionProcessor


def main():
    detector = Yolo8Detector("", "resources/yolov8n-seg.pt",
                                       0.5, 0.5,
                                       DictWrapper({
                                          "classes": []
                                       }))

    # video_file_name = 'videos/funny.mp4'
    video_file_name = 0

    # output_video_name = video_file_name + "_out.mp4"
    start_frame_idx = 0

    pipeline = Pipeline(
        source=OpencvVideoSource(video_file_name, start_frame_idx),
        processors=[
            GetDetectionProcessor(detector),

            CopyImageProcessor('image', 'image_detections'),
            PaintYoloDetectionProcessor('image_detections', show_class=True,
                                        show_mask=True, show_keypoints=False),
            PaintFrameId('image_tracklets')
        ],
        sinks=[
            ShowInWindowsSink('image_detections', "detections"),
            WaitForKeySink(1)
        ]
    )
    pipeline.run_while_source()


main()

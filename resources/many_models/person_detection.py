from detectors.openvino_generic_detector import OpenvinoGenericDetector
from pipelines.camera_tracker_processor import CameraTrackerProcessor
from pipelines.detection_sync_processor import GetDetectionProcessor
from pipelines.image_drawers_processors import PaintFrameId, PaintTrackers, CopyImageProcessor, PaintDetectionsProcessor
from pipelines.pipeline import Pipeline
from pipelines.simple_object_tracker import SimpleObjectTracker
from pipelines.sinks import WaitForKeySink, ShowInWindowsSink
from pipelines.sources import OpencvVideoSource
from detectors.yolo_nass_onnx_detector import YoloNASObjectDetector, DictWrapper


def main():
    detector = OpenvinoGenericDetector("resources/person-detection-0203.xml", "resources/person-detection-0203.bin",
                                       0.25, 0.5,
                                       DictWrapper({
                                           "device": "CPU", "architecture_type": "ssd", "num_streams": "",
                                           "num_threads": None,
                                           "labels": [], "keep_aspect_ratio": True, "max_requests": 0,
                                           "max_pendant_count_per_key": 10
                                       }))

    # video_file_name = "videos/WillfullFalls.webm"
    # video_file_name = "videos/futbol_kids.mp4"
    # video_file_name = "videos/soccer.mp4"
    video_file_name = 0

    # output_video_name = video_file_name + "_out.mp4"
    start_frame_idx = 0

    pipeline = Pipeline(
        source=OpencvVideoSource(video_file_name, start_frame_idx),
        processors=[
            GetDetectionProcessor(detector),

            CopyImageProcessor('image', 'image_detections'),
            PaintDetectionsProcessor('image_detections'),
            PaintFrameId('image_tracklets')
        ],
        sinks=[
            ShowInWindowsSink('image_detections', "detections"),
            WaitForKeySink(1)
        ]
    )
    pipeline.run_while_source()


main()

import cv2

from pipelines.camera_tracker_processor import CameraTrackerProcessor
from pipelines.common_processors import WaitIfGpuHot
from pipelines.detection_sync_processor import GetDetectionProcessor
from pipelines.image_drawers_processors import PaintDetectionsProcessor, CopyImageCamerasProcessor, \
    PaintTrackersInCamera, PaintFrameId
from pipelines.pipeline import Pipeline
from pipelines.simple_object_tracker import SimpleObjectTracker
from pipelines.sinks import WaitForKeySink, ShowCamerasInWindowsSink, FpsCalculator, WriteCamerasToVideoSink
from pipelines.sources import VideoBasedSource
from yolo_nass_onnx_detector import YoloNASObjectDetector, DictWrapper


def main(write_result=False):
    detector = YoloNASObjectDetector("", "resources/ants_nas.onnx", 0.1, 0.5,
                                     DictWrapper({
                                         "device": "GPU",
                                         "classes": {0}
                                     }))
    tracker = SimpleObjectTracker(0.7)
    tracker.min_conf_first_object = 0.9
    video_file_name = "/mnt/hdd/__Docencia/DataAnalysisWithPython/!!2023SepUH/lectures/resources/label_video_train_detector/hormigas.mp4"
    # output_video_name = "/mnt/hdd/__Docencia/DataAnalysisWithPython/!!2023SepUH/lectures/resources/label_video_train_detector/hormigas_track{}.avi"
    start_frame_idx = 0

    pipeline = Pipeline(
        source=VideoBasedSource(((video_file_name, "Chan1"),), start_frame_idx),
        processors=[
            # WaitIfGpuHot(79, 3, 30),
            GetDetectionProcessor(detector),
            CameraTrackerProcessor(tracker, "Chan1"),

            CopyImageCamerasProcessor('image', 'image_detections'),
            CopyImageCamerasProcessor('image', 'image_tracklets'),
            PaintDetectionsProcessor('image_detections'),
            PaintTrackersInCamera('image_tracklets'),
            PaintFrameId('image_tracklets')
        ],
        sinks=[
            FpsCalculator(),
            ShowCamerasInWindowsSink('image_tracklets', "tracklets"),
            ShowCamerasInWindowsSink('image_detections', "detections"),
            # WriteCamerasToVideoSink("image_tracklets", output_video_name, 30),
            WaitForKeySink(0)
        ]
    )
    pipeline.run_k(500)


main()

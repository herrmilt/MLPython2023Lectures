from pipelines.camera_tracker_processor import CameraTrackerProcessor
from pipelines.detection_sync_processor import GetDetectionProcessor
from pipelines.image_drawers_processors import PaintFrameId, PaintTrackers, CopyImageProcessor, PaintDetectionsProcessor
from pipelines.pipeline import Pipeline
from pipelines.simple_object_tracker import SimpleObjectTracker
from pipelines.sinks import WaitForKeySink, ShowInWindowsSink
from pipelines.sources import OpencvVideoSource
from detectors.yolo_nass_onnx_detector import YoloNASObjectDetector, DictWrapper


def main():
    detector = YoloNASObjectDetector("", "resources/ants_nas.onnx", 0.1, 0.5,
                                     DictWrapper({
                                         "device": "GPU",
                                         "classes": {0}
                                     }))
    tracker = SimpleObjectTracker(0.7)
    tracker.min_conf_first_object = 0.9
    video_file_name = "videos/hormigas.mp4"
    # output_video_name = "/mnt/hdd/__Docencia/DataAnalysisWithPython/!!2023SepUH/lectures/resources/label_video_train_detector/hormigas_track{}.avi"
    start_frame_idx = 0

    pipeline = Pipeline(
        source=OpencvVideoSource(video_file_name, start_frame_idx),
        processors=[
            # WaitIfGpuHot(79, 3, 30),
            GetDetectionProcessor(detector),
            CameraTrackerProcessor(tracker),

            CopyImageProcessor('image', 'image_tracklets'),
            CopyImageProcessor('image', 'image_detections'),
            PaintTrackers('image_tracklets'),
            PaintDetectionsProcessor('image_detections'),
            PaintFrameId('image_tracklets')
        ],
        sinks=[
            # FpsCalculator(),
            ShowInWindowsSink('image_tracklets', "tracklets"),
            ShowInWindowsSink('image_detections', "detections"),
            # WriteImageToVideoSink("image_tracklets", output_video_name, 30),
            WaitForKeySink(1)
        ]
    )
    pipeline.run_k(500)


main()

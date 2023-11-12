import cv2
from openvino.inference_engine import IECore
import openvino_tools.models as models
from pathlib import Path


class OpenvinoGenericDetector:

    existing_models = {}
    ie = None
    plugin_config = None

    def __init__(self, model_name, weights_name, min_conf, nms_threshold, args):
        if OpenvinoGenericDetector.ie is None:
            OpenvinoGenericDetector.ie = IECore()
            OpenvinoGenericDetector.plugin_config = self.get_plugin_configs(args.device, args.num_streams, args.num_threads)
        model_specs = OpenvinoGenericDetector.existing_models.get(model_name, None)
        if model_specs is None:
            args.model = Path(model_name)
            model = self.get_model(OpenvinoGenericDetector.ie, args)
            exec_net = OpenvinoGenericDetector.ie.load_network(network=model.net, device_name=args.device,
                                            config=OpenvinoGenericDetector.plugin_config, num_requests=0)
            OpenvinoGenericDetector.existing_models[model_name] = {
                "model": model,
                "exec_net": exec_net,
            }
            self.model = model
            self.exec_net = exec_net
        else:
            self.model = model_specs['model']
            self.exec_net = model_specs['exec_net']
        self.min_conf = min_conf
        self.selector = lambda box, conf, klass: True

    def detect(self, img):
        result = []
        inputs, preprocessing_meta = self.model.preprocess(img)
        net_output = self.exec_net.infer(inputs)
        result_raw = self.model.postprocess(net_output, preprocessing_meta)
        for v in result_raw:
            if v.score < self.min_conf:
                break
                # bbox = detection[3:7] * np.array([self.width, self.height, self.width, self.height])
                # bboxes.append(bbox.astype("int"))
            bbox = [int(v.xmin), int(v.ymin), int(v.xmax - v.xmin) + 1, int(v.ymax - v.ymin) + 1]
            if self.selector(bbox, v.score, v.id):
                result.append({
                    'box': bbox,
                    'conf': v.score,
                    'class': int(v.id)
                })
        return result

    def draw_bboxes(self, image, bboxes, confidences, class_ids):
        """
        Draw the bounding boxes about detected objects in the image.

        Args:
            image (numpy.ndarray): Image or video frame.
            bboxes (numpy.ndarray): Bounding boxes pixel coordinates as (xmin, ymin, width, height)
            confidences (numpy.ndarray): Detection confidence or detection probability.
            class_ids (numpy.ndarray): Array containing class ids (aka label ids) of each detected object.

        Returns:
            numpy.ndarray: image with the bounding boxes drawn on it.
        """

        for bb, conf, cid in zip(bboxes, confidences, class_ids):
            clr = (255, 0, 0)
            cv2.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
            label = "{}:{:.4f}".format("Person", conf)
            (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(bb[1], label_height)
            cv2.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
                         (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (bb[0], y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        return image

    def get_model(self, ie, args):
        input_transform = models.InputTransform(False, None, None)
        if args.architecture_type == 'ssd':
            return models.SSD(ie, args.model, input_transform, labels=args.labels, keep_aspect_ratio_resize=args.keep_aspect_ratio)
        if args.architecture_type == 'ssd_annette':
            return models.SSDAnnette(ie, args.model, input_transform, labels=args.labels, keep_aspect_ratio_resize=args.keep_aspect_ratio)
        elif args.architecture_type == 'ctpn':
            return models.CTPN(ie, args.model, input_size=args.input_size, threshold=args.prob_threshold)
        elif args.architecture_type == 'yolo':
            return models.YOLO(ie, args.model, labels=args.labels,
                               threshold=args.prob_threshold, keep_aspect_ratio=args.keep_aspect_ratio)
        elif args.architecture_type == 'yolov4':
            return models.YoloV4(ie, args.model, labels=args.labels,
                                 threshold=args.prob_threshold, keep_aspect_ratio=args.keep_aspect_ratio)
        elif args.architecture_type == 'faceboxes':
            return models.FaceBoxes(ie, args.model, threshold=args.prob_threshold)
        elif args.architecture_type == 'centernet':
            return models.CenterNet(ie, args.model, labels=args.labels, threshold=args.prob_threshold)
        elif args.architecture_type == 'retinaface':
            return models.RetinaFace(ie, args.model, threshold=args.prob_threshold)
        else:
            raise RuntimeError('No model type or invalid model type (-at) provided: {}'.format(args.architecture_type))

    def get_plugin_configs(self, device, num_streams, num_threads):
        config_user_specified = {}

        devices_nstreams = {}
        if num_streams:
            devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
                if num_streams.isdigit() \
                else dict(device.split(':', 1) for device in num_streams.split(','))

        if 'CPU' in device:
            if num_threads is not None:
                config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
            if 'CPU' in devices_nstreams:
                config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                    if int(devices_nstreams['CPU']) > 0 \
                    else 'CPU_THROUGHPUT_AUTO'

        if 'GPU' in device:
            if 'GPU' in devices_nstreams:
                config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                    if int(devices_nstreams['GPU']) > 0 \
                    else 'GPU_THROUGHPUT_AUTO'

        return config_user_specified



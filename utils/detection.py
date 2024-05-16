from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def init_detector(model_path: str, **kwargs) -> vision.ObjectDetector:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        **kwargs
    )
    detector = vision.ObjectDetector.create_from_options(options)

    return detector

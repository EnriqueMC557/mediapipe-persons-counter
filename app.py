import cv2
import numpy as np
import mediapipe as mp

from utils.visualization import visualize
from utils.detection import init_detector


def main():
    detector = init_detector(
        'models/efficientdet.tflite', 
        score_threshold=0.25
    )

    # Open video capture
    cap = cv2.VideoCapture('data/personas_calle.mp4')

    # Read video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading the frame. Finishing execution.")
            break

        # Image preprocessing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Perform detection
        detection_result = detector.detect(image)
        image_copy = np.copy(image.numpy_view())
        annotated_image = visualize(image_copy, detection_result, categories=['person'], max_elements=5)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Shows the detection result
        cv2.imshow('Persons Counter', rgb_annotated_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

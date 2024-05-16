import cv2
import numpy as np


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 4
FONT_THICKNESS = 5
TEXT_COLOR = (255, 0, 0)  # red


def visualize(image, detection_result, categories: list = None, draw_score=False, max_elements: int = None) -> np.ndarray:
    if max_elements:
        elements_counter = 0
        for detection in detection_result.detections:
            if detection.categories[0].category_name not in categories:
                elements_counter += 1

        if elements_counter < max_elements * 0.70:
            TEXT_COLOR = (0, 255, 0)
            risk_message = f'Detected: {elements_counter} / {max_elements}        Risk: Low'
        elif elements_counter < max_elements * 0.90:
            TEXT_COLOR = (255, 255, 0)
            risk_message = f'Detected: {elements_counter} / {max_elements}        Risk: Medium'
        else:
            TEXT_COLOR = (255, 0, 0)
            risk_message = f'Detected: {elements_counter} / {max_elements}        Risk: High'

        cv2.putText(image, risk_message, (20, 50), cv2.FONT_HERSHEY_PLAIN, 
            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)


    for detection in detection_result.detections:
        # Skip categories that are not of interest
        if categories is not None and detection.categories[0].category_name not in categories:
            continue
        elif max_elements is not None:
            elements_counter += 1
       
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        if draw_score:
            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                            MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

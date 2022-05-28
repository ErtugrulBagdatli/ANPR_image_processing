import cv2
import easyocr
import numpy as np

# Configure easyocr
reader = easyocr.Reader(['en'])


# Debug show function
def show_img(text: str, image):
    """
    :param text:Image window name
    :param image:Required image to display
    :return: None
    """
    pass
    # cv2.imshow(text, image)
    # cv2.waitKey(0)


def detect_plate(image):
    # Resize the image - change width-height 800-600
    image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
    # show_img("Original Image", image)

    # Convert BGR to Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show_img("1 - Grayscale Conversion", gray)

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_channel = lab[:, :, 0]

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge([cl, a, b])

    # Convert image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # show_img("2-Enhanced Image", enhanced_img)

    # Convert Enhanced Image to Grayscale
    enhanced_img_gray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)

    # Contrast recovery using add weight on normal grayscale and enhanced one
    weighted = cv2.addWeighted(gray, 0.8, enhanced_img_gray, 0.2, 10.0)
    # show_img("3 - Weighted", weighted)

    # Noise removal with GaussianBlur(removes noise while preserving edges)
    blurred = cv2.GaussianBlur(weighted, (3, 3), 10, 10)
    # show_img("4 - GaussianBlur Filter", blurred)

    # Find Edges of the grayscale image using Canny filtering
    edged = cv2.Canny(blurred, 10, 100)
    # show_img("5 - Canny Edges", edged)

    # Find contours based on Edges
    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create copy of original image to draw all contours
    img_with_all_contours = image.copy()
    cv2.drawContours(img_with_all_contours, cnts, -1, (0, 255, 0), 3)
    # show_img("6- All Contours", img_with_all_contours)

    # Sort contours based on their area keeping first 30 contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    number_plate_cnt = None  # No Number plate contour

    # Draw Top 30 Contours
    img2 = image.copy()
    cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)
    # show_img("7- Top 30 Contours", img2)

    # Loop over all contours to find the best possible approximate contour of number plate
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:  # Select the contour which has 4 corners
            number_plate_cnt = approx  # Approximated Number Plate Contour

            # Crop selected contour
            x, y, w, h = cv2.boundingRect(c)  # Find out coord for plate

            # if h * 2 < w < h * 8:
            plate = image[y:y + h, x:x + w]  # Crop the number plate from original image
            # show_img("Original", plate)

            # Noise removal filter on cropped plate
            noise_removed = plate - cv2.GaussianBlur(plate, (21, 21), 3) + 127
            # show_img("1- Noise Removed", noise_removed)

            # Resize cropped number plate
            resized_plate = cv2.resize(noise_removed, (int(w * 1.2), int(h * 1.2)),
                                       interpolation=cv2.INTER_AREA)

            # Threshold resized plate with 120 threshold value
            _, threshold_plate = cv2.threshold(np.array(resized_plate), 120, 255,
                                               cv2.THRESH_BINARY)
            # show_img("2- Threshold", threshold_plate)

            # Increase contrast and remove some noise using weighting
            weighted_plate = cv2.addWeighted(resized_plate, 0.6, threshold_plate,
                                             0.4, 10.0)
            # show_img("3-Weighted", weighted_plate)

            # Dilate image to emphasize letters
            dilated_plate = cv2.dilate(weighted_plate, (3, 3))
            # show_img("4- Dilated", dilated_plate)

            break

    # Drawing the selected number plate contour on the original image
    final_image = image.copy()
    cv2.drawContours(final_image, [number_plate_cnt], -1, (0, 255, 0), 3)
    # show_img("Final Image With Number Plate Detected", image)

    # Use EasyOcr to covert image into string
    detected_result = easy_ocr(dilated_plate)
    print("Easy Ocr :", detected_result)

    # Return filtered results to show every unique step
    return image, gray, enhanced_img, weighted, blurred, edged, img_with_all_contours, final_image, detected_result


def easy_ocr(image):
    """
    :param image: Cropped number plate
    :return: Text extracted from given input
    """
    plate_result = reader.readtext(image)
    # There is no detected character
    if len(plate_result) == 0:
        return ""

    result = ""

    for text in plate_result:
        result += text[1]

    return result.upper()

# Alternative OCR library (Tesseract)
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# def ocr_plate(image):
#     return pytesseract.image_to_string(image, lang='eng', timeout=1000)

# Alternative OCR library (Keras OCR)
# import keras_ocr
# def ocr_keras(image):
#     image = np.expand_dims(image, axis=0)
#     prediction_group = pipeline.recognize(image)
#
#     result = ""
#     last_pos = 0.0
#
#     for text in reversed(prediction_group[0]):
#         if text[1][0][0] > last_pos:
#             result += text[0]
#             last_pos = text[1][0][0]
#         else:
#             result = text[0] + result
#             last_pos = text[1][0][0]
#
#     return result.upper()

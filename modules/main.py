import easyocr

reader = easyocr.Reader(['ru'],
                        model_storage_directory='modules/models',
                        user_network_directory='modules/models',
                        recog_network='custom_ocr',
                        detector=False,
                        gpu=False,
                        )

def recognize_digits_ocr(image):
    preds_list = []
    ocr_result = reader.recognize(image)
    conf = str(ocr_result[0][-1])
    digits = list(str(ocr_result[0][-2]))
    print(digits)
    for digit in digits:
        prediction = {
        'x': 0,
        'y': 0,
        'w': 0,
        'h': 0,
        }
        prediction['confidence'] = conf
        prediction['class'] = digit
        prediction['value'] = digit
        preds_list.append(prediction)

    return preds_list
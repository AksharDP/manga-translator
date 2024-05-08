from deep_translator import GoogleTranslator
from paddleocr import PaddleOCR


def get_text(img_path, lang):
    # use_angle_cls=False to disable the angle classifier for faster inference
    ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)
    result = ocr.ocr(img_path, cls=False)
    try:
        return "".join(list(map(lambda x: x[1][0].replace("'", ""), result[0])))
    except TypeError:
        return None


def translate(text):
    return GoogleTranslator(source="auto", target="en").translate(text)

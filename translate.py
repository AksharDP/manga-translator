from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator


def get_text(img_path, lang):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    result = ocr.ocr(img_path, cls=True)
    del ocr
    try:
        return "".join(list(map(lambda x: x[1][0].replace("'", ""), result[0])))
    except TypeError:
        return None


def translate(text):
    return GoogleTranslator(source="auto", target="en").translate(text)

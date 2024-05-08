import configparser
import cv2
import numpy as np
import os
import sys
from segmentation import segmentation
from translate import get_text, translate
import torch
from ultralytics import YOLO
import requests

config = configparser.ConfigParser()
config.read("config.ini")

# paddle ocr config
language = config["ocr"]["language"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

segmentation_model = "https://huggingface.co/AksharPatel/manga-speech-bubble-segmentation/resolve/main/best.pt?download=true"

# text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
thickness = 2


def wrap_text(text, font, font_scale, thickness, max_width):
    """Wrap text to fit within a given width."""
    lines = []
    line = ""
    line_width = 0
    for word in text.split():
        word_width = cv2.getTextSize(word + " ", font, font_scale, thickness)[0][0]
        if line_width + word_width <= max_width:
            line += word + " "
            line_width += word_width
        else:
            lines.append(line.strip())
            line = word + " "
            line_width = word_width
    if line:
        lines.append(line.strip())
    return lines, line_width


def handle_speech_bubbles(results):
    final_image = seg.image.copy()

    for i, r in enumerate(results):
        mask = np.array(r.masks[0].xy, np.int32)
        mask = mask.reshape(-1, 2)

        masked_image_path = f"temp/mask_{i}.jpg"
        if not os.path.exists(masked_image_path):
            mask_image = np.zeros_like(seg.image)
            cv2.fillPoly(mask_image, [mask], (255, 255, 255))
            masked_image = cv2.bitwise_and(seg.image, mask_image)
            cv2.imwrite(masked_image_path, masked_image)

        text = get_text(masked_image_path, language)
        print(f"Extracted text: {text}")
        if text is None:
            continue
        translated_text = translate(text)
        print(f"Translated text: {translated_text}")

        x_min, y_min = mask.min(axis=0)
        x_max, y_max = mask.max(axis=0)
        mask_width = x_max - x_min
        mask_height = y_max - y_min

        wrapped_lines, wrapped_width = wrap_text(
            translated_text, font, font_scale, thickness, mask_width
        )
        print(f"Wrapped text: {wrapped_lines}")

        # Calculate  total height of the wrapped lines
        total_height = 0
        for line in wrapped_lines:
            (line_width, line_height), _ = cv2.getTextSize(
                line, font, font_scale, thickness
            )
            total_height += line_height + thickness

        # Calculate the position to place the text in the center of the bounding box
        bubble_center = (x_min + mask_width // 2, y_min + mask_height // 2)
        text_center = (
            x_min,
            bubble_center[1] - total_height // 2,
        )

        # Create a new image with the same dimensions as the original image but all pixels set to white
        text_image = np.full_like(seg.image, 255)

        # Draw each line of text onto the new image using the mask as a ROI
        for line in wrapped_lines:
            (line_width, line_height), _ = cv2.getTextSize(
                line, font, font_scale, thickness
            )
            text_x = text_center[0] + (mask_width - line_width) // 2
            text_y = text_center[1] + line_height // 2
            cv2.putText(
                text_image,
                line,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )
            text_center = (
                text_center[0],
                text_y + line_height + thickness,
            )

        # Create a mask for the text area using the same coordinates as the speech bubble
        text_mask = np.zeros_like(seg.image)
        cv2.fillPoly(text_mask, [mask], (255, 255, 255))

        # Overlay the new image with the text onto the final image using the original mask
        final_image = cv2.copyTo(text_image, text_mask, final_image)

    # original file name
    file_name = os.path.basename(seg.image_path)
    # Save the final image with all the translated text
    cv2.imwrite(f"output/translated_{file_name}", final_image)
    print(f"translated_{file_name} saved.")
    # Remove the temporary mask images
    for i in range(len(results)):
        os.remove(f"temp/mask_{i}.jpg")


if __name__ == "__main__":

    if not os.path.exists("input"):
        os.makedirs("input")
        print("Input folder created. Put images to be processed in the input folder.")
        sys.exit(1)
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("temp"):
        os.makedirs("temp")
    if not os.path.exists("model"):
        os.makedirs("model")

    if not os.path.exists("model/best.pt"):
        print("Downloading segmentation model...")
        with requests.get(segmentation_model, stream=True) as r:
            with open("model/best.pt", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            r.close()
        print("Segmentation model downloaded.")

    # speech bubble detection model
    model = YOLO("model/best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = 0.3
    verbose = False
    show_labels = False
    show_conf = False
    augment = False

    files = os.listdir("input")
    for file in files:
        seg = segmentation(
            f"input/{file}",
            model,
            device,
            conf,
            verbose,
            show_labels,
            show_conf,
            augment,
        )
        results = seg.perform_detection(seg.image)
        if results[0].masks == None:
            print(f"no masks found for {seg.image}")
            continue
        handle_speech_bubbles(results)
    print("All images processed.")

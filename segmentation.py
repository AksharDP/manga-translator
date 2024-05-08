import cv2
import numpy as np
import os


class segmentation:
    def __init__(
        self,
        image_path,
        model,
        device,
        conf,
        verbose,
        show_labels,
        show_conf,
        augment,
    ):
        self.image_path = image_path
        self.image = self.load_image(image_path)
        self.model = model
        self.device = device
        self.imgz = self.calculate_image_size()
        self.conf = conf
        self.verbose = verbose
        self.show_labels = show_labels
        self.show_conf = show_conf
        self.augment = augment

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def calculate_image_size(self):
        """
        Calculate the best image size for inference
        """
        h, w, c = self.image.shape
        h -= h % 32
        w -= w % 32
        return h, w

    @staticmethod
    def ensure_save_directory_exists():
        save_dir = "save/"
        os.makedirs(save_dir, exist_ok=True)

    def perform_detection(self, image):
        results = self.model(
            image,
            device=self.device,
            imgsz=self.imgz,
            conf=self.conf,
            verbose=self.verbose,
            show_labels=self.show_labels,
            show_conf=self.show_conf,
            augment=self.augment,
        )[0]
        return results

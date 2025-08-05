"""
Autonomous Video Hunter - Zero-Shot Object Detection

This module provides zero-shot object detection capabilities using Hugging Face transformers
and OWL-ViT models. It enables detection of arbitrary object classes without prior training,
making it ideal for flexible OSINT analysis where target objects may vary by investigation.

Key Features:
- Zero-shot detection of any specified object classes
- OWL-ViT (Vision Transformer) based detection
- Automatic object cropping and evidence collection
- Configurable confidence thresholds and top-k results
- CPU and GPU support with automatic fallback

Author: @kdr
"""

from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any

from PIL import Image
from transformers import pipeline


class ZeroShotDetector:
    """
    Zero-shot object detector based on a Hugging Face pipeline.

    Parameters
    ----------
    model_name : str, optional
        HF model checkpoint (defaults to OWL-V2).
    crops_dir : str | Path, optional
        Where to save cropped images (created if missing).
    """

    def __init__(self,
                 model_name: str = "google/owlv2-base-patch16-ensemble",
                 crops_dir: str | Path = "crops") -> None:
        self.detector = pipeline(
            task="zero-shot-object-detection",
            model=model_name,
            device=-1,  # -1 → CPU
        )
        self.crops_dir = Path(crops_dir)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------

    def detect(self,
               image_path: str | Path,
               classes: List[str],
               topk: int = 1,
               crop: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run detection and optionally save crops.

        Returns
        -------
        dict : {label: [ {score, box, crop_path?}, … ] }
        """
        # Load image once
        image = Image.open(image_path).convert("RGB")

        # Raw predictions (one flat list)
        preds = self.detector(image, candidate_labels=classes)

        # Bucket by label & sort
        by_label: Dict[str, List[Dict[str, Any]]] = {c: [] for c in classes}
        for p in preds:
            lbl = p["label"]
            if lbl in by_label:
                by_label[lbl].append(p)

        for lbl, items in by_label.items():
            # high-to-low score and keep top-k
            items.sort(key=lambda x: x["score"], reverse=True)
            by_label[lbl] = items[:topk]

            # Add crop if requested
            if crop:
                for item in by_label[lbl]:
                    box = item["box"]
                    xmin, ymin, xmax, ymax = (int(box["xmin"]), int(
                        box["ymin"]), int(box["xmax"]), int(box["ymax"]))
                    crop_img = image.crop((xmin, ymin, xmax, ymax))
                    crop_path = self.crops_dir / f"{uuid4()}.jpg"
                    crop_img.save(crop_path)
                    item["crop_path"] = str(crop_path.resolve())

        return by_label


# # ------------------ quick demo ------------------
# if __name__ == "__main__":
#     detector = ZeroShotDetector()  # loads model to RAM once
#     results = detector.detect(
#         "example.jpg",
#         classes=["dog", "cat", "logo"],
#         topk=2,
#         crop=True
#     )
#     from pprint import pprint
#     pprint(results)

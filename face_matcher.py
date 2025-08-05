"""
Autonomous Video Hunter - Face Detection and Matching

This module provides face detection and matching capabilities using DeepFace and OpenCV.
It builds in-memory face databases from reference images and performs similarity matching
for OSINT person identification workflows.

Key Features:
- In-memory face embedding database
- Multiple face detection backends (RetinaFace, MTCNN, etc.)
- Configurable similarity thresholds
- Face cropping and storage for evidence collection
- Vectorized similarity computation for performance

Author: @kdr
"""

import os, uuid, cv2, numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from deepface import DeepFace


class FaceMatcher:
    """
    In-memory one-shot face database + matcher
    -------------------------------------------------
    Example
    -------
    refs = ["alice.jpg", "bob.jpg"]
    names = ["Alice", "Bob"]
    fm = FaceMatcher(reference_images=refs, names=names)

    print(fm.is_match("alice.jpg", "party_photo.jpg"))
    print(fm.matches("group_photo.jpg", topk=3, crop=True))
    """

    def __init__(
        self,
        reference_images: List[str],
        names: Optional[List[str]] = None,
        media_dir: str | Path = "./faces",
        model_name: str = "VGG-Face",
        detector_backend: str = "retinaface",
        align: bool = True,
        match_threshold: float = 0.68,  # cosine-distance threshold (lower == closer)
    ):
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.detector_backend = detector_backend
        self.align = align
        self.match_threshold = match_threshold

        if names and len(names) != len(reference_images):
            raise ValueError("names list must match reference_images length")

        # Build in-memory DB of embeddings
        self.db: List[Dict[str, Any]] = []
        for idx, img_path in enumerate(reference_images):
            faces = self._represent(img_path)
            if not faces:
                continue  # nothing detected
            # Store every detected face in this image
            for f in faces:
                self.db.append(
                    {
                        "name": names[idx] if names else Path(img_path).stem,
                        "embedding": np.array(f["embedding"], dtype=np.float32),
                        "facial_area": f["facial_area"],
                        "image_path": img_path,
                    }
                )
        if not self.db:
            raise RuntimeError("No faces found in reference images")

        # Stack embeddings for fast vectorised distance checks
        self._db_embeddings = np.stack([entry["embedding"] for entry in self.db])

    # ---------- public APIs -------------------------------------------------

# --- replace the whole is_match with this ---------------------------------
    def is_match(
        self,
        target_face_image_path: str,
        test_face_img_path: str,
        crop: bool = False,
    ) -> Dict[str, Any]:
        """
        Compare the single face in `target_face_image_path` against *all* faces
        in `test_face_img_path`.  Returns the best-scoring pair.
        """
        tgt = self._first_face_or_none(target_face_image_path)
        if not tgt:
            return {"match": False, "reason": "target_face_not_found"}

        test_faces = self._represent(test_face_img_path)
        if not test_faces:
            return {"match": False, "reason": "test_faces_not_found"}

        best_sim, best_face = -1.0, None
        for face in test_faces:
            sim = self._confidence(tgt["embedding"], face["embedding"])
            if sim > best_sim:
                best_sim, best_face = sim, face

        passed = best_sim >= (1 - self.match_threshold)
        result: Dict[str, Any] = {
            "match": passed,
            "confidence": float(best_sim),
            "bounding_box": best_face["facial_area"],
            "image_path": test_face_img_path,   # <── NEW
            "num_faces": len(test_faces),
        }
        if crop:
            result["crop_image_path"] = self._save_crop(
                test_face_img_path, best_face["facial_area"]
            )
        return result


# --- replace the whole matches with this ----------------------------------
    def matches(
        self,
        test_face_img_path: str,
        topk: int = 1,
        crop: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        For *every* face in `test_face_img_path` find the best-matching reference
        faces.  Returns up to `topk` high-confidence matches (unique by name).
        """
        test_faces = self._represent(test_face_img_path)
        if not test_faces:
            return []

        # Gather (sim, bbox, ref_idx) for *all* face-pair comparisons
        candidates = []
        for face in test_faces:
            sims = self._similarities(face["embedding"])
            for idx, sim in enumerate(sims):
                if sim >= (1 - self.match_threshold):        # passes threshold
                    candidates.append((sim, face["facial_area"], idx))

        if not candidates:
            return []

        # Sort descending by similarity
        candidates.sort(key=lambda x: x[0], reverse=True)

        results, seen_names = [], set()
        for sim, bbox, ref_idx in candidates:
            ref = self.db[ref_idx]
            if ref["name"] in seen_names:        # keep only one hit per person
                continue
            entry = {
                "name": ref["name"],
                "confidence": float(sim),
                "bounding_box": bbox,
                "image_path": ref["image_path"],  # <── NEW  (reference img)
            }
            if crop:
                entry["crop_image_path"] = self._save_crop(test_face_img_path, bbox)
            results.append(entry)
            seen_names.add(ref["name"])
            if len(results) >= topk:
                break

        return results

    # ---------- internal helpers -------------------------------------------

    def _represent(self, img: str):
        """Run DeepFace.represent once."""
        return DeepFace.represent(
            img_path=img,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            align=self.align,
            enforce_detection=False,
        )

    def _first_face_or_none(self, img: str):
        faces = self._represent(img)
        return faces[0] if faces else None

    @staticmethod
    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    def _confidence(self, u: np.ndarray, v: np.ndarray) -> float:
        # Convert cosine *distance* threshold into a "confidence" (similarity) 0-1
        return self._cosine(u, v)

    def _similarities(self, emb: np.ndarray) -> np.ndarray:
        # emb: (d,)  _db_embeddings: (N,d)
        sims = emb @ self._db_embeddings.T / (
            np.linalg.norm(emb) * np.linalg.norm(self._db_embeddings, axis=1)
        )
        return sims.astype(np.float32)

    def _save_crop(self, img_path: str, box: Dict[str, int]) -> str:
        img = cv2.imread(img_path)
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        crop = img[y : y + h, x : x + w]
        out_path = self.media_dir / f"{uuid.uuid4()}.jpg"
        cv2.imwrite(str(out_path), crop)
        return str(out_path.resolve())
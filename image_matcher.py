"""
Autonomous Video Hunter - RANSAC-Based Image Matching

This module provides robust image matching capabilities using SIFT/ORB feature detection
and RANSAC-based homography estimation. It's designed for matching buildings, logos,
landmarks, and other visual elements across video frames for OSINT analysis.

Key Features:
- SIFT and ORB feature detection with automatic fallback
- RANSAC homography estimation for robust matching
- Configurable matching thresholds and quality metrics
- Batch processing against image databases
- Match visualization and evidence generation
- Comprehensive debug logging for analysis verification

Author: @kdr
"""

from __future__ import annotations
import cv2, numpy as np, os, uuid
from pathlib import Path
from typing import List, Dict, Any, Optional


class ImageRansacDBMatcher:
    """
    Tiny in-RAM image DB with RANSAC matching.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        image_paths: List[str | Path],
        labels: Optional[List[str]] = None,
        detector: Optional[str] = None,
    ):
        # ------------ label sanity ------------------------------------ #
        if labels is None:
            labels = [Path(p).stem for p in image_paths]
        if len(labels) != len(image_paths):
            raise ValueError("image_paths and labels length mismatch")

        # ------------ pick detector ----------------------------------- #
        self.detector, self.use_flann = self._choose_detector(detector)

        # ------------ pre-compute DB features ------------------------- #
        self.db: List[Dict[str, Any]] = []
        for p, lab in zip(image_paths, labels):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(p)
            k, d = self.detector.detectAndCompute(img, None)
            self.db.append({
                "path": Path(p),
                "label": lab,
                "keypoints": k,
                "descriptors": d,
                "shape": img.shape,
            })

        # Create matcher once and reuse (like original implementation)
        self.matcher = self._make_matcher()

    # ------------------------------------------------------------------ #
    def _choose_detector(self, requested: Optional[str]):
        req = requested.lower() if requested else None
        if req == "sift":
            return cv2.SIFT_create(), True
        if req == "orb":
            return cv2.ORB_create(5000), False
        # auto probe
        try:
            return cv2.SIFT_create(), True
        except AttributeError:
            return cv2.ORB_create(5000), False

    def _make_matcher(self):
        if self.use_flann:
            return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ------------------------------------------------------------------ #
    def get_matches(
        self,
        query_image: str | Path,
        *,
        min_inliers: int = 30,
        min_inlier_ratio: float = 0.4,
        ratio_test: float = 0.75,
        draw_matches: bool = False,
        save_dir: str | Path = "matches",
        debug: bool = False,
    ) -> List[Dict[str, Any]]:

        if debug:
            print(f"\n      🔧 ImageRansacDBMatcher.get_matches() DEBUG:")
            print(f"         Query image: {query_image}")
            print(
                f"         Parameters: min_inliers={min_inliers}, min_ratio={min_inlier_ratio}, ratio_test={ratio_test}"
            )
            print(f"         Database has {len(self.db)} reference images")

        q_img = cv2.imread(str(query_image), cv2.IMREAD_GRAYSCALE)
        if q_img is None:
            if debug:
                print(f"         ❌ Could not load query image: {query_image}")
            raise FileNotFoundError(query_image)

        q_k, q_d = self.detector.detectAndCompute(q_img, None)
        if q_d is None:
            if debug:
                print(f"         ❌ No descriptors found in query image")
            return []

        if debug:
            print(
                f"         ✅ Query image loaded: {len(q_k)} keypoints, {q_d.shape[0]} descriptors"
            )

        os.makedirs(save_dir, exist_ok=True)
        results: List[Dict[str, Any]] = []

        for i, entry in enumerate(self.db):
            if debug:
                print(
                    f"\n         📷 Comparing against DB image {i+1}/{len(self.db)}: {entry['label']}"
                )
                print(f"            Path: {entry['path']}")

            db_d = entry["descriptors"]
            if db_d is None:
                if debug:
                    print(f"            ❌ DB image has no descriptors")
                    results.append({
                        "db_img_path": str(entry["path"].resolve()),
                        "label": entry["label"],
                        "match_details": {
                            "reason": "DB image has no descriptors"
                        },
                        "match_draw_path": None,
                    })
                continue

            if debug:
                print(
                    f"            ✅ DB image has {len(entry['keypoints'])} keypoints, {db_d.shape[0]} descriptors"
                )

            matcher = self._make_matcher()
            raw = matcher.knnMatch(db_d, q_d, k=2)  # DB → query

            if debug:
                print(f"            🔍 Found {len(raw)} raw matches")

            good = [m for m, n in raw if m.distance < ratio_test * n.distance]

            if debug:
                print(
                    f"            ✂️  After ratio test ({ratio_test}): {len(good)} good matches"
                )

            if len(good) < 4:
                if debug:
                    print(
                        f"            ❌ Not enough good matches (need ≥4, got {len(good)})"
                    )
                    results.append({
                        "db_img_path": str(entry["path"].resolve()),
                        "label": entry["label"],
                        "match_details": {
                            "num_matches": len(good),
                            "reason": "Not enough good feature matches"
                        },
                        "match_draw_path": None,
                    })
                continue

            src = np.float32([entry["keypoints"][m.queryIdx].pt
                              for m in good]).reshape(-1, 1, 2)
            dst = np.float32([q_k[m.trainIdx].pt
                              for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            if H is None:
                if debug:
                    print(f"            ❌ Homography estimation failed")
                    results.append({
                        "db_img_path": str(entry["path"].resolve()),
                        "label": entry["label"],
                        "match_details": {
                            "num_matches": len(good),
                            "reason": "Homography failed"
                        },
                        "match_draw_path": None,
                    })
                continue

            inliers = int(mask.sum())
            inlier_ratio = inliers / len(good)
            is_good = (inliers >= min_inliers) and (inlier_ratio
                                                    >= min_inlier_ratio)

            if debug:
                print(
                    f"            🎯 RANSAC results: {inliers} inliers, ratio: {inlier_ratio:.3f}"
                )
                print(
                    f"            📊 Thresholds: min_inliers={min_inliers}, min_ratio={min_inlier_ratio}"
                )
                print(
                    f"            {'✅ PASS' if is_good else '❌ FAIL'} threshold check"
                )

            if not is_good and not debug:
                continue  # skip silently

            details = {
                "num_matches": len(good),
                "num_inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "homography": H,
            }
            if debug:
                details["keypoints_db"] = len(entry["keypoints"])
                details["keypoints_query"] = len(q_k)
                if not is_good:
                    details["reason"] = "Thresholds not met"

            vis_path = None
            if draw_matches and is_good:
                if debug:
                    print(f"            🎨 Drawing match visualization...")
                vis = cv2.drawMatches(
                    cv2.imread(str(entry["path"]), cv2.IMREAD_GRAYSCALE),
                    entry["keypoints"],
                    q_img,
                    q_k,
                    good,
                    None,
                    matchesMask=mask.ravel().tolist(),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                vis_path = Path(save_dir) / f"{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(vis_path), vis)
                if debug:
                    print(
                        f"            💾 Match visualization saved: {vis_path}")

            results.append({
                "db_img_path":
                str(entry["path"].resolve()),
                "label":
                entry["label"],
                "match_details":
                details,
                "match_draw_path":
                str(vis_path.resolve()) if vis_path else None,
            })

        results.sort(
            key=lambda r: r["match_details"].get("num_inliers", 0),
            reverse=True,
        )

        if debug:
            print(f"\n         📊 FINAL MATCHING RESULTS:")
            print(f"            Total DB images processed: {len(self.db)}")
            print(f"            Results returned: {len(results)}")
            successful_matches = [
                r for r in results
                if r["match_details"].get("num_inliers", 0) >= min_inliers
            ]
            print(
                f"            Successful matches (≥{min_inliers} inliers): {len(successful_matches)}"
            )

            if successful_matches:
                for j, match in enumerate(
                        successful_matches[:3]):  # Show top 3
                    details = match["match_details"]
                    print(
                        f"               #{j+1}: {match['label']} - {details.get('num_inliers', 0)} inliers ({details.get('inlier_ratio', 0):.3f})"
                    )
            else:
                print(
                    f"            ❌ No matches met the threshold requirements")

        return results

    # ------------------------------------------------------------------ #
    def is_a_match(
        self,
        img1_path: str | Path,
        img2_path: str | Path,
        *,
        min_inliers: int = 30,
        min_inlier_ratio: float = 0.4,
        ratio_test: float = 0.75,
        draw_matches: bool = False,
        out_path: Optional[str | Path] = None,
        debug: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare *img1_path* (reference) with *img2_path* (candidate).

        Returns
        -------
        dict | None
            Same schema as get_matches() entries:
                {
                "db_img_path": Path(img1_path),
                "label":       <basename of img1>,
                "match_details": {...},
                "match_draw_path": Path | None
                }
            • When debug=False and thresholds are not met → returns None.
            • When debug=True you always get a dict, with a "reason" key on failure.
        """
        # ---- load & describe both images ---------------------------------- #
        im1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
        if im1 is None or im2 is None:
            raise FileNotFoundError("Could not read one or both images")

        k1, d1 = self.detector.detectAndCompute(im1, None)
        k2, d2 = self.detector.detectAndCompute(im2, None)
        if d1 is None or d2 is None:
            details = {"reason": "No descriptors found."}
            return ({
                "db_img_path": str(Path(img1_path).resolve()),
                "label": Path(img1_path).stem,
                "match_details": details,
                "match_draw_path": None,
            } if debug else None)

        matcher = self._make_matcher()
        raw = matcher.knnMatch(d1, d2, k=2)  # reference  → candidate
        good = [m for m, n in raw if m.distance < ratio_test * n.distance]
        if len(good) < 4:
            if debug:
                details = {
                    "num_matches": len(good),
                    "reason": "Not enough good matches"
                }
                return {
                    "db_img_path": str(Path(img1_path).resolve()),
                    "label": Path(img1_path).stem,
                    "match_details": details,
                    "match_draw_path": None,
                }
            return None

        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None:
            if debug:
                details = {
                    "num_matches": len(good),
                    "reason": "Homography failed"
                }
                return {
                    "db_img_path": str(Path(img1_path).resolve()),
                    "label": Path(img1_path).stem,
                    "match_details": details,
                    "match_draw_path": None,
                }
            return None

        inliers = int(mask.sum())
        inlier_ratio = inliers / len(good)
        is_good = (inliers >= min_inliers) and (inlier_ratio
                                                >= min_inlier_ratio)

        if not is_good and not debug:
            return None

        details = {
            "num_matches": len(good),
            "num_inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "homography": H,
        }
        if debug and not is_good:
            details["reason"] = "Thresholds not met"

        vis_path = None
        if draw_matches and (is_good or debug):
            if out_path is None:
                out_path = Path("matches") / f"{uuid.uuid4().hex}.jpg"
                out_path.parent.mkdir(exist_ok=True)
            vis = cv2.drawMatches(
                im1,
                k1,
                im2,
                k2,
                good,
                None,
                matchesMask=mask.ravel().tolist(),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imwrite(str(out_path), vis)
            vis_path = Path(out_path)

        return {
            "db_img_path": str(Path(img1_path).resolve()),
            "label": Path(img1_path).stem,
            "match_details": details,
            "match_draw_path": str(vis_path.resolve()) if vis_path else None,
        }

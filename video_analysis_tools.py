"""
Autonomous Video Hunter - Core Analysis Tools

This module provides the foundational tools for video analysis within the OSINT workflow.
Each tool is designed to be called by the main agent or specialized sub-agents to perform
specific analysis tasks on video content.

Tools Provided:
- Video context retrieval and collection search
- Face detection and matching against reference databases
- Object detection using zero-shot models
- Image matching for buildings, logos, and visual elements
- Sentiment and temporal analysis of video content
- Cross-video comparison and correlation
- Investigation memory management

Author: @kdr
"""

import os
from typing import Literal, List, Dict, Any, Optional
from pathlib import Path

# Import your existing modules
from video_context import VideoContextStore
from face_matcher import FaceMatcher
from image_matcher import ImageRansacDBMatcher
from zeroshot_detect import ZeroShotDetector
from video_understander import VideoUnderstander


# Initialize global instances with configurable paths
DEFAULT_DB_PATH = os.environ.get("VIDEO_CONTEXT_DB_PATH", "db.jsonl")
context_store = VideoContextStore(DEFAULT_DB_PATH)
video_understander = VideoUnderstander(api_key=os.environ.get("CLOUDGLUE_API_KEY"))


def get_video_context(
    video_path: str,
    include_local_frames: bool = False
) -> Dict[str, Any]:
    """Get comprehensive context and summary for a video from the context store"""
    context = context_store.get_current_context(
        video_path, 
        include_local_frames=include_local_frames
    )
    if context is None:
        return {"error": "Video not found in context database", "video_path": video_path}
    return context


def search_video_collection(
    query: str,
    start: int = 0,
    limit: int = 50,
    include_local_frames: bool = False
) -> Dict[str, Any]:
    """Search through all videos in the context store for relevant content"""
    collection = context_store.get_collection_context(
        start=start,
        limit=limit,
        include_local_frames=include_local_frames
    )
    
    # Simple text search through summaries and memories
    # You could enhance this with semantic search
    filtered_videos = []
    query_lower = query.lower()
    
    for video_ctx in collection["video_contexts"]:
        # Search in description
        description = video_ctx.get("summary", {}).get("description", "")
        if query_lower in description.lower():
            filtered_videos.append(video_ctx)
            continue
            
        # Search in memories
        for memory in video_ctx.get("memories", []):
            if query_lower in memory.get("value", "").lower():
                filtered_videos.append(video_ctx)
                break
    
    return {
        "query": query,
        "total_found": len(filtered_videos),
        "videos": filtered_videos
    }


def detect_faces_in_video(
    video_path: str,
    reference_images: List[str],
    reference_names: Optional[List[str]] = None,
    topk: int = 5,
    crop_faces: bool = True
) -> Dict[str, Any]:
    """Match faces in video against a database of reference faces"""
    try:
        # Get video context to find frame paths
        context = get_video_context(video_path, include_local_frames=True)
        if "error" in context:
            return context
            
        local_frames = context.get("summary", {}).get("local_frames", [])
        if not local_frames:
            return {"error": "No frames available for face detection", "video_path": video_path}
        
        # Initialize face matcher
        face_matcher = FaceMatcher(
            reference_images=reference_images,
            names=reference_names,
            match_threshold=0.68
        )
        
        results = []
        for frame_info in local_frames:
            # Extract thumbnail_path from frame info dict
            frame_path = frame_info.get("thumbnail_path") if isinstance(frame_info, dict) else frame_info
            if frame_path and Path(frame_path).exists():
                matches = face_matcher.matches(frame_path, topk=topk, crop=crop_faces)
                if matches:
                    results.append({
                        "frame_path": frame_path,
                        "timestamp": frame_info.get("timestamp") if isinstance(frame_info, dict) else None,
                        "matches": matches
                    })
        
        return {
            "video_path": video_path,
            "total_frames_analyzed": len(local_frames),
            "frames_with_matches": len(results),
            "face_matches": results
        }
        
    except Exception as e:
        return {"error": str(e), "video_path": video_path}


def detect_objects_in_video(
    video_path: str,
    target_objects: List[str],
    topk: int = 3,
    crop_objects: bool = True
) -> Dict[str, Any]:
    """Detect specific objects in video frames using zero-shot detection"""
    try:
        context = get_video_context(video_path, include_local_frames=True)
        if "error" in context:
            return context
            
        local_frames = context.get("summary", {}).get("local_frames", [])
        if not local_frames:
            return {"error": "No frames available for object detection", "video_path": video_path}
        
        detector = ZeroShotDetector(crops_dir="crops")
        
        results = []
        for frame_info in local_frames:
            # Extract thumbnail_path from frame info dict
            frame_path = frame_info.get("thumbnail_path") if isinstance(frame_info, dict) else frame_info
            if frame_path and Path(frame_path).exists():
                detections = detector.detect(
                    frame_path, 
                    classes=target_objects, 
                    topk=topk, 
                    crop=crop_objects
                )
                
                # Filter out detections with very low confidence
                filtered_detections = {}
                for obj_class, detections_list in detections.items():
                    high_conf_detections = [d for d in detections_list if d.get("score", 0) > 0.3]
                    if high_conf_detections:
                        filtered_detections[obj_class] = high_conf_detections
                
                if filtered_detections:
                    results.append({
                        "frame_path": frame_path,
                        "timestamp": frame_info.get("timestamp") if isinstance(frame_info, dict) else None,
                        "detections": filtered_detections
                    })
        
        return {
            "video_path": video_path,
            "target_objects": target_objects,
            "total_frames_analyzed": len(local_frames),
            "frames_with_detections": len(results),
            "object_detections": results
        }
        
    except Exception as e:
        return {"error": str(e), "video_path": video_path}


def match_images_in_video(
    video_path: str,
    reference_images: List[str],
    reference_labels: Optional[List[str]] = None,
    min_inliers: int = 15,
    min_inlier_ratio: float = 0.4
) -> Dict[str, Any]:
    """Match buildings, logos, or other visual elements against reference images"""
    try:
        print(f"\n🔍 MATCH_IMAGES_IN_VIDEO DEBUG:")
        print(f"  Video: {video_path}")
        print(f"  Reference images: {len(reference_images)} files")
        print(f"  Reference files: {reference_images}")
        print(f"  Min inliers: {min_inliers}, Min ratio: {min_inlier_ratio}")
        
        context = get_video_context(video_path, include_local_frames=True)
        if "error" in context:
            print(f"  ❌ Error getting video context: {context}")
            return context
            
        local_frames = context.get("summary", {}).get("local_frames", [])
        print(f"  📁 Found {len(local_frames)} frames in video context")
        
        if not local_frames:
            print(f"  ❌ No frames available for image matching")
            return {"error": "No frames available for image matching", "video_path": video_path}
        
        # Check if reference images exist
        existing_refs = []
        for ref_img in reference_images:
            if Path(ref_img).exists():
                existing_refs.append(ref_img)
                print(f"  ✅ Reference image exists: {ref_img}")
            else:
                print(f"  ❌ Reference image missing: {ref_img}")
        
        if not existing_refs:
            print(f"  ❌ No valid reference images found!")
            return {"error": "No valid reference images found", "video_path": video_path}
        
        print(f"  🔧 Creating ImageRansacDBMatcher with {len(existing_refs)} reference images...")
        matcher = ImageRansacDBMatcher(
            image_paths=existing_refs,
            labels=reference_labels
        )
        print(f"  ✅ Matcher created successfully")
        
        results = []
        processed_frames = 0
        
        for i, frame_info in enumerate(local_frames):
            # Extract thumbnail_path from frame info dict
            frame_path = frame_info.get("thumbnail_path") if isinstance(frame_info, dict) else frame_info
            timestamp = frame_info.get("timestamp") if isinstance(frame_info, dict) else None
            
            print(f"\n  📸 Frame {i+1}/{len(local_frames)}: {frame_path}")
            print(f"     Timestamp: {timestamp}")
            
            if not frame_path:
                print(f"     ❌ No frame path found")
                continue
                
            if not Path(frame_path).exists():
                print(f"     ❌ Frame file does not exist: {frame_path}")
                continue
                
            print(f"     ✅ Frame file exists, running matcher...")
            processed_frames += 1
            
            matches = matcher.get_matches(
                frame_path,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
                draw_matches=True,
                save_dir="matches",
                debug=True  # Enable debug mode
            )
            
            print(f"     📊 Matcher returned {len(matches)} matches")
            
            if matches:
                print(f"     ✅ Found {len(matches)} matches!")
                for j, match in enumerate(matches):
                    details = match.get("match_details", {})
                    print(f"        Match {j+1}: {match.get('label')} - {details.get('num_inliers', 0)} inliers, ratio: {details.get('inlier_ratio', 0):.3f}")
                
                results.append({
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "matches": matches
                })
            else:
                print(f"     ❌ No matches found for this frame")
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"  Total frames in video: {len(local_frames)}")
        print(f"  Frames actually processed: {processed_frames}")
        print(f"  Frames with matches: {len(results)}")
        
        return {
            "video_path": video_path,
            "total_frames_analyzed": len(local_frames),
            "frames_processed": processed_frames,
            "frames_with_matches": len(results),
            "image_matches": results
        }
        
    except Exception as e:
        print(f"  ❌ Exception in match_images_in_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "video_path": video_path}


def analyze_video_sentiment(
    video_path: str,
    focus_area: Literal["overall", "speech", "visual", "music"] = "overall"
) -> Dict[str, Any]:
    """Analyze sentiment and emotional tone of video content"""
    try:
        context = get_video_context(video_path)
        if "error" in context:
            return context
        
        description = context.get("summary", {}).get("description", "")
        
        # Extract sentiment-relevant information from existing context
        summary = context.get("summary", {})
        
        # Basic sentiment analysis based on video characteristics
        sentiment_indicators = {
            "has_face": summary.get("has_face", False),
            "has_speech": summary.get("has_speech", False), 
            "is_outdoors": summary.get("is_outdoors", False),
            "has_text_on_screen": summary.get("has_text_on_screen", False),
            "has_logo": summary.get("has_logo", False)
        }
        
        # Simple heuristic-based sentiment (you could enhance with ML models)
        sentiment_score = 0.5  # neutral baseline
        
        # Adjust based on content characteristics
        if sentiment_indicators["has_face"]:
            sentiment_score += 0.1  # faces often indicate positive content
        if sentiment_indicators["is_outdoors"]:
            sentiment_score += 0.1  # outdoor scenes often positive
        
        return {
            "video_path": video_path,
            "focus_area": focus_area,
            "sentiment_score": sentiment_score,
            "sentiment_label": "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral",
            "content_indicators": sentiment_indicators,
            "description_excerpt": description[:500] if description else None
        }
        
    except Exception as e:
        return {"error": str(e), "video_path": video_path}


def extract_temporal_info(
    video_path: str,
    info_type: Literal["time_of_day", "weather", "season", "location_clues"] = "time_of_day"
) -> Dict[str, Any]:
    """Extract temporal and environmental information from video"""
    try:
        context = get_video_context(video_path)
        if "error" in context:
            return context
        
        description = context.get("summary", {}).get("description", "")
        summary = context.get("summary", {})
        
        # Analyze description for temporal clues
        temporal_clues = {
            "is_outdoors": summary.get("is_outdoors", False),
            "has_natural_lighting": None,  # Could be enhanced with image analysis
            "weather_indicators": [],
            "time_indicators": [],
            "location_clues": []
        }
        
        # Simple keyword extraction (enhance with NLP)
        if description:
            desc_lower = description.lower()
            
            # Time indicators
            time_keywords = ["morning", "afternoon", "evening", "night", "dawn", "dusk", "sunrise", "sunset"]
            temporal_clues["time_indicators"] = [kw for kw in time_keywords if kw in desc_lower]
            
            # Weather indicators  
            weather_keywords = ["sunny", "cloudy", "rainy", "snowy", "foggy", "clear", "overcast"]
            temporal_clues["weather_indicators"] = [kw for kw in weather_keywords if kw in desc_lower]
            
            # Location clues
            location_keywords = ["urban", "rural", "city", "countryside", "beach", "mountain", "forest", "indoor", "outdoor"]
            temporal_clues["location_clues"] = [kw for kw in location_keywords if kw in desc_lower]
        
        return {
            "video_path": video_path,
            "info_type": info_type,
            "temporal_analysis": temporal_clues,
            "confidence": "low"  # Indicate this is basic analysis
        }
        
    except Exception as e:
        return {"error": str(e), "video_path": video_path}


def compare_videos(
    video_paths: List[str],
    comparison_type: Literal["visual_similarity", "content_similarity", "temporal_sequence"] = "content_similarity"
) -> Dict[str, Any]:
    """Compare multiple videos for similarities, differences, or sequential relationships"""
    try:
        video_contexts = []
        for video_path in video_paths:
            context = get_video_context(video_path)
            if "error" not in context:
                video_contexts.append(context)
        
        if len(video_contexts) < 2:
            return {"error": "Need at least 2 valid videos for comparison"}
        
        comparisons = []
        
        # Compare each pair
        for i in range(len(video_contexts)):
            for j in range(i + 1, len(video_contexts)):
                vid1, vid2 = video_contexts[i], video_contexts[j]
                
                # Basic similarity analysis
                similarity_metrics = {
                    "has_face_both": vid1.get("summary", {}).get("has_face") and vid2.get("summary", {}).get("has_face"),
                    "is_outdoors_both": vid1.get("summary", {}).get("is_outdoors") and vid2.get("summary", {}).get("is_outdoors"),
                    "has_speech_both": vid1.get("summary", {}).get("has_speech") and vid2.get("summary", {}).get("has_speech"),
                    "has_logo_both": vid1.get("summary", {}).get("has_logo") and vid2.get("summary", {}).get("has_logo")
                }
                
                # Calculate simple similarity score
                similarity_score = sum(similarity_metrics.values()) / len(similarity_metrics)
                
                comparisons.append({
                    "video1": vid1["video_path"],
                    "video2": vid2["video_path"],
                    "similarity_score": similarity_score,
                    "similarity_metrics": similarity_metrics
                })
        
        return {
            "comparison_type": comparison_type,
            "videos_analyzed": len(video_contexts),
            "total_comparisons": len(comparisons),
            "comparisons": comparisons
        }
        
    except Exception as e:
        return {"error": str(e)}


def add_video_memory(
    video_path: str,
    memory_name: str,
    memory_value: str
) -> Dict[str, Any]:
    """Add investigative notes or findings to a video's memory store"""
    try:
        context_store.add_memory(video_path, memory_name, memory_value)
        return {
            "video_path": video_path,
            "memory_added": {
                "name": memory_name,
                "value": memory_value
            },
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "video_path": video_path}


def cross_reference_findings(
    entity_type: Literal["person", "object", "location", "logo"],
    entity_identifier: str,
    search_all_videos: bool = True
) -> Dict[str, Any]:
    """Cross-reference a person, object, or location across multiple videos"""
    try:
        if search_all_videos:
            collection = context_store.get_collection_context(start=0, limit=1000)
            videos_to_search = [v["video_path"] for v in collection["video_contexts"]]
        else:
            videos_to_search = []
        
        findings = []
        entity_lower = entity_identifier.lower()
        
        for video_path in videos_to_search:
            context = get_video_context(video_path)
            if "error" in context:
                continue
                
            # Search in description
            description = context.get("summary", {}).get("description", "")
            if entity_lower in description.lower():
                findings.append({
                    "video_path": video_path,
                    "found_in": "description",
                    "context_snippet": description[:200] + "..." if len(description) > 200 else description
                })
            
            # Search in memories
            for memory in context.get("memories", []):
                if entity_lower in memory.get("value", "").lower():
                    findings.append({
                        "video_path": video_path,
                        "found_in": f"memory: {memory.get('name')}",
                        "context_snippet": memory.get("value", "")
                    })
        
        return {
            "entity_type": entity_type,
            "entity_identifier": entity_identifier,
            "total_videos_searched": len(videos_to_search),
            "videos_with_entity": len(findings),
            "findings": findings
        }
        
    except Exception as e:
        return {"error": str(e)}
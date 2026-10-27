import cv2
import numpy as np

# 112x112 alignment target — left eye, right eye, nose tip
# (matches the ArcFace-style template SFace was trained against)
REFERENCE_POINTS_3PT = np.array([
    [38.29, 51.69],
    [73.53, 51.50],
    [56.02, 71.74],
], dtype=np.float32)


class FacialRecognition:
    """Compares a live BlazeFace detection against one enrolled reference face."""

    def __init__(self, sface_model_path, face_detector, reference_image_path, match_threshold=0.36):
        """
        face_detector: an already-constructed mediapipe FaceDetector (reuse the
                        one human_detector.py builds, rather than loading BlazeFace twice)
        reference_image_path: path to a photo of the person to match against
        """
        self.recognizer = cv2.FaceRecognizerSF.create(sface_model_path, "")
        self.match_threshold = match_threshold
        self.reference_embedding = self._embed_reference_image(face_detector, reference_image_path)

    def _get_keypoints_px(self, detection, w, h):
        kps = detection.keypoints
        right_eye = (kps[0].x * w, kps[0].y * h)
        left_eye = (kps[1].x * w, kps[1].y * h)
        nose_tip = (kps[2].x * w, kps[2].y * h)
        return left_eye, right_eye, nose_tip

    def _align(self, frame_bgr, left_eye, right_eye, nose_tip):
        src_pts = np.array([left_eye, right_eye, nose_tip], dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src_pts, REFERENCE_POINTS_3PT, method=cv2.LMEDS)
        if M is None:
            return None
        return cv2.warpAffine(frame_bgr, M, (112, 112))

    def _embed_reference_image(self, face_detector, image_path):
        import mediapipe as mp

        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            raise FileNotFoundError(f"could not read reference image: {image_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = face_detector.detect(mp_image)
        if not result.detections:
            raise RuntimeError(f"no face found in reference image: {image_path}")

        h, w = frame_bgr.shape[:2]
        left_eye, right_eye, nose_tip = self._get_keypoints_px(result.detections[0], w, h)
        aligned = self._align(frame_bgr, left_eye, right_eye, nose_tip)
        if aligned is None:
            raise RuntimeError("could not align reference image face")
        return self.recognizer.feature(aligned)

    def compare(self, frame_bgr, detection):
        """
        frame_bgr: current camera frame, BGR
        detection: one Detection from result.detections (BlazeFace output)
        Returns (is_match: bool, score: float)
        """
        h, w = frame_bgr.shape[:2]
        left_eye, right_eye, nose_tip = self._get_keypoints_px(detection, w, h)

        # temporary: visualize raw keypoints on the unaligned frame
        debug_frame = frame_bgr.copy()
        for pt, color in [(left_eye, (0,255,0)), (right_eye, (0,0,255)), (nose_tip, (255,0,0))]:
            cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 4, color, -1)
        cv2.imwrite('/tmp/debug_keypoints.jpg', debug_frame)

        aligned = self._align(frame_bgr, left_eye, right_eye, nose_tip)
        if aligned is None:
            return False, 0.0

        # cv2.imwrite('/tmp/debug_aligned.jpg', aligned)  # temporary debug line

        embedding = self.recognizer.feature(aligned)
        score = self.recognizer.match(
            self.reference_embedding, embedding, cv2.FaceRecognizerSF_FR_COSINE
        )
        return score >= self.match_threshold, score
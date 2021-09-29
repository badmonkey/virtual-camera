import cv2
from mediapipe.python.solutions import drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_faces
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import selfie_segmentation as mp_selfie_segmentation
from pipey import Pipeable

from virtcam.base import BLACK, Frame, FrameFilter, FrameProcessor, Image, immutable


class SelfieSegmentator(FrameFilter):
    def __init__(self, source: FrameProcessor):
        super().__init__(source)
        self.classifier = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)
        mask = immutable(self.classifier.process(frame.image).segmentation_mask)

        return Frame(frame.config, frame.image, mask)

    @staticmethod
    @Pipeable
    def p(src):
        return SelfieSegmentator(src)


class HolisticDrawer(FrameFilter):
    def __init__(self, source: FrameProcessor):
        super().__init__(source)
        self.classifier = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)
        results = self.classifier.process(frame.image)

        styleT = drawing_styles.get_default_face_mesh_tesselation_style()
        styleC = drawing_styles.get_default_face_mesh_contours_style()
        styleL = drawing_styles.get_default_pose_landmarks_style()
        image = Image(self.config.width, self.config.height, BLACK)

        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=styleT,
        )
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_faces.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=styleC,
        )
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_holistic.POSE_CONNECTIONS,
        #     landmark_drawing_spec=styleL,
        # )
        image.flags.writeable = False

        return Frame(frame.config, image, frame.mask)

    @staticmethod
    @Pipeable
    def p(src):
        return HolisticDrawer(src)


class FaceMeshDrawer(FrameFilter):
    def __init__(self, source: FrameProcessor):
        super().__init__(source)
        self.classifier = mp_faces.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)
        results = self.classifier.process(frame.image)

        styleT = drawing_styles.get_default_face_mesh_tesselation_style()
        styleC = drawing_styles.get_default_face_mesh_contours_style()
        # image = Image(self.config.width, self.config.height, BLACK)
        image = frame.image.copy()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_faces.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=styleT,
                )
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_faces.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=styleC,
                )

        return Frame(frame.config, immutable(image), frame.mask)

    @staticmethod
    @Pipeable
    def p(src):
        return FaceMeshDrawer(src)


class HandDrawer(FrameFilter):
    def __init__(self, source: FrameProcessor):
        super().__init__(source)
        self.classifier = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)
        results = self.classifier.process(frame.image)

        styleHL = drawing_styles.get_default_hand_landmarks_style()
        styleHC = drawing_styles.get_default_hand_connections_style()
        # image = Image(self.config.width, self.config.height, BLACK)
        image = frame.image.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS, styleHL, styleHC
                )
        image.flags.writeable = False

        return Frame(frame.config, image, frame.mask)

    @staticmethod
    @Pipeable
    def p(src):
        return HandDrawer(src)

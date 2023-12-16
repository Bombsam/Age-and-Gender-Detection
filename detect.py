import cv2
import argparse

# Global Constants
FACE_PROTO = "./models/opencv_face_detector.pbtxt"
FACE_MODEL = "./models/opencv_face_detector_uint8.pb"
AGE_PROTO = "./models/age_deploy.prototxt"
AGE_MODEL = "./models/age_net.caffemodel"
GENDER_PROTO = "./models/gender_deploy.prototxt"
GENDER_MODEL = "./models/gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]
GENDER_LIST = ["Male", "Female"]
PADDING = 20


def load_networks():
    try:
        face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        return face_net, age_net, gender_net
    except Exception as e:
        print(f"Error loading networks: {e}")
        raise


def highlight_face(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(
        frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False
    )

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                frame_opencv_dnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                int(round(frame_height / 150)),
                8,
            )
    return frame_opencv_dnn, face_boxes


def detect_gender_age(face_net, age_net, gender_net, frame):
    result_img, face_boxes = highlight_face(face_net, frame)
    if not face_boxes:
        print("No face detected")
        return result_img

    for face_box in face_boxes:
        face = frame[
            max(0, face_box[1] - PADDING) : min(
                face_box[3] + PADDING, frame.shape[0] - 1
            ),
            max(0, face_box[0] - PADDING) : min(
                face_box[2] + PADDING, frame.shape[1] - 1
            ),
        ]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        print(f"Gender: {gender}")

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]
        print(f"Age: {age[1:-1]} years")

        cv2.putText(
            result_img,
            f"{gender}, {age}",
            (face_box[0], face_box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return result_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    args = parser.parse_args()

    try:
        face_net, age_net, gender_net = load_networks()
        video = cv2.VideoCapture(args.image if args.image else 0)
        while cv2.waitKey(1) < 0:
            has_frame, frame = video.read()
            if not has_frame:
                cv2.waitKey()
                break

            result_img = detect_gender_age(face_net, age_net, gender_net, frame)
            cv2.imshow("Detecting age and gender", result_img)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

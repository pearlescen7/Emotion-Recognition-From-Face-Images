import cv2
import argparse, textwrap

def main():
    face_detector = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_alt.xml")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(frame_gray, 1.1, 4)
        
        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            # cv2.imwrite("face.png", face)

            #TODO:Pass the face to the model
            # Depending on the output color the rectangles and add labels
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Emotion recognition from webcam footage.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--model', 
        default="EfficientNet",
        help=textwrap.dedent(
            '''\
                The model to use while classifying faces
                Options: EfficientNet, VGG'''))

    args = parser.parse_args()
    if args.model == "EfficientNet":
        print("Using model: EfficientNet")
        main()
    elif args.model == "VGG":
        print("Using model: VGG")
        main()
    else:
        print("Unrecognized model type.\nAvailable options: EfficientNet, VGG")
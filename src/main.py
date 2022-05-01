import cv2
import argparse, textwrap
from PIL import Image
import torch
import torchvision.models as M
from torchvision import transforms

model_path = None
kdef_num_classes = 7

idx2emote_and_color = \
{
    0 : ("afraid"   , (0, 0, 255)),
    1 : ("angry"    , (0, 215, 255)), 
    2 : ("disgusted", (133, 21, 199)), 
    3 : ("happy"    , (0, 255, 0)), 
    4 : ("neutral"  , (255, 255, 0)), 
    5 : ("sad"      , (255, 0, 0)), 
    6 : ("surprised", (255, 144, 30))
}

def main():

    model = M.vgg11_bn(num_classes=kdef_num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    img_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])

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
            pil_face = Image.fromarray(face)
            input = img_transforms(pil_face)
            output = model(input)
            _, predicted = torch.max(output.data, 0)
            emote, color = idx2emote_and_color[predicted[0]]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, emote, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
        default="VGG",
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
        model_path = "../models/VGG/vgg11_bn_KDEF.pt"
        main()
    else:
        print("Unrecognized model type.\nAvailable options: EfficientNet, VGG")
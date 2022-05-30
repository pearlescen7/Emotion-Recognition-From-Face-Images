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
    if args.model == "VGG":
        from vgg_pytorch import VGG 
        model = VGG.from_pretrained('vgg19_bn', num_classes=kdef_num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # from temp_model import VGG19
        # model = VGG19()
        # model.load_state_dict(torch.load(model_path))
        # model.eval()
    elif args.model == "EfficientNet":
        from efficientnet_pytorch import EfficientNet
        # model = EfficientNet.from_pretrained('efficientnet-b7')
        # model._fc = torch.nn.Linear(in_features=model._fc.in_features, out_features=kdef_num_classes, bias=True)
        model = EfficientNet.from_pretrained('efficientnet-b7')
        model._fc = torch.nn.Linear(in_features=model._fc.in_features, out_features=kdef_num_classes, bias=True)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        raise ValueError("Model type is invalid.")

    if args.cuda == "yes":
        print("CUDA enabled.")
        model.cuda()
    elif args.cuda == "no":
        print("CUDA disabled.")
    else:
        raise ValueError("Invalid CUDA option.")

    img_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    face_detector = cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_alt.xml")
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video capture resolution: {width}x{height}")
   
    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(frame_gray, 1.1, 4)
        
        for (x,y,w,h) in faces:
            face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            # face = frame_gray[y:y+h, x:x+w]
            pil_face = Image.fromarray(face)
            input = img_transforms(pil_face).unsqueeze(0)
            if args.cuda == "yes":
                input = input.cuda()
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            emote, color = idx2emote_and_color[predicted.item()]
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
    
    parser.add_argument(
        '--cuda', 
        default="no",
        help=textwrap.dedent(
            '''\
                Enables CUDA for faster processing.
                Options: no, yes'''))

    args = parser.parse_args()
    if args.model == "EfficientNet":
        print("Using model: EfficientNet")
        model_path = "../models/EfficientNet/EfficientNet_b7_pretrained_FER_batch8_weighted.pt"
        main()
    elif args.model == "VGG":
        print("Using model: VGG")
        model_path = "../models/VGG/VGG19_bn_pretrained_FER_batch8_weighted.pt"
        main()
    else:
        print("Unrecognized model type.\nAvailable options: EfficientNet, VGG")
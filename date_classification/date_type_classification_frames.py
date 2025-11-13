'''
Date types classification of a video (takes frames)

1. https://www.researchgate.net/publication/382588066_Date_Fruit_Detection_and_Classification_based_on…
2. https://www.researchgate.net/publication/351440988_Dataset_for_localization_and_classification_of_M…
3. https://www.mdpi.com/2079-9292/12/3/665 Their dataset: http://doi.org/10.5281/zenodo.4639543
4. (BEST) Model and dataset available (maturity level and date type classification): https://www.kaggle.com/code/ulaelg/date-fruit-predictions
    /kaggle/input/date-fruit-maturity-detection/other/default/1/weights/best.pt

    1. (Swin transformer i think) Multi-task Classification Model – predicts both the variety of the palm tree (Bouisthami, Boufagous, Kholt, or Majhoul) and a binary ripeness status (ripe vs. unripe).
    2. YOLO-based Object Detection Model – localizes individual fruits in images and provides detailed predictions across the four ripeness stages.

https://m.youtube.com/watch?v=SdyV_cQP76o&t=32s&pp=2AEgkAIB
https://www.youtube.com/watch?v=9SZkK5nka_Q    
'''

import os
import base64
from collections import Counter
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from IPython.display import display, HTML, clear_output
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


class MultiTaskSwin(nn.Module):
    def __init__(self, num_classes_type, num_classes_maturity, pretrained=True):
        super(MultiTaskSwin, self).__init__()
        self.backbone = models.swin_t(
            weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.type_head = nn.Linear(in_features, num_classes_type)
        self.maturity_head = nn.Linear(in_features, num_classes_maturity)

    def forward(self, x):
        features = self.backbone(x)
        out_type = self.type_head(features)
        out_maturity = self.maturity_head(features)
        return out_type, out_maturity


class Prediction:
    def __init__(
        self,
        Multi_task_model,
        checkpoint_path,
        Yolo_model,
        confidence_threshold=0.4,
        device='cuda',
        output_dir="videos",
    ):
        if isinstance(checkpoint_path, tuple):
            checkpoint_path = checkpoint_path[0]

        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        self.Multi_task_model = Multi_task_model
        self.Multi_task_model.load_state_dict(self.checkpoint['model_state_dict'])
        self.device = device
        self.Multi_task_model = self.Multi_task_model.to(self.device)
        self.Multi_task_model.eval()
        self.type_classes = self.checkpoint['type_names']
        self.maturity_classes = self.checkpoint['maturity_names']
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # print(f"✅ Model loaded successfully")
        # print("Model Information:")
        # if 'val_acc_type' in self.checkpoint:
            # print(f"   • Type Classes: {self.type_classes}")
            # print(f"   • Date Type accuracy: {self.checkpoint['val_acc_type']:.4f}")

        # if 'val_acc_maturity' in self.checkpoint:
            # print(f"   • Maturity Classes: {self.maturity_classes}")
            # print(f"   • Date Maturity accuracy: {self.checkpoint['val_acc_maturity']:.4f}")

        self.Yolo_model = Yolo_model
        self.confidence_threshold = confidence_threshold
        self.class_names = self.Yolo_model.names
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # print(f"✅ YOLO Model loaded successfully")
        # print("YOLO Model Information:")
        # print(f"   • Model Type: {type(self.Yolo_model).__name__}")
        # print(f"   • Total Classes: {len(self.class_names)}")
        # print(f"   • Confidence Threshold: {self.confidence_threshold}")
        # print(f"   • Class Names:")
        # for class_id, class_name in self.class_names.items():
        #     print(f"     {class_id}: {class_name}")

    def predict_from_folder(self, folder_path, save_csv=None):
        results = []
        exts = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]

        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.endswith(ext) for ext in exts):
                    img_path = os.path.join(root, file)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            outputs_type, outputs_maturity = self.Multi_task_model(img_tensor)
                            probs_type = torch.softmax(outputs_type, dim=1)
                            probs_maturity = torch.softmax(outputs_maturity, dim=1)

                        type_idx = probs_type.argmax(dim=1).item()
                        maturity_idx = probs_maturity.argmax(dim=1).item()

                        yolo_results = self.Yolo_model(img_path, conf=self.confidence_threshold)
                        detections = yolo_results[0]

                        if detections.boxes is not None and len(detections.boxes) > 0:
                            class_ids = detections.boxes.cls.cpu().numpy().astype(int)
                            boxes = detections.boxes.xyxy.cpu().numpy()
                            confidences = detections.boxes.conf.cpu().numpy()
                        else:
                            class_ids = np.array([])
                            boxes = np.array([])
                            confidences = np.array([])

                        class_counts = Counter(class_ids)
                        detection_results = {
                            'total_objects': len(class_ids),
                            'class_counts': {},
                            'class_counts_by_name': {},
                            'detections': []
                        }

                        for class_id, count in class_counts.items():
                            class_name = self.class_names[class_id]
                            detection_results['class_counts'][class_id] = count
                            detection_results['class_counts_by_name'][class_name] = count

                        for class_id, conf, box in zip(class_ids, confidences, boxes):
                            detection_results['detections'].append({
                                'class_id': int(class_id),
                                'class_name': self.class_names[class_id],
                                'confidence': float(conf),
                                'bbox': box.tolist()
                            })

                        summary_parts = [
                            f"{count} {name}"
                            for name, count in detection_results['class_counts_by_name'].items()
                        ]
                        summary_string = ", ".join(summary_parts) if summary_parts else "0 objects"

                        result = {
                            "image_path": img_path,
                            "image_name": os.path.basename(img_path),
                            "pred_type": self.type_classes[type_idx],
                            "type_prob": round(float(probs_type[0][type_idx]), 4),
                            "pred_maturity": self.maturity_classes[maturity_idx],
                            "maturity_prob": round(float(probs_maturity[0][maturity_idx]), 4),
                            "num_object_detect": detection_results['total_objects'],
                            "summary": summary_string
                        }

                        results.append(result)
                    except Exception as e:
                        print(f" Error with {img_path}: {e}")

        df = pd.DataFrame(results)
        if save_csv:
            df.to_csv(save_csv, index=False)
            # print(f"✅ Results saved to {save_csv}")
        clear_output(wait=False)
        # print("====================== Summary =========================\n")

        # print("✅ Total images: ", len(df))
        # print("✅ Type distribution: ", df['pred_type'].value_counts().to_dict())
        # print("✅ Maturity distribution: ", df['pred_maturity'].value_counts().to_dict())
        # print("✅ Avg type confidence: ", round(df['type_prob'].mean(), 2))
        # print("✅ Avg maturity confidence: ", round(df['maturity_prob'].mean(), 4))
        # print("✅ Avg objects per image: ", round(df['num_object_detect'].mean(), 4))
        # print("✅ Total objects detected: ", df['num_object_detect'].sum())
        # print("✅ Images with objects :", (df['num_object_detect'] > 0).sum())
        # print("✅ Images without objects: ", (df['num_object_detect'] == 0).sum())

        return df

    def _predict_single(self, pil_img, verbose=False):
        img = pil_img.convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        self.Multi_task_model.eval()
        with torch.no_grad():
            outputs_type, outputs_maturity = self.Multi_task_model(img_tensor)
            probs_type = torch.softmax(outputs_type, dim=1)
            probs_maturity = torch.softmax(outputs_maturity, dim=1)

        type_idx = probs_type.argmax(dim=1).item()
        maturity_idx = probs_maturity.argmax(dim=1).item()

        type_text = f"{self.type_classes[type_idx]} ({probs_type[0][type_idx]*100:.2f}%)"
        maturity_text = f"{self.maturity_classes[maturity_idx]} ({probs_maturity[0][maturity_idx]*100:.2f}%)"
        if verbose:
            print(f"Predicted type: {type_text}")
            print(f"Predicted maturity:  {maturity_text}")

        np_img = np.array(img)
        bgr_img = np.ascontiguousarray(np_img[:, :, ::-1])
        yolo_results = self.Yolo_model.predict(
            bgr_img, conf=self.confidence_threshold, verbose=False
        )
        detection = yolo_results[0]
        detections = detection.boxes

        summary_1 = "0"
        summary_2 = "No objects detected"
        stage_text = "No stage detected"
        if detections is not None and len(detections) > 0:
            class_ids = detections.cls.cpu().numpy().astype(int)
            confidences = detections.conf.cpu().numpy()

            details = []
            for cls_id, conf in zip(class_ids, confidences):
                cls_name = self.class_names[cls_id]
                details.append(f"{cls_name} ({conf*100:.1f}%)")

            summary_1 = f" {len(class_ids)}"
            summary_2 = f" " + ", ".join(details)
            top_idx = int(np.argmax(confidences))
            stage_text = f"{self.class_names[class_ids[top_idx]]} ({confidences[top_idx]*100:.2f}%)"

        color_map = {
            "Immature": (0, 255, 0),
            "Khalal": (255, 255, 0),
            "Rutab": (255, 165, 0),
            "Tamer": (255, 0, 0),
        }

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:  # noqa: E722
            font = ImageFont.load_default()

        if detections is not None and len(detections) > 0:
            for box, cls, conf in zip(
                detections.xyxy.cpu().numpy(),
                detections.cls.cpu().numpy().astype(int),
                detections.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                class_name = self.class_names[cls]
                color = color_map.get(class_name, (255, 255, 255))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text(
                    (x1, max(y1 - 15, 0)),
                    f"{class_name} {conf:.2f}",
                    fill=color,
                    font=font,
                )

        info_lines = [
            f"Predicted type: {type_text}",
            f"Predicted maturity: {maturity_text}",
            f"Detected stage: {stage_text}",
        ]

        base_image = img.convert("RGBA")
        overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        padding = 12
        margin = 15
        line_spacing = 6
        line_height = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) or font.size
        max_line_width = max(
            (font.getbbox(line)[2] - font.getbbox(line)[0]) for line in info_lines
        )
        block_width = max_line_width + padding * 2
        block_height = (
            line_height * len(info_lines)
            + line_spacing * (len(info_lines) - 1)
            + padding * 2
        )
        overlay_draw.rectangle(
            [margin, margin, margin + block_width, margin + block_height],
            fill=(0, 0, 0, 170),
        )
        text_y = margin + padding
        text_x = margin + padding
        for line in info_lines:
            overlay_draw.text(
                (text_x, text_y),
                line,
                fill=(255, 255, 255, 255),
                font=font,
            )
            text_y += line_height + line_spacing

        annotated_img = Image.alpha_composite(base_image, overlay).convert("RGB")
        return annotated_img, type_text, maturity_text, stage_text, summary_1, summary_2

    def predict_from_path(self, image_path):
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return

        try:
            img = Image.open(image_path).convert("RGB")
            (
                annotated_img,
                type_text,
                maturity_text,
                stage_text,
                summary_1,
                summary_2,
            ) = self._predict_single(img, verbose=True)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            annotated_path = os.path.join(
                self.output_dir, f"{base_name}_annotated.jpg"
            )
            annotated_img.save(annotated_path)
            print(f"Annotated image saved to: {annotated_path}")

            buffered = BytesIO()
            annotated_img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            clear_output(wait=False)
            display(HTML(f"""
               <div style="display: flex; gap: 25px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 30px; border-radius: 20px; box-shadow: 0 15px 35px rgba(0,0,0,0.1);">
                 <img src="data:image/jpeg;base64,{img_base64}" style="height: 500px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
                  <div style="flex: 1; background: white; padding: 25px; border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.2);">
                    <h3 style="color: black; font-size: 24px; margin-bottom: 20px; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px;">🌴 Analysis Results</h3>

                     <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid #3498db; font-size: 19px; color: black;">
                     <span style="font-size: 18px; color: #7A1F0C;font-weight: bold;">🌴 Predicted Type:</span> {type_text}
                     </div>

                    <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid #e74c3c; font-size: 19px; color: black;">
                    <span style="font-size: 18px; color: #7A1F0C;font-weight: bold;">🫘 Predicted Maturity:</span> {maturity_text}
                    </div>

                    <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid #f39c12; font-size: 19px; color: black;">
                    <span style="font-size: 18px; color: #7A1F0C;font-weight: bold;">📊 Number of date bunches:</span> {summary_1}
                    </div>

                    <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid #27ae60; font-size: 19px; color: black;">
                    <p style="font-size: 18px; color:#7A1F0C;font-weight: bold;">📋 Details:</p> {summary_2}
                 </div>
             </div>
           </div>
            """))

        except Exception as e:
            print(f"Error: {str(e)}")

    def predict_from_video(self, video_path, output_path=None):
        if not os.path.exists(video_path):
            print(f"File not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Unable to open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if output_path is None:
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.output_dir, f"{base}_annotated.mp4")
        else:
            output_dirname = os.path.dirname(output_path)
            if not output_dirname:
                output_path = os.path.join(self.output_dir, output_path)
                output_dirname = self.output_dir
            os.makedirs(output_dirname, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                annotated_img, _, _, _, _, _ = self._predict_single(
                    pil_img, verbose=False
                )
                annotated_bgr = cv2.cvtColor(
                    np.array(annotated_img), cv2.COLOR_RGB2BGR
                )
                writer.write(annotated_bgr)

                frame_idx += 1
                if frame_idx % 10 == 0 or frame_idx == frame_total:
                    print(
                        f"Processed {frame_idx}/{frame_total if frame_total else '?'} frames",
                        end="\r",
                    )

        finally:
            cap.release()
            writer.release()

        print(f"\nAnnotated video saved to: {output_path}")

TYPE_CLASSES = ["Boufagous", "Bouisthami", "Kholt", "Majhoul"]
MATURITY_CLASSES = ["Ripe", "Unripe"]


device = "cuda" if torch.cuda.is_available() else "cpu"

multi_task_model = MultiTaskSwin(num_classes_type=len(TYPE_CLASSES),
                                 num_classes_maturity=len(MATURITY_CLASSES),
                                 pretrained=False)

yolo_model = YOLO("best.pt")
checkpoint_path = "Swin_MultiTask.pt"






predictor = Prediction(multi_task_model,
                       checkpoint_path,
                       yolo_model,
                       confidence_threshold=0.4,
                       device=device,
                       output_dir="annotated")
predictor.type_classes = TYPE_CLASSES
predictor.maturity_classes = MATURITY_CLASSES

# predictor.predict_from_video("date_side_drone/ScreenRecording_11-12-2025 22-43-48_1 (1).mov")
# predictor.predict_from_video("date_side_drone/ScreenRecording_11-12-2025 22-43-48_1 (2).mov")
predictor.predict_from_video("date_side_drone/ScreenRecording_11-12-2025 22-43-48_1.mov")
# # predictor.predict_from_video("date_side_drone/stock-video-date-palm-in-montenegro.mp4")
# predictor.predict_from_video("date_side_drone/stock-video-date-palm-in-montenegro2.mp4")
# predictor.predict_from_video("date_side_drone/stock-video-young-green-palm-tree-with-fruits-of-dates-view-from-below.mp4")


# or for single frames
# predictor.predict_from_path("DSC_2998.jpg")
# df = predictor.predict_from_folder("Dataset/1/Test", save_csv="results.csv")

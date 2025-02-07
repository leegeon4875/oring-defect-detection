import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ✅ 모델 경로 설정
MODEL_PATHS = {
    "Baseline Model": "train_test_100.pth",
    "K-Fold Model (Epoch 35)": "k_fold_35_epoch_2.pth",
    "Ensemble Model (Epoch 35)": "ensemble_model_35_epoch.pth"
}

# ✅ 클래스 매핑
CLASS_NAMES = {1: "extruded", 2: "crack", 3: "cutting", 4: "side_stamped"}

# ✅ 라벨별 색상 지정
LABEL_COLORS = {
    "extruded": (255, 0, 0),
    "crack": (0, 0, 255),
    "cutting": (0, 255, 0),
    "side_stamped": (255, 165, 0)
}

# ✅ 배경 제거 클래스
class ImageProcessor:
    @staticmethod
    def remove_background(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        margin = 20
        height, width = image.shape[:2]
        x_new, y_new = max(0, x - margin), max(0, y - margin)
        x_end, y_end = min(width, x + w + margin), min(height, y + h + margin)

        cropped = image[y_new:y_end, x_new:x_end]
        return cropped if cropped.size > 0 else image

# ✅ 모델 로드 및 예측 클래스
class DefectDetector:
    @st.cache_resource
    def load_model(model_path):
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    @staticmethod
    def predict(image, model):
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)

        scores = outputs[0]['scores'].detach().numpy()
        boxes = outputs[0]['boxes'].detach().numpy()
        labels = outputs[0]['labels'].detach().numpy()
        masks = outputs[0]['masks'].detach().squeeze().numpy()

        threshold = 0.5
        selected = scores >= threshold

        if not np.any(selected) or len(boxes) == 0:
            return image, 0, [], []

        return boxes[selected], labels[selected], masks[selected]

# ✅ 시각화 클래스
class Visualizer:
    @staticmethod
    def visualize(image, boxes, labels, masks, mask_display, mask_alpha, line_thickness):
        image_np = np.array(image)
        output = image_np.copy()

        if mask_display == "마스킹 영역 표시":
            mask = np.zeros_like(image_np, dtype=np.uint8)
            for i, m in enumerate(masks):
                m = (m > 0.5).astype(np.uint8) * 255
                color = LABEL_COLORS.get(int(labels[i]), (255, 255, 255))
                mask[m > 0] = color
            output = cv2.addWeighted(image_np, 1 - mask_alpha, mask, mask_alpha, 0)
        else:
            for i, m in enumerate(masks):
                m = (m > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = LABEL_COLORS.get(int(labels[i]), (255, 255, 255))
                cv2.drawContours(output, contours, -1, color, line_thickness)

        return Image.fromarray(output)

# ✅ Streamlit UI
st.title("O-Ring Defect Detection")
st.sidebar.header("설정")

mask_display = st.sidebar.radio("마스킹 표시 옵션", ["마스킹 영역 표시", "경계선만 표시"])
mask_alpha = st.sidebar.slider("마스킹 투명도", 0.1, 1.0, 0.5) if mask_display == "마스킹 영역 표시" else 0.5
line_thickness = st.sidebar.slider("경계선 두께", 1, 5, 2) if mask_display == "경계선만 표시" else 2

model_option = st.sidebar.selectbox("사용할 모델 선택", list(MODEL_PATHS.keys()))
if st.sidebar.button("모델 변경 적용"):
    model = DefectDetector.load_model(MODEL_PATHS[model_option])

uploaded_files = st.sidebar.file_uploader("O-Ring 이미지 업로드 (다중 가능)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    batch_processing = st.sidebar.checkbox("전체 분석 실행")
    
    file_dict = {file.name: file for file in uploaded_files}
    
    if batch_processing:
        for file_name, file in file_dict.items():
            image = Image.open(file).convert("RGB")
            processed_image = ImageProcessor.remove_background(np.array(image))
            processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

            boxes, labels, masks = DefectDetector.predict(processed_pil, model)
            result_image = Visualizer.visualize(processed_pil, boxes, labels, masks, mask_display, mask_alpha, line_thickness)
            st.image(result_image, caption=f"결과: {file_name}", use_container_width=True)
    else:
        selected_file = st.sidebar.selectbox("분석할 이미지 선택", file_dict.keys())
        if selected_file:
            image = Image.open(file_dict[selected_file]).convert("RGB")
            processed_image = ImageProcessor.remove_background(np.array(image))
            processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

            boxes, labels, masks = DefectDetector.predict(processed_pil, model)
            result_image = Visualizer.visualize(processed_pil, boxes, labels, masks, mask_display, mask_alpha, line_thickness)
            st.image(result_image, caption="결함 탐지 결과", use_container_width=True)

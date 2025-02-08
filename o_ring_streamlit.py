import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
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
    def preprocess_image(image):
        """이미지 전처리: RGB 변환 및 크기 조정"""
        if image.mode in ["RGBA", "P", "L"]:
            image = image.convert("RGB")
        image = image.resize((500, 500))  # ✅ 모델에 맞게 크기 조정
        return image

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
        try:
            # ✅ 이미지 타입 및 정보 출력
            st.write(f"📌 **이미지 타입:** {type(image)}")
            st.write(f"📌 **PIL 모드:** {image.mode}")

            # ✅ numpy 변환 확인
            image_np = np.array(image)
            st.write(f"📌 **numpy 변환 완료:** {image_np.shape}, dtype={image_np.dtype}")

            # ✅ Tensor 변환 시도 (오류 발생 여부 확인)
            try:
                image_tensor = F.to_tensor(image).unsqueeze(0)
                st.write("✅ `to_tensor()` 변환 성공!")
            except Exception as e:
                st.error(f"❌ `to_tensor()` 변환 중 오류 발생: {str(e)}")
                return None, 0, [], []

            # ✅ 모델 예측 실행
            with torch.no_grad():
                outputs = model(image_tensor)

            scores = outputs[0]['scores'].detach().numpy()
            boxes = outputs[0]['boxes'].detach().numpy()
            labels = outputs[0]['labels'].detach().numpy()
            masks = outputs[0]['masks'].detach().squeeze().numpy()

            # ✅ 예측 결과 필터링
            threshold = 0.5
            selected = scores >= threshold

            if not np.any(selected) or len(boxes) == 0:
                return image, 0, [], []

            return boxes[selected], labels[selected], masks[selected]

        except Exception as e:
            st.error(f"❌ 예측 중 오류 발생: {str(e)}")
            return None, 0, [], []

# ✅ 시각화 클래스
class Visualizer:
    @staticmethod
    def visualize(image, boxes, labels, masks, mask_display, mask_alpha, line_thickness):
        image_np = np.array(image)

        if mask_display == "마스킹 영역 표시":
            mask = np.zeros_like(image_np, dtype=np.uint8)
            for i, m in enumerate(masks):
                m = (m > 0.5).astype(np.uint8) * 255
                color = LABEL_COLORS.get(int(labels[i]), (255, 255, 255))
                mask[m > 0] = color
            output = cv2.addWeighted(image_np, 1 - mask_alpha, mask, mask_alpha, 0)
        else:
            output = draw_bounding_boxes(
                torch.tensor(image_np).permute(2, 0, 1),
                torch.tensor(boxes),
                labels=[CLASS_NAMES.get(int(l), "unknown") for l in labels],
                colors=[LABEL_COLORS.get(int(l), (255, 255, 255)) for l in labels],
                width=line_thickness,
            ).permute(1, 2, 0).numpy()

        return Image.fromarray(output)

# ✅ Streamlit UI
st.title("O-Ring Defect Detection")
st.sidebar.header("설정")

mask_display = st.sidebar.radio("마스킹 표시 옵션", ["마스킹 영역 표시", "경계선만 표시"])
mask_alpha = st.sidebar.slider("마스킹 투명도", 0.1, 1.0, 0.5) if mask_display == "마스킹 영역 표시" else 0.5
line_thickness = st.sidebar.slider("경계선 두께", 1, 5, 2) if mask_display == "경계선만 표시" else 2

model_option = st.sidebar.selectbox("사용할 모델 선택", list(MODEL_PATHS.keys()))
model = DefectDetector.load_model(MODEL_PATHS[model_option])

uploaded_files = st.sidebar.file_uploader("O-Ring 이미지 업로드 (다중 가능)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    batch_processing = st.sidebar.checkbox("전체 분석 실행")

    file_dict = {file.name: file for file in uploaded_files}

    for file_name, file in file_dict.items() if batch_processing else [list(file_dict.items())[0]]:
        image = Image.open(file).convert("RGB")
        processed_image = ImageProcessor.preprocess_image(image)

        boxes, labels, masks = DefectDetector.predict(processed_image, model)
        result_image = Visualizer.visualize(processed_image, boxes, labels, masks, mask_display, mask_alpha, line_thickness)
        st.image(result_image, caption=f"결과: {file_name}", use_container_width=True)

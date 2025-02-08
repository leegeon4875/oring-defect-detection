import streamlit as st
import torch
import numpy as np
import cv2
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
    "extruded": (255, 0, 0),     # 빨강
    "crack": (0, 0, 255),        # 파랑
    "cutting": (0, 255, 0),      # 초록
    "side_stamped": (255, 165, 0)  # 주황
}

# ✅ 아이콘 매핑
ICON_MAPPING = {
    "extruded": "🔴",
    "crack": "🔵",
    "cutting": "🟢",
    "side_stamped": "🟠"
}

# ✅ 배경 제거 함수
def remove_background(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))  # ✅ np.concatenate 대신 np.vstack 사용
        image_np = image_np[y:y+h, x:x+w]  # 배경을 제거한 관심 영역
    return image_np


# ✅ 이미지 전처리 클래스 (배경 제거 포함)
class ImageProcessor:
    @staticmethod
    def preprocess_image(image):
        """RGB 변환 + 배경 제거 + 크기 조정"""
        if image.mode in ["RGBA", "P", "L"]:
            image = image.convert("RGB")
        image_np = np.array(image)
        image_np = remove_background(image_np)  # ✅ 배경 제거 수행
        image = Image.fromarray(image_np).resize((500, 500))  # ✅ 모델 입력 크기로 조정
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
            # ✅ 이미지 변환 (PIL → Tensor)
            image_tensor = F.to_tensor(image).unsqueeze(0)

            # ✅ 모델 예측 실행
            with torch.no_grad():
                outputs = model(image_tensor)

            scores = outputs[0]['scores'].detach().numpy()
            boxes = outputs[0]['boxes'].detach().numpy()
            labels = outputs[0]['labels'].detach().numpy()
            masks = outputs[0]['masks'].detach().squeeze().numpy()

            # ✅ 예측 결과 필터링 (신뢰도 0.5 이상만)
            threshold = 0.5
            selected = np.where(scores >= threshold)[0]

            if len(selected) == 0:
                return image, [], [], []

            return boxes[selected], labels[selected], masks[selected]

        except Exception as e:
            st.error(f"❌ 예측 중 오류 발생: {str(e)}")
            return None, [], [], []

# ✅ 시각화 클래스
class Visualizer:
    @staticmethod
    def visualize(image, boxes, labels, masks, mask_display, mask_alpha, line_thickness, contour_thickness):
        image_np = np.array(image)

        if mask_display == "마스킹 영역 표시":
            mask = np.zeros_like(image_np, dtype=np.uint8)
            for i, m in enumerate(masks):
                m = (m > 0.5).astype(np.uint8) * 255
                color = LABEL_COLORS.get(CLASS_NAMES[int(labels[i])], (255, 255, 255))
                mask[m > 0] = color

            if len(mask.shape) == 2 or mask.shape[-1] == 1:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            output = cv2.addWeighted(image_np, 1 - mask_alpha, mask, mask_alpha, 0)

        else:
            # ✅ 경계선만 표시 (바운딩 박스 두께와 독립적으로 조절 가능)
            output = image_np.copy()
            for i, m in enumerate(masks):
                m = (m > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = LABEL_COLORS.get(CLASS_NAMES[int(labels[i])], (255, 255, 255))
                cv2.drawContours(output, contours, -1, color, contour_thickness)  # ✅ 경계선 두께 조절 가능

        # ✅ 바운딩 박스 & 결함 종류 추가 (마스킹 & 경계선 옵션 모두 포함)
        labels_list = [CLASS_NAMES[int(l)] for l in labels]
        colors_list = [LABEL_COLORS.get(CLASS_NAMES[int(l)], (255, 255, 255)) for l in labels]

        output = draw_bounding_boxes(
            torch.tensor(output).permute(2, 0, 1),
            boxes_tensor,
            labels=labels_list,
            colors=colors_list,  # ✅ 최적화된 colors_list 사용
            width=line_thickness,
        ).permute(1, 2, 0).numpy()

        return Image.fromarray(output)

# ✅ UI 구성
st.title("O-Ring Defect Detection")

# ✅ 슬라이드바에는 '결과를 확인할 이미지 선택'만 배치
st.sidebar.header("이미지 선택")
selected_file = st.sidebar.selectbox("결과를 확인할 이미지 선택", [file.name for file in uploaded_files])

# ✅ 메인 화면으로 UI 옵션 이동
st.header("설정")
model_option = st.selectbox("사용할 모델 선택", list(MODEL_PATHS.keys()))
mask_display = st.radio("마스킹 표시 옵션", ["마스킹 영역 표시", "경계선만 표시"])
mask_alpha = st.slider("마스킹 투명도", 0.1, 0.6, 0.3, step=0.05) if mask_display == "마스킹 영역 표시" else 0.5
line_thickness = int(st.slider("바운딩 박스 두께", 1.0, 3.0, 1.5, step=0.5))
contour_thickness = int(st.slider("경계선 두께", 1.0, 3.0, 1.5, step=0.5)) if mask_display == "경계선만 표시" else 2  

if uploaded_files:
    selected_file = st.sidebar.selectbox("결과를 확인할 이미지 선택", [file.name for file in uploaded_files])
    file_dict = {file.name: file for file in uploaded_files}
    image = Image.open(file_dict[selected_file]).convert("RGB")
    model = DefectDetector.load_model(MODEL_PATHS[model_option])
    boxes, labels, masks = DefectDetector.predict(image, model)
    result_image = Visualizer.visualize(image, boxes, labels, masks, mask_display, mask_alpha, line_thickness, contour_thickness)
    st.image(result_image, caption=f"결과: {selected_file}", use_container_width=True)

    # ✅ 결함 정보 표시 (lightgray 배경 적용)
    st.write(f"📌 **파일명:** {selected_file}")
    if len(labels) > 0:
        defect_summary = ""
        for defect in set(labels):
            defect_name = CLASS_NAMES[int(defect)]
            defect_count = list(labels).count(defect)
            icon = ICON_MAPPING.get(defect_name, "❓")  # 기본 아이콘 예외처리
            defect_summary += f'<div style="background-color: lightgray; padding: 5px; border-radius: 5px; margin-bottom: 5px;">{icon} <b>{defect_name}</b>: {defect_count}개</div>'
        
        st.markdown(defect_summary, unsafe_allow_html=True)
    else:
        st.write("✅ **정상입니다**")

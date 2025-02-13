import streamlit as st
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
import io

# ✅ 모델 경로 설정
MODEL_PATHS = {
    "Baseline Model": "train_test_100.pth",
    "K-Fold Model": "k_fold_35_epoch_2.pth",
    "Ensemble Model": "ensemble_model_35_epoch.pth"
}

# ✅ 모델 설명 가이드
MODEL_DESCRIPTIONS = {
    "Baseline Model": "빠른 결과가 필요하거나 기본적인 검사를 원할 때 추천합니다.",
    "K-Fold Model": "다양한 환경에서도 안정적인 검사가 필요할 때 선택하세요.",
    "Ensemble Model": "가장 정확한 검사가 필요한 경우 사용하세요."
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

# ✅ 아이콘 매핑
ICON_MAPPING = {
    "extruded": "🔴",
    "crack": "🔵",
    "cutting": "🟢",
    "side_stamped": "🟠"
}

# ✅ 배경 제거 함수
def remove_background(image_np):
    """불필요한 배경 제거 (이진화 + 컨투어 검출 활용)"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # ✅ 가장 큰 컨투어를 찾아서 해당 영역만 남김
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
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
            
            # 신뢰도 임계값 (Threshold) 설정
            confidence_threshold = 0.5

            # ✅ 모델 예측 실행
            with torch.no_grad():
                outputs = model(image_tensor)

            scores = outputs[0]['scores'].detach().numpy() if 'scores' in outputs[0] else []
            filtered_scores = [s for s in scores if s >= confidence_threshold]
            boxes = outputs[0]['boxes'].detach().numpy()
            labels = outputs[0]['labels'].detach().numpy()
            masks = outputs[0]['masks'].detach().squeeze().numpy()

            # ✅ 예측 결과 필터링 (신뢰도 0.5 이상만)
            threshold = 0.5
            selected = np.where(scores >= threshold)[0]

            # ✅ 결함이 없는 경우 빈 리스트 반환 (오류 방지)
            if len(selected) == 0:
                return [], [], []

            return boxes[selected], labels[selected], masks[selected]

        except Exception as e:
            st.error(f"❌ 예측 중 오류 발생: {str(e)}")
            return [], [], []  # ✅ 오류 발생 시에도 빈 리스트 반환

# ✅ 시각화 클래스 추가
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
            output = image_np.copy()
            for i, m in enumerate(masks):
                m = (m > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = LABEL_COLORS.get(CLASS_NAMES[int(labels[i])], (255, 255, 255))
                cv2.drawContours(output, contours, -1, color, contour_thickness)

        # ✅ 바운딩 박스 & 결함 종류 추가 (마스킹 & 경계선 옵션 모두 포함)
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float)
            labels_list = [CLASS_NAMES.get(int(l), "unknown") for l in labels]
            colors_list = [LABEL_COLORS.get(CLASS_NAMES[int(l)], (255, 255, 255)) for l in labels]

            output = draw_bounding_boxes(
                torch.tensor(output).permute(2, 0, 1),
                boxes_tensor,
                colors=colors_list,
                width=line_thickness,
            ).permute(1, 2, 0).numpy()

            # ✅ 바운딩 박스 위에 글자 배경 추가
            for i, (box, label) in enumerate(zip(boxes, labels_list)):
                x1, y1 = int(box[0]), int(box[1])  # 왼쪽 상단 좌표

                # ✅ 배경 사각형 (글자 크기 맞추기 위해 조정)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_w, text_h = text_size
                cv2.rectangle(output, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), (50, 50, 50), -1)  # ✅ 배경 박스 추가

                # ✅ 글자 추가
                cv2.putText(output, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return Image.fromarray(output)

# ✅ JSON 데이터를 저장할 리스트 생성 (결과를 확인한 이미지만 저장)
json_results = []

# ✅ JSON 데이터 변환 함수 (정상 이미지도 포함)
def add_to_json_results(file_name, boxes, labels):
    """결과 데이터를 JSON 형식으로 변환 후 리스트에 추가"""
    results = []
    for i in range(len(labels)):
        result = {
            "class": CLASS_NAMES.get(int(labels[i]), "unknown"),
            "bounding_box": [float(coord) for coord in boxes[i]]
        }
        results.append(result)

    json_data = {
        "file_name": file_name,
        "detections": results  # 결함이 없으면 빈 리스트 []
    }

    # ✅ JSON 결과 리스트에 추가 (중복 방지)
    json_results.append(json_data)
        
# ✅ UI 구성
st.title("O-Ring Defect Detection")
model_option = st.selectbox("사용할 모델 선택", list(MODEL_PATHS.keys()))

# ✅ 선택된 모델 설명 표시
st.write(f"📌 **선택한 모델:** {model_option}")  
st.info(MODEL_DESCRIPTIONS[model_option])  # 모델 설명 표시

mask_display = st.radio("마스킹 표시 옵션", ["마스킹 영역 표시", "경계선만 표시"])
mask_alpha = st.slider("마스킹 투명도", 0.1, 0.7, 0.3, step=0.1) if mask_display == "마스킹 영역 표시" else 0.5
line_thickness = int(st.slider("바운딩 박스 두께", 1.0, 3.0, 1.5, step=0.5))
contour_thickness = int(st.slider("경계선 두께", 1.0, 3.0, 1.5, step=0.5)) if mask_display == "경계선만 표시" else 2

uploaded_files = st.file_uploader("O-Ring 이미지 업로드 (다중 가능)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    selected_file = st.sidebar.selectbox("결과를 확인할 이미지 선택", [file.name for file in uploaded_files])
    file_dict = {file.name: file for file in uploaded_files}
    image = Image.open(file_dict[selected_file]).convert("RGB")
    processed_image = ImageProcessor.preprocess_image(image)  
    model = DefectDetector.load_model(MODEL_PATHS[model_option])
    boxes, labels, masks = DefectDetector.predict(processed_image, model)

    # ✅ JSON 데이터 저장 (결과가 없는 경우도 포함)
    add_to_json_results(selected_file, boxes, labels)

    # ✅ 결과가 있을 경우 시각화
    if len(boxes) > 0:
        result_image = Visualizer.visualize(processed_image, boxes, labels, masks, mask_display, mask_alpha, line_thickness, contour_thickness)
        st.image(result_image, caption=f"결과: {selected_file}", use_container_width=True)
        
        # ✅ 탐지된 결함 정보 요약
        defect_counts = {}
        for label in labels:
            class_name = CLASS_NAMES.get(int(label), "unknown")
            defect_counts[class_name] = defect_counts.get(class_name, 0) + 1

        # ✅ 신뢰도(Confidence) 평균값 계산
        scores = model(F.to_tensor(processed_image).unsqueeze(0))[0]['scores'].detach().numpy()
        avg_confidence = np.mean(filtered_scores) if len(scores) > 0 else 0
                        
        # ✅ 탐지된 결함 정보 표시
        st.write(f"📊 **탐지된 결함 요약 ({selected_file})**")
        for defect, count in defect_counts.items():
            st.write(f"- {ICON_MAPPING.get(defect, '')} **{defect}**: {count}개")
        st.write(f"🔍 **평균 신뢰도:** {avg_confidence:.2f}")   
             
        # ✅ 이미지 데이터를 Bytes로 변환
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # ✅ Streamlit 다운로드 버튼 생성
        st.download_button(
            label="📷 시각화된 이미지 다운로드",
            data=img_byte_arr,
            file_name=f"{selected_file}_result.png",
            mime="image/png"
        )
        
    else:
        st.image(processed_image, caption=f"✅ 정상 이미지: {selected_file}", use_container_width=True)
        st.write("✅ **정상입니다! 결함이 탐지되지 않았습니다.**")

# ✅ JSON 저장 및 다운로드 버튼 추가 (JSON 데이터가 있을 때만 표시)
if json_results:
    st.write("📥 **업로드한 이미지들의 결과를 JSON 파일로 저장할 수 있습니다.**")
    
    if st.button("📥 JSON 저장 및 다운로드"):
        json_path = "results.json"
        with open(json_path, "w") as json_file:
            json.dump(json_results, json_file, indent=4)

        # ✅ JSON 다운로드 버튼 추가
        with open(json_path, "rb") as file:
            st.download_button(
                label="📥 JSON 파일 다운로드",
                data=file,
                file_name="results.json",
                mime="application/json"
            )

        st.success("📁 JSON 파일이 저장 및 다운로드되었습니다!")

# 완벽한 모델
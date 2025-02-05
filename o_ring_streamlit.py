import streamlit as st
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
from collections import Counter

# 클래스 라벨 및 색상 정의
CLASS_LABELS = {1: "extruded", 2: "crack", 3: "cutting", 4: "side_stamped"}
CLASS_COLORS = {1: "red", 2: "blue", 3: "green", 4: "orange"}  # 결함별 색상 지정
BACKGROUND_COLORS = {1: "white", 2: "yellow", 3: "lightgray", 4: "cyan"}  # 가독성을 위한 배경색

# 모델 로딩 함수
@st.cache_resource
def load_model(model_path):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    num_classes = 5  # 실제 클래스 수 + 1 (배경)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 이미지 전처리 함수
def transform_image(image):
    original_size = image.size  # 원본 크기 저장 (width, height)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0), original_size

# 결함 시각화 함수
def visualize_defects(image, outputs, original_size, score_threshold=0.5):
    draw = ImageDraw.Draw(image)

    # 폰트 로드 (Arial이 없을 경우 기본 폰트로 대체)
    try:
        font = ImageFont.truetype("arial.ttf", size=24)  # 글자 크기 확대
    except OSError:
        font = ImageFont.load_default()

    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']
    masks = outputs[0].get('masks')

    # 스케일 비율 계산
    orig_w, orig_h = original_size
    scale_x = orig_w / 256
    scale_y = orig_h / 256

    detected_defects = []  # 탐지된 결함 저장 리스트

    for idx in range(len(boxes)):
        score = scores[idx].item()
        if score < score_threshold:
            continue

        box = boxes[idx].tolist()
        box = [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]

        label = labels[idx].item()
        label_name = CLASS_LABELS.get(label, "unknown")
        box_color = CLASS_COLORS.get(label, "red")  # 결함별 색상 적용
        bg_color = BACKGROUND_COLORS.get(label, "white")  # 가독성을 위한 배경색 적용

        detected_defects.append(label_name)  # 탐지된 결함 저장

        draw.rectangle(box, outline=box_color, width=2)  # 박스 두께 증가

        # 텍스트 배경 처리
        text_size = draw.textsize(label_name, font=font)
        text_background = [
            (box[0], box[1] - text_size[1] - 4),
            (box[0] + text_size[0] + 4, box[1])
        ]
        draw.rectangle(text_background, fill=bg_color)

        draw.text((box[0] + 2, box[1] - text_size[1] - 2), f"{label_name}", fill=box_color, font=font)

        if masks is not None:
            mask = masks[idx, 0].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            mask = cv2.resize(mask, (orig_w, orig_h))

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                for i in range(len(contour) - 1):
                    draw.line([tuple(contour[i]), tuple(contour[i + 1])], fill=box_color, width=2)

    return image, detected_defects

# Streamlit UI
st.title("O-ring 결함 탐지")

# 모델 선택 UI
model_choice = st.selectbox("사용할 모델을 선택하세요", ["Train-Test Split 모델", "K-Fold 모델"])

# 선택한 모델 로딩
model_path = "train_test_100.pth" if model_choice == "Train-Test Split 모델" else "k_fold_50.pth"
model = load_model(model_path)

# 이미지 업로드
uploaded_file = st.file_uploader("O-ring 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='업로드한 이미지', use_column_width=True)

    if st.button("결함 탐지 시작"):
        input_tensor, original_size = transform_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)

        result_image, detected_defects = visualize_defects(image.copy(), outputs, original_size)
        st.image(result_image, caption='결함 탐지 결과', use_column_width=True)

        # 결함 탐지 결과 표시
        if detected_defects:
            defect_counts = Counter(detected_defects)
            st.subheader("🔍 탐지된 결함 및 개수")
            for defect, count in defect_counts.items():
                st.write(f"- {defect}: {count}개")
        else:
            st.subheader("✅ 정상: 결함이 발견되지 않았습니다")

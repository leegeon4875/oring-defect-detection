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
BACKGROUND_COLOR = "lightgray"  # 가독성을 위한 배경색 통일

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
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(binary.shape[1] - x, w + 2 * margin)
        h = min(binary.shape[0] - y, h + 2 * margin)
        cropped_image = image.crop((x, y, x + w, y + h))
    else:
        cropped_image = image

    original_size = cropped_image.size

    transform = T.Compose([
        T.Resize((500, 500)),  # 모델 입력 크기를 500x500으로 확대
        T.ToTensor()
    ])

    return transform(cropped_image).unsqueeze(0), original_size

# 결함 시각화 함수 (마스킹 추가)
def visualize_defects(image, outputs, original_size, score_threshold=0.5):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=36)  # 글자 크기 1.5배~2배 확대
    except OSError:
        font = ImageFont.load_default()

    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']
    masks = outputs[0].get('masks')  # 마스크 정보 가져오기

    detected_defects = []

    # 스케일 비율 계산
    scale_x = image.width / 500
    scale_y = image.height / 500

    for idx in range(len(boxes)):
        score = scores[idx].item()
        if score < score_threshold:
            continue

        box = boxes[idx].tolist()
        box = [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]  # 정확한 스케일링 적용

        label = labels[idx].item()
        label_name = CLASS_LABELS.get(label, "unknown")
        box_color = CLASS_COLORS.get(label, "red")

        detected_defects.append(label_name)

        # 마스크 적용 (투명도 조정 및 외곽선 강조)
        if masks is not None:
            mask = masks[idx, 0].mul(255).byte().cpu().numpy()
            mask_resized = cv2.resize(mask, (image.width, image.height))
            mask_color = np.zeros_like(np.array(image))
            for c in range(3):  # RGB 채널에 색상 적용
                mask_color[:, :, c] = (mask_resized > 127) * np.array(Image.new('RGB', (1, 1), box_color))[0, 0, c]

            # 투명도 조정 (0.4로 증가)
            blended = cv2.addWeighted(np.array(image), 0.6, mask_color, 0.4, 0)

            # 외곽선 강조
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)  # 흰색 외곽선 추가

            image = Image.fromarray(blended)
            draw = ImageDraw.Draw(image)  # 다시 그리기 객체 초기화

        draw.rectangle(box, outline=box_color, width=3)
        bbox = draw.textbbox((0, 0), label_name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_background = [(box[0], box[1] - text_height - 4), (box[0] + text_width + 4, box[1])]
        draw.rectangle(text_background, fill=BACKGROUND_COLOR)
        draw.text((box[0] + 2, box[1] - text_height - 2), f"{label_name}", fill=box_color, font=font)

    return image, detected_defects

# Streamlit UI
st.title("O-ring 결함 탐지")

# 모델 선택 UI
model_choice = st.selectbox("사용할 모델을 선택하세요", ["Train-Test Split 모델", "K-Fold 모델"])
model_path = "train_test_100.pth" if model_choice == "Train-Test Split 모델" else "k_fold_50.pth"
model = load_model(model_path)

# 이미지 업로드
uploaded_files = st.file_uploader("O-ring 이미지를 업로드하세요", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# 탐지 결과 저장
results = {}

if uploaded_files:
    st.sidebar.header("업로드된 파일 목록")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor, original_size = transform_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)

        result_image, detected_defects = visualize_defects(image.copy(), outputs, original_size)
        defect_counts = Counter(detected_defects)

        # 결과 저장
        results[uploaded_file.name] = {
            "result": result_image,
            "defects": defect_counts
        }

    # 파일 목록 표시
    selected_file = st.sidebar.radio("이미지를 선택하세요", list(results.keys()))

    # 선택한 파일의 결과 표시 (결과 이미지만 표시)
    if selected_file:
        st.image(results[selected_file]["result"], caption=f"결함 탐지 결과 - {selected_file}", width=600)

        # 결함 요약 표시
        st.subheader(f"🔍 탐지된 결함 - {selected_file}")
        if results[selected_file]["defects"]:
            for defect, count in results[selected_file]["defects"].items():
                st.write(f"- {defect}: {count}개")
        else:
            st.write("✅ 정상: 결함이 발견되지 않았습니다")

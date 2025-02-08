import streamlit as st
from PIL import Image

# Streamlit 페이지 제목
st.title("🖼️ 이미지 모드 확인 도구")

# 파일 업로드
uploaded_files = st.file_uploader("이미지 업로드 (다중 가능)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"📌 {uploaded_file.name}", use_container_width=True)
            st.write(f"✅ **파일명:** {uploaded_file.name}")
            st.write(f"🎨 **이미지 모드:** `{image.mode}`")
        except Exception as e:
            st.error(f"❌ {uploaded_file.name} 처리 중 오류 발생: {str(e)}")

# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1l14YM0VFtfKbPnTZNSnAMD2TnIn6Phx4")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },

    labels[0]:{"texts":["타코는 멕시칸 유명 요리이다"],
              "videos":["https://youtu.be/jFabzMoMERM?si=Xg4sUR_m-w2WM_Vb"],
              "images":["https://media.istockphoto.com/id/459396345/ko/%EC%82%AC%EC%A7%84/%ED%83%80%EC%BD%94.jpg?s=612x612&w=0&k=20&c=jCegNwXKOV9xcxQXTvxFJu_VH4cl9Ph5YM9z9-QWPMU="]},
    labels[1]:{"texts":["파스타는 이탈리아 유명 요리이다"],
            "videos":["https://youtu.be/-G478hXpaEk?si=V8RYcoWSRndMPUSF"],
              "images":["https://semie.cooking/image/contents/recipe/cv/ru/hwgnaulb/IRD/144082280wkmc.jpg"]},
    labels[2]:{"texts":["피자는 이탈리아 유명 반죽 요리이다"],
               "videos":["https://youtu.be/5LsdZ3QTU0w?si=QgqKJAt8_auKrTzi"],
              "images":["data:image/jpeg;base64,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"]
              },



}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")

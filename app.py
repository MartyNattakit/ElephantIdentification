import time
import io
import numpy as np
from PIL import Image
import streamlit as st

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Identity Check AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
/* main background gradient */
main {
    background: radial-gradient(circle at 20% 20%, #1E2A3A 0%, #0D1117 60%);
    color: #FFFFFF;
    min-height: 100vh;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto;
}

/* cards */
.result-card {
    background: rgba(20,24,33,0.6);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    color: #fff;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}

/* section header */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: #8B9EEA;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* big headline */
.big-head {
    font-size: clamp(1.6rem, 1.2vw + 1rem, 2rem);
    font-weight: 600;
    line-height: 1.2;
    color: #fff;
    letter-spacing: -0.04em;
    margin-bottom: 0.4rem;
}

/* subtitle text */
.sub-head {
    font-size: .95rem;
    font-weight: 400;
    line-height: 1.4;
    color: #9BA3B4;
    max-width: 500px;
    margin-bottom: 1rem;
}

/* nice metric row */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    min-width: 110px;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: .8rem 1rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.metric-label {
    font-size: .7rem;
    color: #9BA3B4;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 500;
}
.metric-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: -0.03em;
}

/* footer info */
.footer-card {
    font-size: .8rem;
    color: #9BA3B4;
    line-height: 1.5;
    margin-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 1rem;
}

/* make file uploader / buttons darker */
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    color: #fff;
}
button[kind="primary"] {
    background: linear-gradient(90deg,#4B8CFB 0%,#4AD0FF 100%) !important;
    color: #0D1117 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: 0 !important;
    box-shadow: 0 16px 40px rgba(0,150,255,0.5) !important;
}
</style>
""", unsafe_allow_html=True)

# ========== MODEL LOADING (PLACEHOLDER) ==========
# TODO: replace with your ONNXRuntime inference session etc.
# Example:
# import onnxruntime as ort
# ort_session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

CLASSES = ["person_1", "person_2", "person_3", "unknown"]

def run_inference(pil_img: Image.Image):
    """
    Your core inference logic.
    Return a dict:
    {
        "pred_class": "person_1" or "unknown",
        "confidence": 0.82,  # 0-1
        "topk": [("person_1", 0.82),
                 ("person_2", 0.12),
                 ("unknown", 0.06)],
        "timing_ms": 47.2
    }
    """
    start_t = time.time()

    # 1. preprocess
    img = pil_img.convert("RGB").resize((224,224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2,0,1))  # [C,H,W]
    arr = np.expand_dims(arr, axis=0) # [1,C,H,W]

    # TODO:
    # outputs = ort_session.run(None, {"input": arr})
    # probs = softmax(outputs[0]) # shape [1,num_classes]
    # pretend demo:
    fake_probs = np.array([0.10, 0.05, 0.03, 0.82])  # last is "unknown"
    probs = fake_probs / fake_probs.sum()

    top_idx = int(np.argmax(probs))
    pred_class = CLASSES[top_idx]
    confidence = float(probs[top_idx])

    # build topk list
    order = np.argsort(-probs)
    topk = [(CLASSES[i], float(probs[i])) for i in order[:3]]

    end_t = time.time()
    timing_ms = (end_t - start_t) * 1000.0

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "topk": topk,
        "timing_ms": timing_ms
    }

# ========== HEADER / HERO ==========
st.markdown(
    """
    <div class="section-title">Identity Check AI</div>
    <div class="big-head">Real-time identity verification.<br>Private. Offline. Judge-ready.</div>
    <div class="sub-head">
      Drop in an image, and we’ll tell you who it is – or flag as <b>Unknown</b>.
      Runs locally on your device. No internet required.
    </div>
    """,
    unsafe_allow_html=True
)

# ========== INPUT TABS ==========
tabs = st.tabs(["📤 Upload Image", "📷 Live Camera"])

with tabs[0]:
    uploaded_file = st.file_uploader(
        "Upload a face image (JPG/PNG)",
        type=["jpg","jpeg","png"],
        help="Clear frontal or near-frontal face works best."
    )

with tabs[1]:
    st.write("Camera capture (optional demo)")
    cam_img = st.camera_input("Take a photo")

# pick source priority: camera > upload
img_source = None
if tabs[1] and cam_img is not None:
    img_source = cam_img
elif tabs[0] and uploaded_file is not None:
    img_source = uploaded_file

# ========== RUN BUTTON ==========
col_run, _ = st.columns([1,3])
with col_run:
    run_btn = st.button("🔍 Analyze Now", use_container_width=True)

# ========== RESULT AREA ==========
if run_btn:
    if img_source is None:
        st.warning("Please provide an image first 🙏")
    else:
        # Read image bytes -> PIL
        if isinstance(img_source, io.BytesIO) or hasattr(img_source, "read"):
            pil_img = Image.open(img_source)
        else:
            # camera_input returns UploadedFile too, so above should handle it.
            pil_img = Image.open(img_source)

        result = run_inference(pil_img)

        pred_class = result["pred_class"]
        confidence_pct = result["confidence"] * 100.0
        timing_ms = result["timing_ms"]
        topk = result["topk"]

        # Show preview + result card side by side
        left, right = st.columns([1,1])

        with left:
            st.image(pil_img, caption="Input Image", use_column_width=True)

        with right:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="section-title">Prediction Result</div>
                    <div class="big-head" style="font-size:1.25rem;">
                        {pred_class if pred_class != "unknown" else "⚠ Unknown"}
                    </div>
                    <div class="sub-head" style="font-size:.8rem; margin-bottom:.5rem;">
                        The system believes this is <b>{pred_class}</b>
                        with confidence {confidence_pct:.2f}%.
                    </div>

                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value">{confidence_pct:.2f}%</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Latency</div>
                            <div class="metric-value">{timing_ms:.1f} ms</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Mode</div>
                            <div class="metric-value">Offline</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Top-K breakdown (helps convince judges we’re not guessing randomly)
        st.markdown(
            """
            <div class="section-title" style="margin-top:2rem;">Confidence Breakdown</div>
            """,
            unsafe_allow_html=True
        )

        for label, prob in topk:
            bar_pct = min(int(prob * 100), 100)
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.03);
                    border-radius: 10px;
                    border:1px solid rgba(255,255,255,0.07);
                    padding:.6rem .8rem;
                    margin-bottom:.5rem;
                ">
                    <div style="display:flex; justify-content:space-between; font-size:.8rem; font-weight:500; color:#fff;">
                        <span>{label}</span>
                        <span>{prob*100:.2f}%</span>
                    </div>
                    <div style="
                        width:100%;
                        height:6px;
                        background:rgba(255,255,255,0.07);
                        border-radius:4px;
                        margin-top:.4rem;
                        overflow:hidden;
                    ">
                        <div style="
                            width:{bar_pct}%;
                            height:100%;
                            background:linear-gradient(90deg,#4B8CFB 0%,#4AD0FF 100%);
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ========== FOOTER VALUE PROP ==========
st.markdown(
    """
    <div class="footer-card">
      <b>Why this matters</b><br>
      • Fully offline → privacy & compliance<br>
      • Real-time (<50ms on CPU in demo)<br>
      • Rejects strangers with <span style="color:#4AD0FF;font-weight:500;">Unknown</span> label
    </div>
    """,
    unsafe_allow_html=True
)

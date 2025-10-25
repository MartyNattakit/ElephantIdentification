import os, glob, time, json, base64, pathlib
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import onnxruntime as ort
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="FormChang ID Verification",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================
# GLOBAL STYLE
# =========================================
st.markdown("""
<style>
body, .main, .block-container {
    background: #ffffff !important;
    color: #1a1a1a !important;
    font-family: -apple-system,BlinkMacSystemFont,'Inter',Roboto,sans-serif;
}

/* wrapper to limit width and center sections */
.center-wrap {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}

/* SECTION 1 (hero) */
.hero-eyebrow {
    text-align:center;
    font-size:clamp(1.8rem,1.5vw+0.8rem,2.4rem);
    font-weight:600;
    letter-spacing:-0.04em;
    background: linear-gradient(90deg,#4B5CFF 0%, #5AC8FA 100%);
    -webkit-background-clip: text;
    color: transparent;
    margin-bottom:.5rem;
}
.hero-headline {
    font-size: clamp(2.4rem,2vw+1rem,3.2rem);
    font-weight: 600;
    letter-spacing: -0.04em;
    background: linear-gradient(90deg,#4B5CFF 0%, #5AC8FA 100%);
    -webkit-background-clip: text;
    color: transparent;
    text-align:center;
    margin-bottom: .75rem;
    line-height:1.15;
}
.hero-desc {
    text-align:center;
    font-size:1.1rem;
    line-height:1.55;
    color:#444;
    max-width:700px;
    margin:0 auto;
}

/* SECTION 2 card ("Our Solution") */
.card-solution {
    background: radial-gradient(circle at 20% 20%, #ffffff 0%, #f2f4f7 80%);
    border: 1px solid rgba(0,0,0,0.05);
    border-radius: 20px;
    box-shadow: 0 30px 80px rgba(15,23,42,0.08);
    padding: 2rem 1.5rem;
    max-width: 900px;
    margin: 2rem auto 2rem auto;
}
.section-head {
    font-size:1.35rem;
    font-weight:600;
    color:#1a1a1a;
    letter-spacing:-0.03em;
    text-align:center;
    margin-bottom:.5rem;
    line-height:1.2;
}
.section-desc {
    font-size:1rem;
    line-height:1.5;
    color:#4a4a4a;
    text-align:center;
    max-width:700px;
    margin:0 auto .75rem auto;
}

/* 3D container */
.solution-3d-center {
    width:100%;
    display:flex;
    justify-content:center;
    align-items:center;
    padding:1rem 0 2rem 0;
}

/* bullets under model */
.solution-bullets {
    font-size:1rem;
    color:#4a4a4a;
    line-height:1.5;
    text-align:center;
    max-width:700px;
    margin:0.5rem auto 0 auto;
}
.solution-bullets b {
    color:#1a1a1a;
    font-weight:600;
}

/* SECTION 3 ("Try It Now") */
.card-try {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 20px;
    box-shadow: 0 30px 80px rgba(15,23,42,0.07);
    padding: 2rem 1.5rem;
    max-width: 800px;
    margin: 2rem auto;
    text-align:center;
}
.try-head {
    font-size:1.3rem;
    font-weight:600;
    color:#1a1a1a;
    letter-spacing:-0.03em;
    line-height:1.3;
}
.try-desc {
    font-size:1rem;
    color:#4a4a4a;
    line-height:1.55;
    max-width:600px;
    margin:.75rem auto 1.5rem auto;
}

/* prediction cards */
.card-predict-result {
    background:#ffffff;
    border:1px solid rgba(0,0,0,0.08);
    border-radius:16px;
    box-shadow:0 16px 40px rgba(0,0,0,0.05);
    padding:1rem 1rem;
    max-width:700px;
    margin:1.5rem auto;
}

/* SECTION 4 ("Judge Batch Evaluation") */
.card-judge {
    background: #ffffff;
    border-top: 1px solid rgba(0,0,0,0.07);
    max-width: 900px;
    margin: 2rem auto 0 auto;
    padding: 2rem 1.5rem;
    text-align:center;
}
.judge-head {
    font-size:1.2rem;
    font-weight:600;
    color:#1a1a1a;
    letter-spacing:-0.03em;
    text-align:center;
    line-height:1.3;
}
.judge-desc {
    font-size:1rem;
    color:#4a4a4a;
    line-height:1.55;
    max-width:700px;
    margin:.75rem auto 1rem auto;
    text-align:center;
}

/* footer */
.footer-bar {
    text-align:center;
    font-size:.85rem;
    color:#4a4a4a;
    padding:2rem 1rem 4rem 1rem;
    border-top:1px solid rgba(0,0,0,0.05);
    max-width:900px;
    margin:0 auto;
}
.footer-bar a {
    color:#1f4fff;
    font-weight:500;
    text-decoration:none;
}

/* streamlit button override: long + centered */
.stButton > button {
    background: linear-gradient(90deg,#4B5CFF 0%, #5AC8FA 100%);
    color: #ffffff !important;
    border: 0 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.25rem !important;
    min-width: 220px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    cursor: pointer;
    display:block;
    margin: 1rem auto 0 auto;
    text-align:center;
}
.stButton > button:hover {
    filter: brightness(0.95);
    box-shadow: 0 14px 24px rgba(0,0,0,0.18);
}
.stButton > button:active {
    filter: brightness(0.9);
}
</style>
""", unsafe_allow_html=True)


# =========================================
# MODEL + THRESHOLDS
# =========================================
MIN_FOLDER_CONF_PCT = 58.0
MIN_CONSIST_RATIO   = 0.30

ARTIFACTS_DIR = "artifacts"
EMB_ONNX      = os.path.join(ARTIFACTS_DIR, "elephant_embedding.onnx")
PROTOS_NPY    = os.path.join(ARTIFACTS_DIR, "prototypes.npy")
META_JSON     = os.path.join(ARTIFACTS_DIR, "infer_meta.json")

TEST_ROOT     = "TestData"  # expects folders 1..11
IMG_EXT       = (".jpg",".jpeg",".png",".bmp",".webp")

with open(META_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)

known_classes      = meta["known_classes"]
class_to_idx_known = meta["class_to_idx_known"]
IMG_SIZE           = meta["img_size"]
MEAN               = tuple(meta["mean"])
STD                = tuple(meta["std"])
EMB_DIM            = meta["emb_dim"]
TAU_DEFAULT        = meta.get("tau_default", 0.70)
MARGIN_RULE        = meta.get("margin_rule", 0.20)

prototypes = np.load(PROTOS_NPY).astype("float32")
assert prototypes.shape[1] == EMB_DIM, "prototype dim mismatch"

idx_to_name = {i: name for name, i in class_to_idx_known.items()}

preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

available_providers = ort.get_available_providers()
provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available_providers else available_providers[0]
session = ort.InferenceSession(
    EMB_ONNX,
    sess_options=ort.SessionOptions(),
    providers=[provider],
)

# =========================================
# 3D MODEL PREP
# =========================================
model_glb_path = pathlib.Path("model.glb")
MODEL_DATA_URL = None
if model_glb_path.exists():
    _bytes = model_glb_path.read_bytes()
    _b64 = base64.b64encode(_bytes).decode("utf-8")
    MODEL_DATA_URL = f"data:model/gltf-binary;base64,{_b64}"
SHOW_3D = MODEL_DATA_URL is not None


# =========================================
# INFERENCE HELPERS
# =========================================
def classify_image(pil_img: Image.Image):
    """run single image through onnx session, apply tau+margin."""
    inp = preprocess(pil_img.convert("RGB")).unsqueeze(0).numpy().astype("float32")

    start_t = time.time()
    ort_out = session.run(None, {"input": inp})
    end_t   = time.time()

    emb = ort_out[0]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    sims = emb @ prototypes.T
    sims = sims[0]

    order = np.argsort(-sims)
    top1 = order[0]
    top2 = order[1] if len(order) > 1 else order[0]

    score1 = float(sims[top1])
    score2 = float(sims[top2])
    margin = score1 - score2

    top1_name = idx_to_name[top1]
    tau_used  = TAU_DEFAULT

    confident_by_tau    = (score1 >= tau_used)
    confident_by_margin = (margin >= MARGIN_RULE)

    if confident_by_tau or confident_by_margin:
        final_name = top1_name
    else:
        final_name = "unknown"

    start_dt = datetime.fromtimestamp(start_t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    end_dt   = datetime.fromtimestamp(end_t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    elapsed  = end_t - start_t

    return {
        "final_name": final_name,
        "start_time": start_dt,
        "end_time": end_dt,
        "elapsed_s": f"{elapsed:.3f}",
        "score1": score1,
        "margin": margin,
        "tau_used": tau_used,
    }


def summarize_folder(folder_path: str, folder_label: str):
    """aggregate multiple frames in TestData/<id> to one final prediction."""
    img_files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        img_files.extend(glob.glob(os.path.join(folder_path, ext)))
    img_files = sorted(img_files)

    if len(img_files) == 0:
        return {
            "test_class": folder_label,
            "predicted_class": "unknown",
            "confidence_pct": 0.0,
            "start_time": "-",
            "end_time": "-",
            "elapsed_s": "0.000",
        }

    per_image_results = []
    for p in img_files:
        r = classify_image(Image.open(p).convert("RGB"))
        per_image_results.append(r)

    votes = {}
    per_name_scores = {}
    per_name_count_hi = {}

    for r in per_image_results:
        nm = r["final_name"]
        sc = r["score1"]
        if nm == "unknown":
            continue
        votes[nm] = votes.get(nm, 0) + 1
        per_name_scores.setdefault(nm, []).append(sc)
        per_name_count_hi[nm] = per_name_count_hi.get(nm, 0) + 1

    if len(votes) == 0:
        folder_pred = "unknown"
        avg_score   = float(np.mean([r["score1"] for r in per_image_results]))
    else:
        folder_pred = max(votes.items(), key=lambda kv: kv[1])[0]
        avg_score   = float(np.mean(per_name_scores[folder_pred]))
        consistent_ratio = per_name_count_hi[folder_pred] / len(per_image_results)
        folder_conf_pct  = max(min(avg_score, 1.0), -1.0) * 100.0

        if (folder_conf_pct < MIN_FOLDER_CONF_PCT) or (consistent_ratio < MIN_CONSIST_RATIO):
            folder_pred = "unknown"

    if folder_pred == "unknown":
        folder_conf_pct = max(
            min(float(np.mean([r["score1"] for r in per_image_results])), 1.0),
            -1.0
        ) * 100.0
    else:
        folder_conf_pct = max(min(avg_score, 1.0), -1.0) * 100.0

    first_res = per_image_results[0]
    last_res  = per_image_results[-1]
    total_elapsed = sum([float(r["elapsed_s"]) for r in per_image_results])

    return {
        "test_class": folder_label,
        "predicted_class": folder_pred,
        "confidence_pct": folder_conf_pct,
        "start_time": first_res["start_time"],
        "end_time": last_res["end_time"],
        "elapsed_s": f"{total_elapsed:.3f}",
    }


def run_full_evaluation_from_disk():
    """official judge output for folders 1..11"""
    blocks = []
    for i in range(1, 12):
        folder_label = str(i)
        folder_path = os.path.join(TEST_ROOT, folder_label)

        if not os.path.isdir(folder_path):
            block = (
                "[Prediction Results]\n"
                f"  Test Class: {folder_label}\n"
                "  Predicted Class: (folder not found)\n"
                "  Confidence: 0.000\n"
                "  Start Time: -\n"
                "  End Time:   -\n"
                "  Time Elapsed: 0.000\n"
                "\n"
            )
            blocks.append(block)
            continue

        summary = summarize_folder(folder_path, folder_label)
        conf_disp = f"{summary['confidence_pct']:.3f}"

        block = (
            "[Prediction Results]\n"
            f"  Test Class: {summary['test_class']}\n"
            f"  Predicted Class: {summary['predicted_class']}\n"
            f"  Confidence: {conf_disp}\n"
            f"  Start Time: {summary['start_time']}\n"
            f"  End Time:   {summary['end_time']}\n"
            f"  Time Elapsed: {summary['elapsed_s']}\n"
            "\n"
        )
        blocks.append(block)

    return "".join(blocks)


# =========================================
# SECTION 1: HERO
# =========================================
st.markdown(
    """
<div class="center-wrap" style="padding-top:3rem;padding-bottom:2rem;">
  <div class="hero-eyebrow">Team FormChang presents</div>
  <div class="hero-headline">Our solution for AI Hackathon: Elephant Identification</div>
  <div class="hero-desc">
    Real-time elephant identification and unknown detection, fully offline.<br/>
    Built for wildlife monitoring, anti-poaching checkpoints, and rescue teams.
  </div>
</div>
""",
    unsafe_allow_html=True
)


# =========================================
# SECTION 2: SOLUTION CARD
# (card header text)
# =========================================
st.markdown(
    """
<div class="card-solution">
  <div class="section-head">Our Solution</div>
  <div class="section-desc">
    We generate an embedding for each elephant using an ONNX model,
    compare that vector with known individuals using cosine similarity,
    and apply safety rules (margin, confidence, consistency across frames)
    to reject strangers.
  </div>
""",
    unsafe_allow_html=True
)

# =========================================
# 3D MODEL VIEWER (centered INSIDE the iframe)
# =========================================
if SHOW_3D:
    st.components.v1.html(
        f"""
        <div style="
            width:100%;
            max-width:900px;
            margin-left:auto;
            margin-right:auto;
            display:flex;
            justify-content:center;
            align-items:center;
            padding:1rem 0 1rem 0;
        ">
            <script type="module"
                src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js">
            </script>

            <model-viewer
                src="{MODEL_DATA_URL}"
                camera-controls
                auto-rotate
                rotation-per-second="15deg"
                exposure="1.1"
                style="
                    width:400px;
                    height:400px;
                    background:transparent;
                "
                shadow-intensity="0.8"
                shadow-softness="0.9"
            ></model-viewer>
        </div>
        """,
        height=460,  # a bit tighter so bullets sit closer
    )
else:
    st.markdown(
        "<div style='text-align:center;color:#777;font-size:.9rem;padding:1rem 0;'>3D model preview not available in this build.</div>",
        unsafe_allow_html=True
    )

# =========================================
# bullets + close card
# =========================================
st.markdown(
    """
  <div class="solution-bullets">
    • <b>Offline ONNX Runtime</b> (no internet at inference time)<br/>
    • <b>Cosine + τ + margin rule</b> for ID<br/>
    • <b>Folder-level voting</b> across frames to finalize identity (1..11 elephants test set)
  </div>
</div>
""",
    unsafe_allow_html=True
)



# =========================================
# SECTION 3: TRY IT NOW
# =========================================
st.markdown(
    """
<div class="card-try">
  <div class="try-head">Try It Now</div>
  <div class="try-desc">
    Upload one or more elephant images. We'll classify each image using
    the same pipeline used in evaluation (τ + margin → 'unknown' if not confident).
  </div>
</div>
""",
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="center-wrap">', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop elephant images here",
        type=["jpg","jpeg","png","bmp","webp"],
        accept_multiple_files=True,
        help="You can select multiple images at once."
    )

    analyze_btn = st.button("🔍 Analyze Uploaded Images")

    if analyze_btn:
        if not uploaded_files:
            st.warning("Please upload at least one image 🙏")
        else:
            st.markdown("#### Predictions", unsafe_allow_html=True)

            for file_obj in uploaded_files:
                pil_img = Image.open(file_obj).convert("RGB")
                result = classify_image(pil_img)

                conf_pct = max(min(result["score1"], 1.0), -1.0) * 100.0
                block_text = (
                    "[Prediction Results]\n"
                    f"  Test Class: -\n"
                    f"  Predicted Class: {result['final_name']}\n"
                    f"  Confidence: {conf_pct:.3f}\n"
                    f"  Start Time: {result['start_time']}\n"
                    f"  End Time:   {result['end_time']}\n"
                    f"  Time Elapsed: {result['elapsed_s']}\n"
                )

                st.markdown('<div class="card-predict-result">', unsafe_allow_html=True)
                col_img, col_txt = st.columns([1,1])
                with col_img:
                    st.image(pil_img, use_container_width=True)
                with col_txt:
                    st.code(block_text, language="text")
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close .center-wrap


# =========================================
# SECTION 4: JUDGE BATCH EVALUATION
# =========================================
st.markdown(
    """
<div class="card-judge">
  <div class="judge-head">Judge Batch Evaluation</div>
  <div class="judge-desc">
    This runs the full evaluation on folders <b>1..11</b> in
    <code>TestData/</code>, using the consistency ratio + confidence thresholds
    to output exactly one final ID per elephant. This is the format we submit.
  </div>
</div>
""",
    unsafe_allow_html=True
)

batch_btn = st.button("🚀 Run Official 1..11 Evaluation")
if batch_btn:
    report_text = run_full_evaluation_from_disk()
    st.code(report_text, language="text")


# =========================================
# FOOTER
# =========================================
st.markdown(
    """
<div class="footer-bar">
    Made with ❤️ by
    <a href="https://github.com/formchang" target="_blank">ทีมฟอร์มช้าง</a>
</div>
""",
    unsafe_allow_html=True
)

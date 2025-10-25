import time
import io
import base64
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import onnxruntime as ort

# =========================
# CONFIG ส่วนนี้เปลี่ยนชื่อโมเดล/คลาสตามของคุณ
# =========================
ONNX_MODEL_PATH = "model.onnx"  # <-- ใส่ชื่อไฟล์โมเดล .onnx ของคุณ
CLASSES = ["person_1", "person_2", "person_3", "unknown"]  # ต้องตรงกับลำดับ output ของโมเดล
UNKNOWN_CLASS_NAME = "unknown"
UNKNOWN_THRESHOLD = 0.60  # <--- ปรับเองตาม validation (0.6 = 60%)

IMG_SIZE = (224, 224)  # ต้องตรงกับตอนเทรน


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

# ========== โหลดโมเดล ONNX แค่ครั้งเดียว ==========
@st.cache_resource
def load_session():
    return ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

ort_session = load_session()


# ========== helper render โมเดล 3D ==========
def render_3d_viewer(glb_bytes: bytes, height: int = 320):
    """
    แสดงโมเดล 3D (.glb / .gltf) ในกล่องหมุนได้ ด้วย three.js
    glb_bytes: ไฟล์โมเดลแบบ bytes
    """
    b64 = base64.b64encode(glb_bytes).decode("utf-8")
    data_url = f"data:model/gltf-binary;base64,{b64}"

    html_code = f"""
    <div style="width:100%;height:{height}px;background:#0D1117;border-radius:12px;
                border:1px solid rgba(255,255,255,0.08);box-shadow:0 20px 60px rgba(0,0,0,0.6);
                position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;width:100%;height:100%;">
            <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/loaders/GLTFLoader.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>

            <canvas id="viewerCanvas" style="width:100%;height:100%;display:block;"></canvas>

            <script>
            (function() {{
                const canvas = document.getElementById('viewerCanvas');
                const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, alpha: true }});
                renderer.setPixelRatio(window.devicePixelRatio || 1);

                const scene = new THREE.Scene();
                scene.background = null;

                // กล้อง
                const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
                camera.position.set(0, 1.2, 2.5);

                // แสง
                const light1 = new THREE.DirectionalLight(0xffffff, 1.2);
                light1.position.set(2, 2, 2);
                scene.add(light1);

                const light2 = new THREE.DirectionalLight(0x55aaff, 0.6);
                light2.position.set(-2, -1, -2);
                scene.add(light2);

                const amb = new THREE.AmbientLight(0xffffff, 0.4);
                scene.add(amb);

                // controls หมุน/ซูม
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enablePan = false;
                controls.enableZoom = true;
                controls.target.set(0, 1.0, 0);

                // โหลดโมเดล 3D จาก base64
                const loader = new THREE.GLTFLoader();
                loader.load(
                    '{data_url}',
                    function(gltf) {{
                        const model = gltf.scene;
                        scene.add(model);

                        // auto center / scale
                        const box = new THREE.Box3().setFromObject(model);
                        const size = new THREE.Vector3();
                        const center = new THREE.Vector3();
                        box.getSize(size);
                        box.getCenter(center);

                        model.position.x -= center.x;
                        model.position.y -= center.y;
                        model.position.z -= center.z;

                        const maxDim = Math.max(size.x, size.y, size.z);
                        const scale = 1.5 / maxDim;
                        model.scale.setScalar(scale);

                        controls.update();
                    }},
                    undefined,
                    function(error) {{
                        console.error("GLB load error:", error);
                    }}
                );

                function resizeRenderer() {{
                    const parent = canvas.parentElement;
                    const w = parent.clientWidth;
                    const h = parent.clientHeight;
                    renderer.setSize(w, h, false);
                    camera.aspect = w / h;
                    camera.updateProjectionMatrix();
                }}

                function animate() {{
                    requestAnimationFrame(animate);
                    resizeRenderer();
                    renderer.render(scene, camera);
                }}
                animate();
            }})();
            </script>
        </div>
    </div>
    """
    components.html(html_code, height=height, scrolling=False)


# ========== inference functions ==========
def softmax(x: np.ndarray):
    # x shape: [1, num_classes] หรือ [num_classes]
    x = x - np.max(x, axis=-1, keepdims=True)  # ป้องกัน overflow
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def preprocess_image(pil_img: Image.Image):
    """
    แปลงรูปเป็นเทนเซอร์ numpy พร้อมเข้า ONNX:
    - RGB
    - resize เป็น IMG_SIZE
    - normalize ด้วย mean/std แบบ ImageNet (แก้ได้ถ้าคุณเทรนด้วยค่าอื่น)
    - CHW
    - batch dim
    """
    img = pil_img.convert("RGB").resize(IMG_SIZE)

    arr = np.array(img).astype("float32") / 255.0  # [H,W,C] float32 0..1

    # mean/std ถ้าเทรนด้วยค่าอื่นให้ปรับตรงนี้
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std  # normalize

    arr = np.transpose(arr, (2, 0, 1))      # -> [C,H,W]
    arr = np.expand_dims(arr, axis=0)       # -> [1,C,H,W]
    arr = arr.astype("float32")
    return arr


def run_inference(pil_img: Image.Image):
    """
    รันโมเดลจริงผ่าน onnxruntime
    return dict:
    {
        "pred_class": str,
        "confidence": float 0..1,
        "topk": [("class_name", prob_float), ...],
        "timing_ms": float
    }
    """
    start_t = time.time()

    # เตรียม input
    input_tensor = preprocess_image(pil_img)

    # ชื่อ input/output ของโมเดล
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # รัน ONNX
    logits = ort_session.run(
        [output_name],
        {input_name: input_tensor}
    )[0]  # [1, num_classes]

    # softmax -> probs
    probs = softmax(logits)[0]  # [num_classes]

    # หา class ที่คะแนนสูงสุด
    top_idx = int(np.argmax(probs))
    pred_class = CLASSES[top_idx]
    confidence = float(probs[top_idx])

    # ถ้าความมั่นใจต่ำกว่า threshold -> บังคับ unknown
    if pred_class != UNKNOWN_CLASS_NAME and confidence < UNKNOWN_THRESHOLD:
        pred_class = UNKNOWN_CLASS_NAME

    # top-k 3 อันดับ
    order = np.argsort(-probs)
    topk = []
    for i in order[: min(3, len(CLASSES))]:
        topk.append((CLASSES[i], float(probs[i])))

    end_t = time.time()
    timing_ms = (end_t - start_t) * 1000.0

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "topk": topk,
        "timing_ms": timing_ms
    }


# =========================
# UI เริ่มต้น ส่วนหัว
# =========================
st.markdown(
    """
    <div class="section-title">Identity Check AI</div>
    <div class="big-head">Real-time identity verification.<br>Private. Offline. Judge-ready.</div>
    <div class="sub-head">
      ใส่ภาพใบหน้า แล้วระบบจะระบุว่าเป็นใคร หรือจัดเป็น <b>Unknown</b><br>
      รันออฟไลน์ในเครื่องคุณ ความเป็นส่วนตัวสูง
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Sidebar: อัปโหลดโมเดล 3D
# =========================
with st.sidebar:
    st.markdown(
        "<div class='section-title'>3D Model (optional)</div>"
        "<div class='sub-head' style='font-size:.8rem;'>"
        "อัปโหลดไฟล์ .glb / .gltf เพื่อนำเสนอฮาร์ดแวร์/กล่องอุปกรณ์/หัวหุ่น ฯลฯ "
        "จะแสดง viewer หมุนได้ข้างผลทำนาย"
        "</div>",
        unsafe_allow_html=True
    )

    model3d_file = st.file_uploader(
        "Upload 3D model (.glb / .gltf)",
        type=["glb", "gltf"],
        help="ไฟล์จะแสดงใน 3D viewer",
        key="model3d_uploader"
    )

    model3d_bytes = None
    if model3d_file is not None:
        model3d_bytes = model3d_file.read()

# =========================
# Tabs สำหรับใส่รูป
# =========================
tabs = st.tabs(["📤 Upload Image", "📷 Live Camera"])

with tabs[0]:
    uploaded_file = st.file_uploader(
        "อัปโหลดรูปหน้า (JPG/PNG)",
        type=["jpg","jpeg","png"],
        help="หน้าเต็มหรือกึ่งหน้าตรงชัด ๆ จะดีที่สุด"
    )

with tabs[1]:
    st.write("ถ่ายรูปจากกล้อง (ทดลอง)")
    cam_img = st.camera_input("Take a photo")

# เลือก source: กล้องมาก่อน ถ้าไม่มีค่อยดูอัปโหลด
img_source = None
if tabs[1] and cam_img is not None:
    img_source = cam_img
elif tabs[0] and uploaded_file is not None:
    img_source = uploaded_file

# =========================
# ปุ่ม RUN
# =========================
col_run, _ = st.columns([1,3])
with col_run:
    run_btn = st.button("🔍 Analyze Now", use_container_width=True)

# =========================
# โซนผลลัพธ์
# =========================
if run_btn:
    if img_source is None:
        st.warning("กรุณาใส่รูปภาพก่อน 🙏")
    else:
        # แปลงไฟล์อัปโหลดเป็น PIL.Image
        if isinstance(img_source, io.BytesIO) or hasattr(img_source, "read"):
            pil_img = Image.open(img_source)
        else:
            pil_img = Image.open(img_source)

        # รันโมเดล
        result = run_inference(pil_img)

        pred_class = result["pred_class"]
        confidence_pct = result["confidence"] * 100.0
        timing_ms = result["timing_ms"]
        topk = result["topk"]

        # ========= LAYOUT สองคอลัมน์: รูป+สรุป / โมเดล 3D =========
        colA, colB = st.columns([1,1])

        with colA:
            # รูปอินพุต
            st.image(pil_img, caption="Input Image", use_column_width=True)

            # การ์ดผลทำนาย
            st.markdown(
                f"""
                <div class="result-card" style="margin-top:1rem;">
                    <div class="section-title">Prediction Result</div>
                    <div class="big-head" style="font-size:1.25rem;">
                        {pred_class if pred_class != "unknown" else "⚠ Unknown"}
                    </div>
                    <div class="sub-head" style="font-size:.8rem; margin-bottom:.5rem;">
                        ระบบประเมินว่าเป็น <b>{pred_class}</b>
                        ด้วยความมั่นใจ {confidence_pct:.2f}%.
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

        with colB:
            st.markdown(
                """
                <div class="section-title">3D Prototype</div>
                <div class="sub-head" style="font-size:.8rem;margin-bottom:.5rem;">
                    หมุน / ซูม โมเดลฮาร์ดแวร์หรือโครงสร้างอุปกรณ์ที่ใช้รัน AI
                </div>
                """,
                unsafe_allow_html=True
            )

            if model3d_bytes is not None:
                render_3d_viewer(model3d_bytes, height=320)
            else:
                st.markdown(
                    """
                    <div style="
                        background: rgba(255,255,255,0.03);
                        border-radius: 12px;
                        border:1px solid rgba(255,255,255,0.08);
                        padding:1rem;
                        font-size:.8rem;
                        color:#9BA3B4;
                        text-align:center;
                        box-shadow:0 20px 60px rgba(0,0,0,0.6);
                    ">
                        ยังไม่มีโมเดล 3D<br>
                        อัปโหลดไฟล์ .glb หรือ .gltf ใน Sidebar
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ========= Top-K breakdown =========
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
      • ออฟไลน์ทั้งหมด → privacy / compliance สูง<br>
      • Real-time (~<50ms CPU demo)<br>
      • ถ้าไม่ใช่ใครในระบบ เราจะติดป้าย <span style="color:#4AD0FF;font-weight:500;">Unknown</span>
    </div>
    """,
    unsafe_allow_html=True
)

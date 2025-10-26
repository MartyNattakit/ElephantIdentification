import os, time, json, base64, pathlib, glob
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import streamlit as st
import onnxruntime as ort
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pandas as pd  # ตารางโหมดหลายรูป

# =========================================
# STREAMLIT PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Elephant ID Demo",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# CONSTANTS FOR BATCH JUDGE (เหมือนฝั่งขาว)
# =========================================
MIN_FOLDER_CONF_PCT = 58.0      # step กรองโฟลเดอร์
MIN_CONSIST_RATIO   = 0.30      # อย่างน้อยรูปส่วนหนึ่งต้องเห็นเหมือนกัน
TEST_ROOT           = "TestData"
IMG_EXTS            = (".jpg",".jpeg",".png",".bmp",".webp")

# =========================================
# 3D MODEL CONFIG (หน้า hero)
# =========================================
MODEL_URL_FALLBACK = "https://modelviewer.dev/shared-assets/models/Astronaut.glb"

USE_LOCAL = True  # ถ้าคุณมีไฟล์ .glb ของตัวเอง
LOCAL_GLTF_PATH = Path(r"C:\Users\MartyNattakit\Desktop\elewebsite\model.glb")

def to_data_url(p: Path) -> str:
    """
    เอาไฟล์ .glb มาเป็น data:URL base64
    """
    mime = "model/gltf-binary"
    b = p.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"

if USE_LOCAL and LOCAL_GLTF_PATH.exists():
    model_src = to_data_url(LOCAL_GLTF_PATH)
else:
    model_src = MODEL_URL_FALLBACK


# =========================================
# INFERENCE MODEL CONFIG
# =========================================
ARTIFACTS_DIR = "artifacts"
EMB_ONNX      = os.path.join(ARTIFACTS_DIR, "elephant_embedding.onnx")
PROTOS_NPY    = os.path.join(ARTIFACTS_DIR, "prototypes.npy")
META_JSON     = os.path.join(ARTIFACTS_DIR, "infer_meta.json")

with open(META_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)

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
provider = (
    "CUDAExecutionProvider"
    if "CUDAExecutionProvider" in available_providers
    else available_providers[0]
)

session = ort.InferenceSession(
    EMB_ONNX,
    sess_options=ort.SessionOptions(),
    providers=[provider],
)

# =========================================
# CORE INFERENCE HELPERS
# =========================================
def classify_image(pil_img: Image.Image):
    """
    - run ONNX
    - หา similarity
    - ตัดสินชื่อสุดท้าย
    - คืนข้อมูลพร้อม timestamp
    """
    inp = preprocess(pil_img.convert("RGB")).unsqueeze(0).numpy().astype("float32")

    start_t = time.time()
    ort_out = session.run(None, {"input": inp})
    end_t   = time.time()

    # timestamp สไตล์รีพอร์ต
    start_dt = datetime.fromtimestamp(start_t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    end_dt   = datetime.fromtimestamp(end_t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    emb = ort_out[0]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    sims = emb @ prototypes.T  # cosine-ish sim
    sims = sims[0]

    order = np.argsort(-sims)
    top1 = order[0]
    top2 = order[1] if len(order) > 1 else order[0]

    score1 = float(sims[top1])
    score2 = float(sims[top2])
    margin = score1 - score2

    best_name = idx_to_name[top1]

    conf_pct = max(min(score1, 1.0), -1.0) * 100.0  # ประมาณเปอร์เซ็นต์ความมั่นใจ

    confident_by_tau        = (score1 >= TAU_DEFAULT)
    confident_by_margin     = (margin >= MARGIN_RULE)
    confident_by_threshold  = (conf_pct >= 40.0)

    if (confident_by_tau or confident_by_margin) and confident_by_threshold:
        final_name = best_name
    else:
        final_name = "unknown"

    elapsed  = end_t - start_t

    return {
        "final_name": final_name,                # predicted class / unknown
        "conf_pct": conf_pct,                    # ความมั่นใจ (%)
        "elapsed_s": f"{elapsed:.3f}",           # เวลาประมวลผล
        "start_time": start_dt,                  # timestamp start
        "end_time": end_dt,                      # timestamp end
        "score1": score1,                        # raw sim score (0-1ish)
    }


def summarize_folder(folder_path: str, folder_label: str):
    """
    รวมหลายรูปในโฟลเดอร์เดียวกัน (เช่น TestData/1) -> ให้ผลทายสุดท้าย 1 ตัว
    ใช้ logic consistency ratio + avg confidence คล้ายฝั่งสีขาว
    """
    # หาไฟล์รูปทั้งหมดในโฟลเดอร์
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

    # นับโหวตของชื่อที่โมเดลไม่มองว่า unknown
    votes = {}
    per_name_scores = {}
    per_name_counts = {}

    for r in per_image_results:
        nm = r["final_name"]
        sc = r["score1"]
        if nm == "unknown":
            continue
        votes[nm] = votes.get(nm, 0) + 1
        per_name_scores.setdefault(nm, []).append(sc)
        per_name_counts[nm] = per_name_counts.get(nm, 0) + 1

    # ตัดสินชื่อโฟลเดอร์
    if len(votes) == 0:
        folder_pred = "unknown"
        avg_score   = float(np.mean([r["score1"] for r in per_image_results]))
    else:
        # เอาชื่อที่ถูกโหวตเยอะสุด
        folder_pred = max(votes.items(), key=lambda kv: kv[1])[0]
        avg_score   = float(np.mean(per_name_scores[folder_pred]))

        consistent_ratio = per_name_counts[folder_pred] / len(per_image_results)
        folder_conf_pct  = max(min(avg_score, 1.0), -1.0) * 100.0

        # ถ้าความมั่นใจเฉลี่ยหรือตัว consistency ต่ำกว่ากฎ -> ขยับเป็น unknown
        if (folder_conf_pct < MIN_FOLDER_CONF_PCT) or (consistent_ratio < MIN_CONSIST_RATIO):
            folder_pred = "unknown"

    # confidence สุดท้าย
    if folder_pred == "unknown":
        folder_conf_pct = max(
            min(float(np.mean([r["score1"] for r in per_image_results])), 1.0),
            -1.0
        ) * 100.0
    else:
        folder_conf_pct = max(min(avg_score, 1.0), -1.0) * 100.0

    # เวลา
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
    """
    Loop ผ่านโฟลเดอร์ TestData/1 .. TestData/11
    พ่นรายงาน text block แบบ [Prediction Results] ต่อกันยาว ๆ
    """
    blocks = []
    for i in range(1, 12):  # 1..11
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
# HERO + CONTENT (iframe theme dark)
# =========================================
landing_html = r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    :root {
      --target-x: 26vw;
      --target-y: -60vh;
      --target-scale: 0.55;
      --hero-height: 100vh;
      --fade-dur: 1000ms;
      --slide-dur: 800ms;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, "IBM Plex Sans Thai", sans-serif;
      color: #f8fafc;
      background: #0f172a;
    }

    .hero {
      position: relative;
      height: var(--hero-height);
      overflow: hidden;
      background: radial-gradient(circle at 50% 20%, rgba(255,255,255,0.08) 0%, rgba(15,23,42,0) 70%);
      color:#fff;
    }

    .three-wrap {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      display: grid;
      place-items: center;
      transform: translate3d(0,-20vh,0) scale(1.1);
      transition: transform 300ms linear;
    }

    model-viewer {
      width: min(110vmin, 1200px);
      height: min(110vmin, 1200px);
      filter: drop-shadow(0 24px 48px rgba(0,0,0,0.65));
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(
        60% 60% at 50% 60%,
        rgba(15,23,42,0) 0%,
        rgba(15,23,42,0.6) 60%,
        rgba(15,23,42,0.9) 90%
      );
      z-index: 1;
      pointer-events: none;
    }

    .hero-content {
      position: relative;
      z-index: 2;
      height: 100%;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 24px;
      color:#fff;
    }

    .kicker {
      font-size: clamp(14px, 1.8vw, 18px);
      letter-spacing: .08em;
      text-transform: uppercase;
      opacity: 0;
      transform: translateY(10px);
      animation: fadeUp var(--fade-dur) ease 200ms forwards;

      text-shadow:
        0 1px 0 rgba(255,255,255,0.8),
        0 2px 8px rgba(0,0,0,0.8);
      color:#fff;
    }

    .title {
      margin-top: 8px;
      line-height: 1.05;
      font-weight: 800;
      font-size: clamp(32px, 6.2vw, 72px);
      opacity: 0;
      transform: translateY(12px);
      animation: fadeUp var(--fade-dur) ease 400ms forwards;
      color: #fff;

      text-shadow:
        0 2px 4px rgba(0,0,0,0.9),
        0 10px 32px rgba(0,0,0,0.9);
    }

    .subtitle {
      margin-top: 12px;
      font-size: clamp(14px, 2vw, 18px);
      opacity: 0;
      transform: translateY(14px);
      animation: fadeUp var(--fade-dur) ease 600ms forwards;
      color: #d1d5db;
      max-width: 860px;
      margin-left: auto;
      margin-right: auto;
      line-height: 1.5;
      text-shadow:
        0 1px 2px rgba(0,0,0,0.9),
        0 8px 24px rgba(0,0,0,0.8);
    }

    .cta-upload {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;

      margin-top: 32px;

      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #fff;

      font-weight: 600;
      font-size: clamp(16px, 2vw, 20px);
      line-height: 1.2;

      padding: 1.1rem 1.5rem;
      border-radius: 16px;
      min-width: min(280px, 80vw);
      text-align: center;
      text-decoration: none;

      box-shadow: 0 24px 60px rgba(118,75,162,0.7);
      border: 1px solid rgba(255,255,255,0.25);

      transition: all .18s ease;
      cursor: pointer;
    }
    .cta-upload:hover {
      transform: translateY(-3px) scale(1.03);
      box-shadow: 0 30px 80px rgba(118,75,162,0.9);
      filter: brightness(1.07);
    }

    .scroll-indicator {
      position: absolute;
      bottom: 24px;
      left: 0;
      right: 0;
      z-index: 2;
      display: grid;
      place-items: center;
      font-size: 12px;
      color: #9ca3af;
      opacity: .8;
      animation: floaty 2.4s ease-in-out infinite;
      text-transform: uppercase;
      letter-spacing: .1em;
    }

    @keyframes fadeUp {
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes floaty {
      0%,100% { transform: translateY(0); }
      50%     { transform: translateY(-6px); }
    }

    .content {
      position: relative;
      z-index: 3;
      background: radial-gradient(circle at 20% 0%,
        rgba(30,58,138,0.6) 0%,
        rgba(15,23,42,1) 60%);
      background-color:#0f172a;
      color:#fff;
    }

    .section {
      min-height: 30vh;
      padding: clamp(24px, 4vw, 48px) clamp(16px, 6vw, 48px);
      display: grid;
      align-items: center;
    }

    .card {
      max-width: 980px;
      margin: 0 auto;
      background: rgba(15,23,42,0.8);
      -webkit-backdrop-filter: blur(8px);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(148,163,184,0.18);
      border-radius: 20px;
      padding: clamp(16px, 3.5vw, 36px);
      box-shadow: 0 30px 80px rgba(0,0,0,0.8);

      opacity: 0;
      transform: translateY(30px);
      transition:
        opacity var(--slide-dur) ease,
        transform var(--slide-dur) ease,
        max-height 600ms ease,
        padding-bottom 600ms ease;
      overflow: hidden;
      color:#f8fafc;
    }

    .card.visible {
      opacity: 1;
      transform: translateY(0);
    }

    .card .details {
      max-height: 0;
      opacity: 0;
      transition: max-height 600ms ease, opacity 600ms ease;
    }
    .card.expanded .details {
      max-height: 400px;
      opacity: 1;
    }

    .card h3 {
      margin: 0 0 8px 0;
      font-size: clamp(22px, 3.8vw, 36px);
      color: #fff;
      font-weight: 700;
      line-height: 1.2;
      text-shadow:0 2px 4px rgba(0,0,0,0.8);
    }
    .card p {
      margin: 0;
      font-size: clamp(14px, 2.2vw, 18px);
      color: #cbd5e1;
      line-height: 1.6;
      font-weight: 400;
    }

    .hint-upload {
      font-size: 0.9rem;
      color: #94a3b8;
      margin-top: 1rem;
      line-height: 1.5;
      background: rgba(15,23,42,0.6);
      border: 1px solid rgba(148,163,184,0.25);
      border-radius: 12px;
      padding: 12px 16px;
    }
  </style>
</head>
<body>
  <div class="hero">
    <div class="three-wrap" id="threeWrap">
      <model-viewer
        src="MODEL_SRC_HERE"
        alt="3D Model"
        exposure="0.9"
        camera-controls
        disable-zoom
        auto-rotate
        rotation-per-second="20deg"
        interaction-prompt="none"
        ar
        shadow-intensity="0.8">
      </model-viewer>
    </div>

    <div class="hero-content">
      <div>
        <div class="kicker">Team FormChang presents</div>
        <div class="title">
          Elephant Identification<br/>With Vision Transformer
        </div>
        <div class="subtitle">
          โมเดลเราสามารถบอกได้ว่าช้างในภาพคือ "ตัวไหน" ในฐานข้อมูลอนุรักษ์
          โดยดูจุดเด่นบนร่างกาย เช่น รอยแผล หู งวง โหนกหน้าผาก.<br/>
          ทำงานออฟไลน์ได้ เหมาะกับผู้พิทักษ์ป่าหน้างาน ไม่มีเน็ตก็ยังใช้ได้.
        </div>

        <a class="cta-upload"
           href="#"
           onclick="window.parent.document.getElementById('real-upload-host').scrollIntoView({behavior:'smooth'}); return false;">
          UPLOAD
        </a>
      </div>
    </div>

    <div class="scroll-indicator">Scroll to explore ↓</div>
  </div>

  <main class="content">
    <section class="section">
      <div class="card" id="card-intro">
        <h3>1) แนวคิด</h3>
        <p>
          เราพยายามแก้ปัญหา "ช้างตัวนี้คือใคร" โดยดูลักษณะเฉพาะ
          เช่น รอยแผลฉีกตรงหู โครงหน้างวง แผลเป็นตามตัว
          แล้วจับคู่กับช้างที่รู้จักในฐานข้อมูล.
        </p>

        <div class="details">
          <p style="margin-top:16px;">
            จุดสำคัญคือระบบรันได้ในสนามป่า: แค่ถ่ายรูปจากมือถือ
            → วิ่งโมเดล ONNX ในเครื่อง
            → ได้ชื่อช้างกับความมั่นใจ
            โดยไม่ต้องอัปโหลดขึ้นคลาวด์.
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card" id="card-upload">
        <h3>2) อัปโหลดรูปเพื่อตรวจสอบ ID</h3>
        <p>
          ตอนเลื่อนลง คุณสามารถอัปโหลดทั้ง "รูปเดียวแบบละเอียด"
          หรือ "หลายรูปพร้อมกัน" แล้วจะได้ผลสรุปในตาราง.
        </p>
        <div class="hint-upload">
          ⬇ ด้านล่าง (ในหน้าเดียวนี้) จะมีโหมดวิเคราะห์เดี่ยวและโหมดกลุ่ม
          พร้อมผลวิเคราะห์สดด้วยโมเดล ONNX
        </div>
      </div>
    </section>

    <section class="section">
      <div class="card" id="card-report">
        <h3>3) รายงานผลการทำนาย</h3>
        <p>
          สำหรับรูปเดียว จะมีรายงานสไตล์ <code>[Prediction Results]</code><br/>
          สำหรับหลายรูป จะมีตารางรวม: ไฟล์ไหน → ช้างตัวไหน → ความมั่นใจ → เวลาประมวลผล
        </p>
      </div>
    </section>
  </main>

  <script>
    function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
    function splitUnit(vStr){
      const m = vStr.trim().match(/^(-?[0-9.]+)(.*)$/);
      return {
        num: m ? parseFloat(m[1]) : 0,
        unit: m ? (m[2] || "") : ""
      };
    }

    const threeWrap = document.getElementById('threeWrap');
    const hero = document.querySelector('.hero');

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('visible');
          if (e.target.id === "card-intro") {
            e.target.classList.add('expanded');
          }
        }
      });
    }, { threshold: 0.18 });

    document.querySelectorAll('.card').forEach(el => observer.observe(el));

    function onScroll(){
      const heroRectH = hero.getBoundingClientRect().height || window.innerHeight;
      const y = window.scrollY || window.pageYOffset;
      let p = y / heroRectH;
      p = clamp(p, 0, 1);

      const styles = getComputedStyle(document.documentElement);
      const targetXRaw = styles.getPropertyValue('--target-x');
      const targetYRaw = styles.getPropertyValue('--target-y');
      const targetScale = parseFloat(styles.getPropertyValue('--target-scale'));

      const X = splitUnit(targetXRaw);
      const Y = splitUnit(targetYRaw);

      const startScale = 1.1;
      const curX = (X.num * p) + X.unit;
      const curY = (Y.num * p) + Y.unit;
      const curScale = startScale + (targetScale - startScale) * p;

      threeWrap.style.transform =
        `translate3d(${curX}, ${curY}, 0) scale(${curScale})`;
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  </script>
</body>
</html>
"""

# bind model src
landing_final = landing_html.replace("MODEL_SRC_HERE", model_src)

# render hero + animated scroll sections (iframe)
st.components.v1.html(landing_final, height=1800, scrolling=True)

# =========================================
# REAL UPLOADER CARD (โหมดเดี่ยว + โหมดกลุ่ม)
# =========================================
st.markdown(
    """
    <div id="real-upload-host" style="
        max-width:1000px;
        margin:4rem auto 2rem auto;
        background:rgba(15,23,42,0.8);
        -webkit-backdrop-filter:blur(8px);
        backdrop-filter:blur(8px);
        border:1px solid rgba(148,163,184,0.25);
        border-radius:20px;
        box-shadow:0 40px 120px rgba(0,0,0,0.9);
        padding:2rem 2rem 2.5rem 2rem;
        font-family:-apple-system,BlinkMacSystemFont,'Inter',Roboto,'IBM Plex Sans Thai',sans-serif;
        color:#f8fafc;
    ">

      <div style="
        font-size:clamp(22px,3.8vw,32px);
        font-weight:700;
        color:#fff;
        line-height:1.2;
        margin:0 0 .5rem 0;
        text-shadow:0 2px 4px rgba(0,0,0,0.8);
        display:flex;
        align-items:center;
        gap:.5rem;
        flex-wrap:wrap;
      ">
        <span style="font-size:1.4rem;">🔍</span>
        <span>วิเคราะห์รูปช้าง</span>
        <span style="font-weight:400;color:#94a3b8;font-size:.9rem;">(โหมดเดี่ยว & โหมดกลุ่ม)</span>
      </div>

      <div style="
        font-size:.95rem;
        color:#94a3b8;
        line-height:1.6;
        margin:0 0 1.75rem 0;
      ">
        โหมด A: อัปโหลดทีละรูป → ได้รายงานละเอียดแบบ [Prediction Results]<br/>
        โหมด B: อัปโหลดหลายรูปพร้อมกัน → ได้ตารางสรุปช้างที่คาดว่าเป็น / ความมั่นใจ / เวลา infer
      </div>
    """,
    unsafe_allow_html=True
)

# ---------- โหมด A: วิเคราะห์รูปเดียว ----------
st.markdown(
    "<div style='color:#fff;font-weight:600;font-size:1.05rem;margin-bottom:.5rem;'>โหมด A · วิเคราะห์รูปเดียว (รายงานละเอียด)</div>",
    unsafe_allow_html=True
)

single_file = st.file_uploader(
    "เลือกภาพช้าง 1 รูป (JPG/PNG/WEBP ฯลฯ)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=False,
    key="single_file_uploader_dark",
    help="เลือกรูปช้างให้เห็นหัว/งวง/หูชัดที่สุด"
)

run_single_btn = st.button("วิเคราะห์รูปนี้เลย", use_container_width=True, key="run_single_btn_dark")

if single_file and run_single_btn:
    pil_img = Image.open(single_file).convert("RGB")
    result = classify_image(pil_img)

    file_name   = single_file.name
    pred_name   = result["final_name"]
    conf_val    = result["conf_pct"]
    start_time  = result["start_time"]
    end_time    = result["end_time"]
    elapsed_s   = result["elapsed_s"]

    # block text สไตล์ benchmark
    report_text = (
        "[Prediction Results]\n"
        f"  Test Class: {file_name}\n"
        f"  Predicted Class: {pred_name}\n"
        f"  Confidence: {conf_val:.3f}\n"
        f"  Start Time: {start_time}\n"
        f"  End Time:   {end_time}\n"
        f"  Time Elapsed: {elapsed_s}\n"
    )

    col_left, col_right = st.columns([1,1], gap="large")
    with col_left:
        st.image(pil_img, use_container_width=True, caption=file_name)
    with col_right:
        st.markdown(
            "<div style='background:#1e2538;border:1px solid #475569;border-radius:12px;padding:1rem;color:#fff;'>",
            unsafe_allow_html=True
        )
        st.code(report_text, language="text")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<hr style='border:0;border-top:1px solid rgba(148,163,184,0.3);margin:2rem 0 1.5rem 0;'/>",
    unsafe_allow_html=True
)

# ---------- โหมด B: วิเคราะห์หลายรูปพร้อมกัน ----------
st.markdown(
    "<div style='color:#fff;font-weight:600;font-size:1.05rem;margin-bottom:.5rem;'>โหมด B · วิเคราะห์หลายรูปพร้อมกัน (ตารางสรุป)</div>",
    unsafe_allow_html=True
)

multi_files = st.file_uploader(
    "เลือกหลายรูป (ลากหลายไฟล์พร้อมกันได้)",
    type=["jpg","jpeg","png","bmp","webp"],
    accept_multiple_files=True,
    key="multi_file_uploader_dark",
    help="กด Ctrl / Shift เพื่อเลือกหลายรูป"
)

run_batch_local_btn = st.button("วิเคราะห์รูปทั้งหมดในชุดนี้", use_container_width=True, key="run_batch_local_btn_dark")

if multi_files and run_batch_local_btn:
    rows = []
    thumbs = []

    for f in multi_files:
        pil_img = Image.open(f).convert("RGB")
        res = classify_image(pil_img)

        rows.append({
            "ไฟล์": f.name,
            "ช้างที่คาดว่าเป็น": res["final_name"],
            "ความมั่นใจ (%)": f"{res['conf_pct']:.1f}",
            "เวลา infer (วินาที)": res["elapsed_s"],
        })

        thumbs.append((pil_img, f.name))

    df_batch = pd.DataFrame(rows)
    st.dataframe(df_batch, use_container_width=True, hide_index=True)

    st.markdown(
        "<div style='color:#fff;font-weight:600;font-size:1rem;margin-top:2rem;margin-bottom:.5rem;'>ตัวอย่างรูปในชุดนี้</div>",
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    for i, (img_pil, nm) in enumerate(thumbs):
        with cols[i % 3]:
            st.image(img_pil, caption=nm, use_container_width=True)

# ปิด uploader card
st.markdown("</div>", unsafe_allow_html=True)


# =========================================
# SECTION 4: JUDGE BATCH EVALUATION (โฟลเดอร์ 1..11)
# =========================================
st.markdown(
    """
    <div style="
        max-width:1000px;
        margin:2rem auto 0 auto;
        background:rgba(15,23,42,0.6);
        -webkit-backdrop-filter:blur(6px);
        backdrop-filter:blur(6px);
        border:1px solid rgba(148,163,184,0.2);
        border-radius:20px;
        box-shadow:0 30px 80px rgba(0,0,0,0.9);
        padding:2rem 2rem;
        color:#f8fafc;
        font-family:-apple-system,BlinkMacSystemFont,'Inter',Roboto,'IBM Plex Sans Thai',sans-serif;
        text-align:center;
    ">
      <div style="
          font-size:1.2rem;
          font-weight:600;
          line-height:1.3;
          color:#fff;
          text-shadow:0 2px 4px rgba(0,0,0,0.8);
          margin-bottom:.5rem;
      ">
        Judge Batch Evaluation
      </div>

      <div style="
          font-size:.95rem;
          color:#94a3b8;
          line-height:1.55;
          max-width:700px;
          margin:0 auto 1rem auto;
      ">
        รันการประเมินเต็มรูปแบบในโฟลเดอร์ <b>1..11</b> ภายใต้ <code>TestData/</code><br/>
        เราจะใช้ consistency ratio + confidence threshold<br/>
        เพื่อสรุป "ช้างตัวสุดท้าย" ต่อหนึ่งโฟลเดอร์ (รูปหลายมุมของช้างตัวเดียว)
      </div>
    """,
    unsafe_allow_html=True
)

judge_btn = st.button("🚀 Run Official 1..11 Evaluation", use_container_width=True)

if judge_btn:
    report_text = run_full_evaluation_from_disk()
    st.markdown(
        "<div style='max-width:1000px;margin:1rem auto 0 auto;'>",
        unsafe_allow_html=True
    )
    st.code(report_text, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

# ปิดการ์ด judge
st.markdown("</div>", unsafe_allow_html=True)


# =========================================
# FOOTER (ธีมมืด)
# =========================================
st.markdown(
    """
    <div style="
        max-width:1000px;
        margin:3rem auto 4rem auto;
        padding-top:1.5rem;
        border-top:1px solid rgba(148,163,184,0.2);
        text-align:center;
        font-family:-apple-system,BlinkMacSystemFont,'Inter',Roboto,'IBM Plex Sans Thai',sans-serif;
        color:#94a3b8;
        font-size:.85rem;
        line-height:1.5;
    ">
      Made with ❤️ by
      <a href="https://github.com/formchang" target="_blank"
         style="color:#60a5fa;text-decoration:none;font-weight:500;">
         ทีมฟอร์มช้าง
      </a>
    </div>
    """,
    unsafe_allow_html=True
)

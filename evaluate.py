import os, glob, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

import onnxruntime as ort
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

MIN_FOLDER_CONF_PCT = 75   
MIN_CONSIST_RATIO   = 0.30   
ARTIFACTS_DIR = "artifacts"
EMB_ONNX      = os.path.join(ARTIFACTS_DIR, "elephant_embedding.onnx")
PROTOS_NPY    = os.path.join(ARTIFACTS_DIR, "prototypes.npy")
META_JSON     = os.path.join(ARTIFACTS_DIR, "infer_meta.json")

TEST_ROOT     = r"Main_task_M1"   


#################################
# LOAD ARTIFACTS
#################################

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

#centroid
prototypes = np.load(PROTOS_NPY).astype("float32")  
assert prototypes.shape[1] == EMB_DIM, f"prototypes dim {prototypes.shape} != {EMB_DIM}"

idx_to_name = {i: name for name, i in class_to_idx_known.items()}

# preprocess → tensor → normalize → numpy float32
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# onnxruntime session
available_providers = ort.get_available_providers()
provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available_providers else available_providers[0]
sess_options = ort.SessionOptions()
session = ort.InferenceSession(EMB_ONNX, sess_options, providers=[provider])


#################################
# HELPERS
#################################

def classify_image(pil_img):

    inp = preprocess(pil_img).unsqueeze(0).numpy().astype("float32")  # [1,3,H,W]

    start_t = time.time()
    ort_out = session.run(None, {"input": inp})
    end_t   = time.time()

    emb = ort_out[0]                   # [1, EMB_DIM]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    sims = emb @ prototypes.T          # [1, C]
    sims = sims[0]                     # [C]

    order = np.argsort(-sims)
    top1 = order[0]
    top2 = order[1] if len(order) > 1 else top1

    score1 = float(sims[top1])
    score2 = float(sims[top2])
    margin = score1 - score2

    top1_name = idx_to_name[top1]
    tau_used  = TAU_DEFAULT      # <-- ตรงนี้มาจาก meta["tau_default"], ปัจจุบัน = 0.70
    confident_by_tau    = (score1 >= tau_used)
    confident_by_margin = (margin >= MARGIN_RULE)  # margin คือ score1 - score2
    if confident_by_tau or confident_by_margin:
        final_name = top1_name
    else:
        final_name = "unknown"

    # format timing -> string millisecond
    start_dt = datetime.fromtimestamp(start_t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    end_dt   = datetime.fromtimestamp(end_t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    elapsed  = end_t - start_t
    elapsed_s= f"{elapsed:.3f}"
    

    return {
        "final_name": final_name,
        "top1_name": top1_name,
        "score1": score1,
        "score2": score2,
        "margin": margin,
        "tau_used": tau_used,
        "start_time": start_dt,
        "end_time": end_dt,
        "elapsed_s": elapsed_s,
    }


def summarize_folder(folder_path, folder_label):
    results_each_img = []

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

    # infer ทุกรูป
    for p in img_files:
        pil_img = Image.open(p).convert("RGB")
        r = classify_image(pil_img)
        results_each_img.append(r)

    # -----------------------
    # STEP 1: vote เฉพาะผลที่ 'final_name' != "unknown"
    # -----------------------
    votes = {}
    per_name_scores = {}   # เก็บ score1 ของรูปที่ออกชื่อช้างนั้น
    per_name_count_hi = {} # นับรูปที่ "มั่นใจสูง" (ผ่าน rule เป็นช้าง ไม่ใช่ unknown)

    for r in results_each_img:
        nm = r["final_name"]
        sc = r["score1"]          # cosine ~ [0..1]
        if nm == "unknown":
            continue
        votes[nm] = votes.get(nm, 0) + 1
        per_name_scores.setdefault(nm, []).append(sc)
        per_name_count_hi[nm] = per_name_count_hi.get(nm, 0) + 1

    # ถ้าไม่มีชื่อช้างเลย -> โฟลเดอร์ unknown
    if len(votes) == 0:
        folder_pred = "unknown"
        avg_score   = float(np.mean([r["score1"] for r in results_each_img])) if results_each_img else 0.0
    else:
        # class ที่ถูกโหวตเยอะสุด
        folder_pred = max(votes.items(), key=lambda kv: kv[1])[0]
        # ค่าเฉลี่ย cosine ของรูปที่โหวตเป็นชื่อนั้น
        avg_score   = float(np.mean(per_name_scores[folder_pred]))

        # consistency: สัดส่วนรูปที่สรุปเป็นชื่อเดียวกัน
        consistent_ratio = per_name_count_hi[folder_pred] / len(results_each_img)

        # แปลง avg_score -> %
        folder_conf_pct = max(min(avg_score, 1.0), -1.0) * 100.0

        # RULE OVERRIDE:
        # ถ้าไม่ผ่าน threshold ทั้งสองอย่าง => unknown
        if (folder_conf_pct < MIN_FOLDER_CONF_PCT) or (consistent_ratio < MIN_CONSIST_RATIO):
            folder_pred = "unknown"

    # คำนวณใหม่หลัง override
    if folder_pred == "unknown":
        folder_conf_pct = max(
            min(float(np.mean([r["score1"] for r in results_each_img])), 1.0),
            -1.0
        ) * 100.0
    else:
        folder_conf_pct = max(min(avg_score, 1.0), -1.0) * 100.0

    # เวลาเริ่ม-จบ ของโฟลเดอร์
    first_res = results_each_img[0]
    last_res  = results_each_img[-1]
    start_time_folder = first_res["start_time"]
    end_time_folder   = last_res["end_time"]

    # total elapsed = sum ของ elapsed_s
    total_elapsed = sum([float(r["elapsed_s"]) for r in results_each_img])
    total_elapsed_fmt = f"{total_elapsed:.3f}"
    
    return {
        "test_class": folder_label,
        "predicted_class": folder_pred,
        "confidence_pct": folder_conf_pct,
        "start_time": start_time_folder,
        "end_time": end_time_folder,
        "elapsed_s": total_elapsed_fmt,
    }

#################################
# MAIN
#################################

def main():
    # โฟลเดอร์ย่อยจะเป็น "1","2","3",...
    subdirs = [d for d in sorted(os.listdir(TEST_ROOT))
               if os.path.isdir(os.path.join(TEST_ROOT, d))]

    # ให้เรียงตามเลข (ถ้าชื่อเป็นตัวเลข)
    def try_int(x):
        try:
            return int(x)
        except:
            return x
    subdirs = sorted(subdirs, key=try_int)
    
    n = []

    for folder_name in subdirs:
        folder_path = os.path.join(TEST_ROOT, folder_name)
        
        summary = summarize_folder(folder_path, folder_name)
           
        n.append(float(summary["elapsed_s"]))

        # พิมพ์ format แบบที่ต้องส่ง
        # confidence ทศนิยม 3 ตำแหน่ง
        conf_disp = f"{summary['confidence_pct']:.3f}"
        
        print("[Prediction Results]")
        print(f"  Test Class: {summary['test_class']}")
        print(f"  Predicted Class: {summary['predicted_class']}")
        print(f"  Confidence: {conf_disp}")
        print(f"  Start Time: {summary['start_time']}")
        print(f"  End Time: {summary['end_time']}")
        print(f"  Time Elapsed: {summary['elapsed_s']} s")
        print(" ")
    print(f"  เวลาทั้งที่ใช้ในการประมวลผลทั้งหมด: {sum(n):.3f} วินาที")

if __name__ == "__main__":
    main()

import os, json, glob, random, shutil, time
from math import floor

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================
# -------- CONFIG ----------
# ==========================

SOURCE_DIR = "Dataset_10_Ele"
DATA_DIR   = "Dataset"
ARTIFACTS  = "artifacts"

UNKNOWN_NAME   = "Unknown"
TRAIN_RATIO    = 0.6
VAL_RATIO      = 0.35
TEST_RATIO     = 0.05
EXT_OK         = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SEED           = 42
DEVICE         = "cuda"

BACKBONE       = "vit_base_patch32_clip_448.laion2b_ft_in12k_in1k"
IMG_SIZE       = 448
EMB_DIM        = 512
DROP           = 0.30
ARC_S          = 24.0
ARC_M          = 0.20

FT_EPOCHS      = 15
BATCH_SIZE     = 128
NUM_WORKERS    = 2
PIN_MEMORY     = True

LR_HEAD        = 1e-3
LR_LAST        = 3e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.0
MARGIN_TRIPLET = 0.2

ONNX_OPSET     = 17

# image norm (ใช้ตอนเทรนและอินเฟอร์)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

BEST_CKPT_PATH = os.path.join(ARTIFACTS, "Best_ViT_ft_only_head.pt")
EMB_ONNX_PATH  = os.path.join(ARTIFACTS, "elephant_embedding.onnx")
PROTOS_PATH    = os.path.join(ARTIFACTS, "prototypes.npy")
META_JSON_PATH = os.path.join(ARTIFACTS, "infer_meta.json")


# ===================================================
# --------- STEP 0. UTILS / REPRODUCIBILITY ---------
# ===================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ===================================================
# --------- STEP 1. SPLIT DATA INTO FOLDERS ---------
# ===================================================

def split_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DATA_DIR, split), exist_ok=True)

    classes = [d for d in os.listdir(SOURCE_DIR)
               if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    classes = sorted(classes)

    stats = {}
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)

    random.seed(SEED)

    for cls in classes:
        cls_dir = os.path.join(SOURCE_DIR, cls)
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(EXT_OK)]
        random.shuffle(imgs)

        n       = len(imgs)
        n_train = floor(n * TRAIN_RATIO)
        n_val   = floor(n * VAL_RATIO)
        n_test  = n - n_train - n_val

        train_files = imgs[:n_train]
        val_files   = imgs[n_train:n_train + n_val]
        test_files  = imgs[n_train + n_val:]

        stats[cls] = {
            "total": n,
            "train": len(train_files),
            "val":   len(val_files),
            "test":  len(test_files),
        }

        for fname in train_files:
            shutil.copy2(
                os.path.join(cls_dir, fname),
                os.path.join(DATA_DIR, "train", cls, fname)
            )
        for fname in val_files:
            shutil.copy2(
                os.path.join(cls_dir, fname),
                os.path.join(DATA_DIR, "val", cls, fname)
            )
        for fname in test_files:
            shutil.copy2(
                os.path.join(cls_dir, fname),
                os.path.join(DATA_DIR, "test", cls, fname)
            )

    known_ids = [c for c in classes if c != UNKNOWN_NAME]
    label_known_map = {cls_name: idx for idx, cls_name in enumerate(sorted(known_ids))}
    label_all_map = {**label_known_map}
    if UNKNOWN_NAME in classes:
        label_all_map[UNKNOWN_NAME] = -1

    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(os.path.join(DATA_DIR, "labels_known.json"), "w", encoding="utf-8") as f:
        json.dump(label_known_map, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, "labels_all.json"), "w", encoding="utf-8") as f:
        json.dump(label_all_map, f, ensure_ascii=False, indent=2)

    print("===== SPLIT STATS =====")
    for cls, info in stats.items():
        print(f"{cls}: total {info['total']}, "
              f"train {info['train']}, val {info['val']}, test {info['test']}")
    print()
    print("labels_known.json =", label_known_map)
    print("labels_all.json   =", label_all_map)


# ===================================================
# --------- STEP 2. DATASET / TRANSFORMS ------------
# ===================================================
def build_tf(img_size, is_train):
    aug_list = [
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
    ]
    if is_train:
        aug_list += [
            transforms.RandomGrayscale(0.1),
            transforms.GaussianBlur((3,3),sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomPerspective(0.5),
            transforms.RandomRotation(degrees=0.1),
            transforms.RandomCrop(img_size)

        ]
    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
    return transforms.Compose(aug_list)


class EleDataset(Dataset):
    def __init__(self,
                 root_split,
                 img_size,
                 is_train,
                 known_classes,
                 unknown_name="Unknown",
                 seed=42):
        self.root = root_split
        self.img_size = img_size
        self.is_train = is_train
        self.unknown_name = unknown_name
        self.known_classes = list(known_classes)
        self.class_to_idx_known = {c: i for i, c in enumerate(self.known_classes)}

        random.seed(seed)

        self.samples = []  # (path, label_id(int or -1), cls_name)
        exts = EXT_OK
        subdirs = sorted(next(os.walk(self.root))[1])
        for cls_name in subdirs:
            cls_dir = os.path.join(self.root, cls_name)
            img_files = sorted(glob.glob(os.path.join(cls_dir, "*")))
            img_files = [p for p in img_files if p.lower().endswith(exts)]

            if cls_name == self.unknown_name:
                label_id = -1
            else:
                if cls_name not in self.class_to_idx_known:
                    raise ValueError(
                        f"{cls_name} found in {self.root} but not in known_classes={self.known_classes}"
                    )
                label_id = self.class_to_idx_known[cls_name]

            for p in img_files:
                self.samples.append((p, label_id, cls_name))

        random.shuffle(self.samples)
        self.tf = build_tf(img_size, is_train=is_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, label_id, cls_name = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(label_id, dtype=torch.long)
        return x, y, cls_name
    
def _unnormalize_img(tensor_img, mean=MEAN, std=STD):
    # tensor_img: [3,H,W] after transforms.Normalize
    m = torch.tensor(mean).view(3,1,1)
    s = torch.tensor(std).view(3,1,1)
    img = tensor_img * s + m
    img = img.clamp(0,1)
    return img.permute(1,2,0).cpu().numpy()  # -> HWC, [0..1]

def debug_preview_augment(sample_ds, n_random=8, same_idx_repeats=4):
    import numpy as np
    idxs = np.random.choice(len(sample_ds), size=min(n_random, len(sample_ds)), replace=False)

    # --- (A) รูปสุ่มหลายรูป
    cols = 4
    rows = int(np.ceil(len(idxs)/cols))
    plt.figure(figsize=(4*cols, 4*rows))
    for i, ds_idx in enumerate(idxs):
        x, y, cls_name = sample_ds[ds_idx]
        img_vis = _unnormalize_img(x)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img_vis)
        plt.axis("off")
        plt.title(f"{cls_name} (y={int(y)})", fontsize=10)
    plt.suptitle("Augmented samples (random picks from train_ds)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS, "debug_aug_random.png"), dpi=150)
    plt.close()

    # --- (B) รูปเดียว แต่หลายครั้ง
    ref_idx = int(idxs[0])
    plt.figure(figsize=(3*same_idx_repeats, 3))
    for k in range(same_idx_repeats):
        x, y, cls_name = sample_ds[ref_idx]
        img_vis = _unnormalize_img(x)
        plt.subplot(1, same_idx_repeats, k+1)
        plt.imshow(img_vis)
        plt.axis("off")
        plt.title(f"aug#{k+1}\n{cls_name}", fontsize=10)
    plt.suptitle("Same source image -> different augmentations", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS, "debug_aug_sameimg.png"), dpi=150)
    plt.close()

    print("✅ Saved augmentation previews to:")
    print("   artifacts/debug_aug_random.png")
    print("   artifacts/debug_aug_sameimg.png")


def build_loaders_ft(data_dir=DATA_DIR,
                     img_size=IMG_SIZE,
                     batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS,
                     pin_memory=PIN_MEMORY,
                     seed=SEED):
    set_seed(seed)

    subdirs = sorted(next(os.walk(os.path.join(data_dir, "train")))[1])
    known_classes = [c for c in subdirs if c != UNKNOWN_NAME]

    train_ds = EleDataset(
        root_split=os.path.join(data_dir, "train"),
        img_size=img_size,
        is_train=True,
        known_classes=known_classes,
        unknown_name=UNKNOWN_NAME,
        seed=seed,
    )
    val_ds = EleDataset(
        root_split=os.path.join(data_dir, "val"),
        img_size=img_size,
        is_train=False,
        known_classes=known_classes,
        unknown_name=UNKNOWN_NAME,
        seed=seed,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    class_to_idx_known = {c: i for i, c in enumerate(known_classes)}

    print("Known classes:", known_classes, "-> num_known =", len(known_classes))
    print("Total train samples:", len(train_ds), "Total val samples:", len(val_ds))

    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(os.path.join(ARTIFACTS, "class_to_idx_known.json"), "w", encoding="utf-8") as f:
        json.dump(class_to_idx_known, f, ensure_ascii=False, indent=2)

    return train_ds, val_ds, train_dl, val_dl, known_classes, class_to_idx_known


# ===================================================
# --------- STEP 3. MODEL / LOSSES ------------------
# ===================================================

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, n_classes, s=32.0, m=0.30, easy_margin=False):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, n_classes))
        nn.init.xavier_uniform_(self.W)
        self.s = s
        self.m = m
        self.easy = easy_margin

    def forward(self, x, labels=None):
        W = F.normalize(self.W, dim=0)     # [emb_dim, C]
        cos = x @ W                        # [B, C] (cosine)
        if labels is None:
            return self.s * cos
        theta = torch.acos(torch.clamp(cos, -1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.m)
        if self.easy:
            target = torch.where(cos > 0, target, cos)
        onehot = F.one_hot(labels, cos.size(1)).float()
        logits = cos * (1 - onehot) + target * onehot
        return self.s * logits


class ElephantIDNet(nn.Module):
    def __init__(self, backbone_name, num_classes, emb_dim=512, drop=0.3,
                 arc_s=32.0, arc_m=0.30):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        out_f = self.backbone.num_features

        self.bnneck = nn.BatchNorm1d(out_f)
        self.dropout = nn.Dropout(drop)
        self.proj = nn.Linear(out_f, emb_dim, bias=False)
        self.emb_bn = nn.BatchNorm1d(emb_dim)
        self.emb_bn.bias.requires_grad_(False)

        self.arc = ArcMarginProduct(emb_dim, num_classes, s=arc_s, m=arc_m)

        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')

    def forward(self, x, labels=None):
        f = self.backbone(x)        # [B, F]
        f = self.bnneck(f)
        f = self.dropout(f)
        z = self.proj(f)
        z = self.emb_bn(z)
        z = F.normalize(z, dim=-1)  # [B, emb_dim], L2 norm
        logits = self.arc(z, labels.clamp(min=0)) if labels is not None else self.arc(z, None)
        return z, logits


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, emb, labels):
        emb_f32 = emb.float()
        labels_f32 = labels

        device = emb_f32.device
        B = emb_f32.size(0)

        sim = emb_f32 @ emb_f32.t()
        dist = 1.0 - sim

        labels_col = labels_f32.view(B, 1)
        same_label = labels_col.eq(labels_col.t())            # True if same label id
        both_known = labels_col.ge(0) & labels_col.t().ge(0)  # both not -1
        mask_pos = same_label & both_known

        mask_self = torch.eye(B, dtype=torch.bool, device=device)

        neg_big = torch.tensor(-1e4, dtype=dist.dtype, device=device)
        dist_pos = dist.clone()
        dist_pos[mask_self] = neg_big
        dist_pos[~mask_pos] = neg_big
        hardest_pos, _ = dist_pos.max(dim=1)
        has_pos = hardest_pos > (-1e3)

        mask_neg = ~same_label
        pos_big = torch.tensor(1e4, dtype=dist.dtype, device=device)
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = pos_big
        hardest_neg, _ = dist_neg.min(dim=1)
        has_neg = hardest_neg < (1e3)

        valid_anchor = has_pos & has_neg
        trip_raw = hardest_pos - hardest_neg + self.margin
        trip_raw = torch.relu(trip_raw)

        if valid_anchor.any():
            loss_f32 = trip_raw[valid_anchor].mean()
        else:
            loss_f32 = torch.tensor(0.0, dtype=dist.dtype, device=device)

        return loss_f32.to(emb.dtype)


@torch.no_grad()
def evaluate(model, dl, ce_loss_fn, device=DEVICE):
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_preds = []
    all_gts = []

    for imgs, labels, _names in dl:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        z, logits = model(imgs, labels=labels)

        known_mask = (labels >= 0)
        if known_mask.any():
            known_logits = logits[known_mask]
            known_labels = labels[known_mask]

            batch_loss = ce_loss_fn(known_logits, known_labels)
            bsz = known_logits.size(0)
            total_loss += float(batch_loss) * bsz
            total_count += bsz

            pred_idx = known_logits.softmax(-1).argmax(-1).cpu().tolist()
            gt_idx = known_labels.cpu().tolist()

            all_preds.extend(pred_idx)
            all_gts.extend(gt_idx)

    if total_count > 0:
        avg_loss = total_loss / total_count
    else:
        avg_loss = 0.0

    if len(all_gts) > 0:
        acc = accuracy_score(all_gts, all_preds)
        f1_macro = f1_score(all_gts, all_preds, average="macro")
    else:
        acc = 0.0
        f1_macro = 0.0

    return avg_loss, acc, f1_macro


def freeze_backbone(model, freeze=True):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def head_params(model):
    mods = [model.bnneck, model.proj, model.emb_bn, model.arc]
    return [p for m in mods for p in m.parameters() if p.requires_grad is not False]


# ===================================================
# --------- STEP 4. TRAIN / FINE-TUNE ---------------
# ===================================================

def finetune_only_head():
    os.makedirs(ARTIFACTS, exist_ok=True)

    train_ds, val_ds, train_dl, val_dl, known_classes, class_to_idx_known = build_loaders_ft(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        seed=SEED,
    )

    num_known = len(known_classes)

    model = ElephantIDNet(
        backbone_name=BACKBONE,
        num_classes=num_known,
        emb_dim=EMB_DIM,
        drop=DROP,
        arc_s=ARC_S,
        arc_m=ARC_M,
    ).to(DEVICE)

    #freeze backbone ทั้งหมดก่อน
    freeze_backbone(model, True)

    #แล้วค่อย unfreeze บางส่วนของ backbone 
    backbone_train_params = []

    if hasattr(model.backbone, "blocks"):
        #ปลดสองบล็อคท้ายสุด
        for p in model.backbone.blocks[-2:].parameters():
            p.requires_grad = True
        backbone_train_params += list(model.backbone.blocks[-2:].parameters())

    #ถ้า backbone มี fc_norm / norm_pre อะไรท้ายๆ ให้ปลดด้วยเพราะสำคัญกับ CLS token
    for attr_name in ["fc_norm", "norm", "norm_pre"]:
        if hasattr(model.backbone, attr_name):
            m = getattr(model.backbone, attr_name)
            for p in m.parameters():
                p.requires_grad = True
            backbone_train_params += list(m.parameters())

    head_train_params = head_params(model)

    #loss
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    triplet_fn = BatchHardTripletLoss(margin=MARGIN_TRIPLET)

    #optimizer head LR สูง, backbone LR ต่ำ
    opt = optim.AdamW(
        [
            {"params": head_train_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
            {"params": backbone_train_params, "lr": LR_LAST, "weight_decay": WEIGHT_DECAY},
        ],
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )
    #GradScaler
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_f1 = -1.0
    best_path = BEST_CKPT_PATH

    print("=== เริ่ม Fine-tune ===")
    for ep in range(1, FT_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        for imgs, labels, _names in train_dl:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                z, logits = model(imgs, labels=labels)

                known_mask = (labels >= 0)
                if known_mask.any():
                    ce_l = ce_loss_fn(
                        logits[known_mask],
                        labels[known_mask]
                    )
                else:
                    ce_l = torch.tensor(0.0, device=DEVICE)

                trip_l = triplet_fn(z, labels)
                loss = ce_l + trip_l
           
            running_loss = 0.0
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.detach()) * imgs.size(0)
            running_count += imgs.size(0)
        sch.step()

        val_loss, val_acc, val_f1 = evaluate(model, val_dl, ce_loss_fn, device=DEVICE)

        avg_train = running_loss / max(1, running_count)
        print(
            f"Epoch {ep:02d}/{FT_EPOCHS} | "
            f"train_loss={avg_train:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt = {
                "model": model.state_dict(),
                "backbone": BACKBONE,
                "emb_dim": EMB_DIM,
                "arc_s": ARC_S,
                "arc_m": ARC_M,
                "img_size": IMG_SIZE,
                "mean": MEAN,
                "std": STD,
                "known_classes": known_classes,
                "class_to_idx_known": class_to_idx_known,
            }
            torch.save(ckpt, best_path)
            print(f"  Saved best → {best_path}  (val_f1={best_f1:.4f})")

    if ep == 1:
            # 1) ปิด grad ทั้ง backbone ก่อน
            for p in model.backbone.parameters():
                p.requires_grad = False

            # 2) เปิดเฉพาะบล็อกท้าย ๆ (เราจะจูนให้มันแยก unknown ดีขึ้น)
            if hasattr(model.backbone, "blocks"):
                for p in model.backbone.blocks[-2:].parameters():
                    p.requires_grad = True
                print("🔓 Unfroze last 2 ViT blocks")
            else:
                print("⚠ backbone ไม่มี .blocks ข้ามการ unfreeze partial")

            # 3) ดึง param ที่ optimizer มีอยู่แล้ว
            already_params = set()
            for g in opt.param_groups:
                for q in g["params"]:
                    already_params.add(id(q))

            # 4) เก็บเฉพาะ param ใหม่จริง ๆ + unique
            new_params_unique = []
            seen_new = set()

            for p in model.backbone.parameters():
                if p.requires_grad:
                    pid = id(p)
                    if (pid not in already_params) and (pid not in seen_new):
                        new_params_unique.append(p)
                        seen_new.add(pid)

            # 5) ค่อย add เข้า optimizer ถ้ามีจริง
            if len(new_params_unique) > 0:
                opt.add_param_group({
                    "params": new_params_unique,
                    "lr": LR_LAST,
                    "weight_decay": WEIGHT_DECAY,
                })
                print(f"➕ Added {len(new_params_unique)} new backbone params with lr={LR_LAST}")
            else:
                print("ℹ No new params to add")

    print("Done. Best macro-F1 (known only) =", best_f1)
    return best_path


# ===================================================
# --------- STEP 5. EXPORT ARTIFACTS ----------------
# ===================================================

def export_artifacts(ckpt_path=BEST_CKPT_PATH):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    backbone_name = ckpt["backbone"]
    emb_dim = ckpt["emb_dim"]
    arc_s = ckpt["arc_s"]
    arc_m = ckpt["arc_m"]
    img_size = ckpt["img_size"]
    mean = ckpt["mean"]
    std = ckpt["std"]
    known_classes = ckpt["known_classes"]
    class_to_idx_known = ckpt["class_to_idx_known"]
    num_known = len(known_classes)

    model = ElephantIDNet(
        backbone_name=backbone_name,
        num_classes=num_known,
        emb_dim=emb_dim,
        drop=DROP,
        arc_s=arc_s,
        arc_m=arc_m,
    ).to(DEVICE)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print("export_artifacts: missing keys:", missing)
    print("export_artifacts: unexpected keys:", unexpected)

    model.eval()

    class EmbeddingOnly(nn.Module):
        def __init__(self, trained_model):
            super().__init__()
            self.trained_model = trained_model
        def forward(self, x):
            z, _ = self.trained_model(x, labels=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            return z

    embed_model = EmbeddingOnly(model).to(DEVICE).eval()

    with torch.no_grad():
        dummy = torch.randn(1, 3, img_size, img_size, device=DEVICE)
        z = embed_model(dummy)
        print("embed shape:", tuple(z.shape))

    with torch.no_grad():
        W = model.arc.W.detach().clone().to("cpu").numpy()  # [emb_dim, C]
        W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
        W = W.T.astype(np.float32)  # [C, emb_dim]
    np.save(PROTOS_PATH, W)
    print("saved prototypes:", PROTOS_PATH, "shape", W.shape)

    torch.onnx.export(
        embed_model,
        dummy,
        EMB_ONNX_PATH,
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
    )
    print("exported ONNX:", EMB_ONNX_PATH)

    meta = {
        "known_classes": known_classes,
        "class_to_idx_known": class_to_idx_known,
        "img_size": img_size,
        "mean": mean,
        "std": std,
        "emb_dim": emb_dim,
        "tau_default": 0.70,
        "margin_rule": 0.20,
        "note": "row i of prototypes.npy matches known_classes[i]"
    }
    with open(META_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("saved meta json:", META_JSON_PATH)


# ===================================================
# --------- STEP 6. EVALUATION (VAL/TEST) -----------
# ===================================================

def build_eval_loaders(ckpt_path=BEST_CKPT_PATH):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    known_classes = ckpt["known_classes"]

    val_ds = EleDataset(
        root_split=os.path.join(DATA_DIR, "val"),
        img_size=ckpt["img_size"],
        is_train=False,
        known_classes=known_classes,
        unknown_name=UNKNOWN_NAME,
        seed=SEED,
    )
    test_ds = EleDataset(
        root_split=os.path.join(DATA_DIR, "test"),
        img_size=ckpt["img_size"],
        is_train=False,
        known_classes=known_classes,
        unknown_name=UNKNOWN_NAME,
        seed=SEED,
    )

    val_dl = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return val_ds, test_ds, val_dl, test_dl, known_classes


def eval_split(name, dl, model, known_classes, save_prefix):
    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for x, y, _clsname in dl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            _, logits = model(x, labels=y)
            pred_ids = logits.softmax(-1).argmax(-1)

            mask_known = (y >= 0)
            if mask_known.any():
                all_preds.extend(pred_ids[mask_known].cpu().tolist())
                all_gts.extend(y[mask_known].cpu().tolist())

    if len(all_gts) == 0:
        print(f"[{name}] no known-class samples to evaluate")
        return

    acc = accuracy_score(all_gts, all_preds)
    rep = classification_report(
        all_gts,
        all_preds,
        target_names=known_classes,
        digits=4,
        zero_division=0,
    )
    print(f"[{name}] Accuracy: {acc:.4f}")
    print(rep)

    cm = confusion_matrix(
        all_gts, all_preds,
        labels=list(range(len(known_classes)))
    )
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=known_classes,
        yticklabels=known_classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} Confusion — Acc={acc:.3f}")
    plt.tight_layout()

    fig_path = f"{save_prefix}_confusion_matrix.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    rep_path = f"{save_prefix}_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"{name} Accuracy: {acc:.4f}\n\n")
        f.write(rep)

    print("saved:", rep_path, fig_path)


def run_eval(ckpt_path=BEST_CKPT_PATH):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    known_classes = ckpt["known_classes"]
    num_known = len(known_classes)

    model = ElephantIDNet(
        backbone_name=ckpt["backbone"],
        num_classes=num_known,
        emb_dim=ckpt["emb_dim"],
        drop=DROP,
        arc_s=ckpt["arc_s"],
        arc_m=ckpt["arc_m"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    val_ds, test_ds, val_dl, test_dl, known_classes = build_eval_loaders(ckpt_path)

    print("val size :", len(val_ds))
    print("test size:", len(test_ds))
    print("known_classes (order):", known_classes)

    eval_split("VAL",  val_dl,  model, known_classes, save_prefix="val")
    eval_split("TEST", test_dl, model, known_classes, save_prefix="test")


# ===================================================
# --------------------- MAIN -----------------------
# ===================================================

def main():
    torch.multiprocessing.set_start_method("spawn", force=True)

    # 1) split raw data -> Dataset/train|val|test
    # split_dataset()
    
    subdirs = sorted(next(os.walk(os.path.join(DATA_DIR, "train")))[1])
    known_classes = [c for c in subdirs if c != UNKNOWN_NAME]
    preview_ds = EleDataset(
        root_split=os.path.join(DATA_DIR, "train"),
        img_size=IMG_SIZE,
        is_train=True,
        known_classes=known_classes,
        unknown_name=UNKNOWN_NAME,
        seed=SEED,
    )
    debug_preview_augment(preview_ds)
    # 2) train head-only fine-tune
    best_path = finetune_only_head()
    print("Training done. best ckpt:", best_path)

    # 3) export artifacts for inference (onnx, prototypes, meta)
    export_artifacts(best_path)

    # 4) evaluate model on val/test and save reports + confusion matrices
    run_eval(best_path)


if __name__ == "__main__":
    main()


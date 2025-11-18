'''
在persam2_ori.py的基础上，修改了自动点选模块，使其支持多峰参考特征的聚类表示。
主要修改点包括：
1. 在set_reference方法中，提取参考图像的特征时，使用
KMeans对前景特征进行聚类，得到多个中心点作为视觉提示。
2. 在predict方法中，计算测试图像与参考特征的相似度时，使用参考图像的所有前景特征进行相似度计算，并对相似度进行聚合。
3. 修改了cal_point方法，使其能够处理多个视觉提示，并为每个提示选择一个正点，同时选择一个全局负点。
说明（关键代码修改点）
target_feat 改为 list，每个元素是参考前景所有像素的 normalized feature [N, C]。在 predict() 中用矩阵乘法得到 [N, Hf*Wf]，再对 N 做 mean（或可改为 max）来保持多峰响应。此处保留 mean，稳定且能覆盖多个部件。
target_embedding 仍保留 [B,1,C]（mean），以兼容 predictor.predict(..., target_embedding=...) 的 API。
visual_prompt 改为多中心 [B, K, C, 1, 1]，使用 KMeans 在前景像素上生成 1~4 个中心（按前景像素数量自适应），并在 cal_point() 中对每个中心取 argmax 产生多个正点。
修复了 prompt/reshape 的错误取法与 torch.bmm 传入 list 导致的异常。
''' 
import os
import glob
import argparse
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class AutomaticPointor:
    def __init__(self, sam2_checkpoint: str, sam2_cfg: str, device: Optional[str] = None):
        self.model = build_sam2(sam2_cfg, sam2_checkpoint)
        self.predictor = SAM2ImagePredictor(self.model)
        self.device = device or self.model.device
        self.model.eval()
        # --- modified: store visual_prompt and target_feat differently to keep multi-peak info
        self.visual_prompt: Optional[torch.Tensor] = None        # [B, K, C, 1, 1]
        self.ref_feats_for_clustering: Optional[torch.Tensor] = None
        self.ref_mask_for_clustering: Optional[torch.Tensor] = None
        self.num_prompt_clusters = 1
        self.cluster_agg_method = "mean"
        self.last_points: Optional[torch.Tensor] = None
        self.last_labels: Optional[torch.Tensor] = None
        # target_embedding: keep [B,1,C] for predictor API; target_feat: list per batch of dense [N, C] for sim
        self.target_embedding : Optional[torch.Tensor] = None   # [B,1,C]
        self.target_feat : Optional[List[torch.Tensor]] = None  # list of length B, each [N, C]

    def set_reference(self, ref_image: np.ndarray, ref_mask: np.ndarray) -> None:
        self.predictor.set_image(ref_image)
        ref_features = self.predictor._features
        mask = torch.as_tensor(ref_mask, dtype=torch.float, device=self.device)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        processed_mask = self.predictor._transforms.transform_masks(mask)

        feats = ref_features["image_embed"]
        if feats.dim() == 3:
            feats = feats.unsqueeze(0)
        Bf, C, Hf, Wf = feats.shape
        # --- modified: keep ref_feats_for_clustering for later cluster sampling
        self.ref_feats_for_clustering = feats.permute(0, 2, 3, 1).detach()  # [B, Hf, Wf, C]
        mask_feat = F.interpolate(processed_mask.float(), size=(Hf, Wf), mode="bilinear", align_corners=False)
        self.ref_mask_for_clustering = (mask_feat > 0.5).squeeze(1).bool().detach()  # [B, Hf, Wf]
        b, hf, wf, c = self.ref_feats_for_clustering.shape

        # --- modified: build dense per-pixel target_feat list and mean target_embedding
        dense_embeddings = []
        mean_embeddings = []
        for i in range(b):
            feat_b = self.ref_feats_for_clustering[i]  # [Hf, Wf, C]
            mask_b = self.ref_mask_for_clustering[i]  # [Hf, Wf]
            target_feat_pixels = feat_b[mask_b]       # [N, C]
            if target_feat_pixels.shape[0] == 0:
                # fallback: use all pixels
                target_feat_pixels = feat_b.reshape(-1, c)

            # normalize per-vector and keep dense list (N, C)
            target_feat_pixels = F.normalize(target_feat_pixels, p=2, dim=1).detach()
            dense_embeddings.append(target_feat_pixels)

            # keep mean for predictor API but do not use it for sim aggregation directly
            mean_emb = target_feat_pixels.mean(0, keepdim=True)  # [1, C]
            mean_embeddings.append(mean_emb)

        # store
        self.target_feat = dense_embeddings                     # list of [N, C]
        self.target_embedding = torch.cat(mean_embeddings, dim=0).unsqueeze(1).to(self.device)  # [B,1,C]

        # --- modified: extract visual_prompt as multi-centroid prompts [B, K, C, 1, 1]
        self.visual_prompt = self.extract_visual_prompt(ref_features, processed_mask)

    def predict(self, test_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.visual_prompt is None or self.target_feat is None:
            raise ValueError("Reference not set. Call set_reference() first.")
        self.predictor.set_image(test_image)
        cached_features = self.predictor._features
        orig_hw = self.predictor._orig_hw[0]
        test_feat_embed = cached_features["image_embed"] # [B, C, Hf, Wf]
        b, c, hf, wf = test_feat_embed.shape

        # normalize test features for cosine similarity
        norm_test_feat = F.normalize(test_feat_embed, p=2, dim=1)  # [B, C, Hf, Wf]
        norm_test_feat_flat = norm_test_feat.view(b, c, hf * wf)   # [B, C, Hf*Wf]

        # --- modified: compute similarity using dense reference features (list) and aggregate across ref pixels
        sim_list = []
        for bi in range(b):
            ref_dense = self.target_feat[bi]          # [N, C]
            test_b_flat = norm_test_feat_flat[bi]    # [C, Hf*Wf]
            # matmul: [N, C] x [C, Hf*Wf] -> [N, Hf*Wf]
            sim_dense = torch.matmul(ref_dense, test_b_flat)      # [N, Hf*Wf]
            # aggregate over N (ref pixels): mean (keeps multi-peak behavior)
            sim_agg = sim_dense.mean(0)                           # [Hf*Wf]
            sim_agg = sim_agg.view(1, 1, hf, wf)                  # [1,1,Hf,Wf]
            sim_list.append(sim_agg)
        sim = torch.cat(sim_list, dim=0)  # [B,1,Hf,Wf]

        # upsample and postprocess to image size
        sim_up = self.predictor._transforms.postprocess_masks(
            sim,
            orig_hw=orig_hw,
        )  # [B,1,H,W]

        sim_orig = sim_up.squeeze(1)  # [B, H, W]

        # build attn_sim similar to original (used for attention guidance)
        attn_sim_list = []
        for i in range(b):
            sim_b = sim_orig[i] # [orig_h, orig_w]
            sim_std = torch.std(sim_b)
            if sim_std == 0:
                sim_std = 1.0
            sim_b = (sim_b - sim_b.mean()) / sim_std
            sim_b_64 = F.interpolate(sim_b.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
            attn_sim_b = sim_b_64.sigmoid_().unsqueeze(0).flatten(3)
            attn_sim_list.append(attn_sim_b)
        attn_sim = torch.cat(attn_sim_list, dim=0)  # [B, 1, 4096] or similar

        # auto point coords / labels using cal_point (cal_point uses visual_prompt multi-centroids)
        auto_point_coords, auto_point_labels = self.cal_point(
            cached_features, self.visual_prompt, orig_hw
        )

        self.last_points = auto_point_coords.clone()
        self.last_labels = auto_point_labels.clone()

        # call predictor with multimodal guidance
        masks, scores, logits = self.predictor.predict(
            point_coords=auto_point_coords,
            point_labels=auto_point_labels,
            multimask_output=True,
            attn_sim=attn_sim,
            target_embedding=self.target_embedding  # keep mean for predictor
        )

        best_idx = int(np.argmax(scores))
        best_logits = logits[best_idx][None, ...]

        masks_ref1, scores_ref1, logits_ref1 = self.predictor.predict(
            point_coords=auto_point_coords,
            point_labels=auto_point_labels,
            mask_input=best_logits,
            multimask_output=True,
        )

        logits_ref1_t = torch.as_tensor(logits_ref1[0][None, ...], device=self.device)

        mask_bool = masks_ref1[0].astype(bool)
        ys, xs = np.nonzero(mask_bool)
        if xs.size and ys.size:
            x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
            input_box = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
        else:
            input_box = None

        masks_ref2, scores_ref2, logits_ref2 = self.predictor.predict(
            point_coords=auto_point_coords,
            point_labels=auto_point_labels,
            box=input_box if input_box is not None else None,
            mask_input=logits_ref1_t,
            multimask_output=True,
        )

        return masks_ref2, scores_ref2, logits_ref2

    def extract_visual_prompt(self, ref_features: Dict[str, torch.Tensor], ref_mask: torch.Tensor) -> torch.Tensor:
        """
        Modified: return multi-centroid visual prompts of shape [B, K, C, 1, 1]
        Each centroid represents a mode (component) in the reference foreground.
        """
        features = ref_features["image_embed"]
        if features.dim() == 3:
            features = features.unsqueeze(0)
        B, C, Hf, Wf = features.shape

        mask = F.interpolate(ref_mask.float(), size=(Hf, Wf), mode="bilinear", align_corners=False)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).float().squeeze(1)  # [B, Hf, Wf]

        prompts_per_batch = []
        for i in range(B):
            feat = features[i].permute(1, 2, 0)  # [Hf, Wf, C]
            fg_feat = feat[mask[i].bool()]       # [N, C]
            if fg_feat.shape[0] == 0:
                # fallback to global mean
                center = features[i].view(C, -1).mean(1, keepdim=True).T  # [1, C]
                centers = center
            else:
                # number of centers adaptive to foreground size, clamp in [1,4]
                n_centers = int(min(max(1, fg_feat.shape[0] // 200), 4))
                if n_centers <= 1:
                    centers = fg_feat.mean(0, keepdim=True)  # [1, C]
                else:
                    kmeans = KMeans(n_clusters=n_centers, random_state=0).fit(
                        fg_feat.cpu().numpy()
                    )
                    centers = torch.tensor(kmeans.cluster_centers_, dtype=features.dtype, device=features.device)  # [K, C]

            centers = F.normalize(centers, p=2, dim=1)    # [K, C]
            centers = centers.unsqueeze(-1).unsqueeze(-1)  # [K, C, 1, 1]
            prompts_per_batch.append(centers)

        # pad clusters to same K per batch if needed (choose max K)
        max_k = max([p.shape[0] for p in prompts_per_batch])
        padded_prompts = []
        for p in prompts_per_batch:
            k = p.shape[0]
            if k < max_k:
                # pad by repeating first center
                pad = p[0:1].repeat(max_k - k, 1, 1, 1)
                p = torch.cat([p, pad], dim=0)
            padded_prompts.append(p)
        prompt_tensor = torch.stack(padded_prompts, dim=0)  # [B, K, C, 1, 1]
        return prompt_tensor

    def cal_point(self, test_features: Dict[str, torch.Tensor],
                  visual_prompt: torch.Tensor,
                  original_image_size: Tuple[int, int]):
        """
        Modified cal_point to accept visual_prompt [B, K, C, 1, 1]
        and compute per-cluster argmax to produce multiple positive points and one negative.
        """
        device = test_features["image_embed"].device
        B, C, Hf, Wf = test_features["image_embed"].shape
        orig_h, orig_w = original_image_size

        test_feat = F.normalize(test_features["image_embed"], p=2, dim=1)  # [B, C, Hf, Wf]

        # visual_prompt expected [B, K, C, 1, 1]
        prompt_multi = visual_prompt.to(device)  # [B, K, C, 1, 1]
        num_clusters = prompt_multi.shape[1]

        # compute similarity maps for each cluster centroid
        sim_list = []
        for k in range(num_clusters):
            p = prompt_multi[:, k]            # [B, C, 1, 1]
            sim = F.cosine_similarity(test_feat, p, dim=1)  # [B, Hf, Wf]
            sim_list.append(sim.unsqueeze(1))
        sim_map = torch.cat(sim_list, dim=1)  # [B, K, Hf, Wf]

        # aggregate across clusters for a global map used for negative selection (mean or max)
        if getattr(self, "cluster_agg_method", "mean") == "max":
            sim_agg = sim_map.max(dim=1)[0]  # [B, Hf, Wf]
        else:
            sim_agg = sim_map.mean(dim=1)    # [B, Hf, Wf]

        # upsample maps to original image resolution
        sim_up_multi = F.interpolate(sim_map, size=(orig_h, orig_w), mode="bilinear", align_corners=False)  # [B, K, H, W]
        sim_up_agg = F.interpolate(sim_agg.unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)  # [B, H, W]

        auto_coords = []
        auto_labels = []

        for b_idx in range(B):
            h, w = sim_up_agg[b_idx].shape
            batch_coords = []
            batch_labels = []

            # for each cluster pick the highest response location (positive)
            for k in range(sim_up_multi.shape[1]):
                sim_k = sim_up_multi[b_idx, k]
                flat_k = sim_k.flatten()
                pos_idx = torch.argmax(flat_k)
                pos_y, pos_x = divmod(int(pos_idx.item()), w)
                pos_x = float(max(0, min(w - 1, pos_x)))
                pos_y = float(max(0, min(h - 1, pos_y)))
                batch_coords.append([pos_x, pos_y])
                batch_labels.append(1)

            # pick global negative (argmin on aggregated map)
            flat_g = sim_up_agg[b_idx].flatten()
            neg_idx = torch.argmin(flat_g)
            ny = int(neg_idx // w)
            nx = int(neg_idx % w)
            neg_x = float(max(0, min(w - 1, nx)))
            neg_y = float(max(0, min(h - 1, ny)))
            batch_coords.append([neg_x, neg_y])
            batch_labels.append(0)

            coords = torch.tensor(batch_coords, device=device, dtype=torch.float32).unsqueeze(0)
            labels = torch.tensor(batch_labels, device=device, dtype=torch.long).unsqueeze(0)
            auto_coords.append(coords)
            auto_labels.append(labels)

        auto_point_coords = torch.cat(auto_coords, dim=0)
        auto_point_labels = torch.cat(auto_labels, dim=0)

        return auto_point_coords, auto_point_labels

    def save_vis(self,
                 image: np.ndarray,
                 mask: np.ndarray,
                 output_path: str,
                 pos_icon_path: str = "icon/click3.png",
                 neg_icon_path: str = "icon/click4.png"):

        if self.last_points is None or self.last_labels is None:
            print(f"Warning: 'predict()' must be called before 'save_vis()'. Skipping visualization for {output_path}")
            return
        overlay_img = image.copy()
        alpha = 0.5
        overlay_img[mask > 0] = (alpha * np.array([0, 255, 0]) + (1 - alpha) * overlay_img[mask > 0])

        pos_icon = cv2.imread(pos_icon_path, cv2.IMREAD_UNCHANGED) if os.path.exists(pos_icon_path) else None
        neg_icon = cv2.imread(neg_icon_path, cv2.IMREAD_UNCHANGED) if os.path.exists(neg_icon_path) else None

        overlay = overlay_img.copy()
        points = self.last_points[0]
        labels = self.last_labels[0]

        for i in range(points.shape[0]):
            x = int(points[i, 0].item())
            y = int(points[i, 1].item())
            label = labels[i].item()
            icon_to_draw = pos_icon if label == 1 else neg_icon
            if icon_to_draw is None:
                continue

            new_size = (64, 64)
            icon_to_draw = cv2.resize(icon_to_draw, new_size, interpolation=cv2.INTER_AREA)

            ih, iw = icon_to_draw.shape[:2]
            y1 = max(0, y - ih // 2)
            y2 = min(image.shape[0], y + (ih - ih // 2))
            x1 = max(0, x - iw // 2)
            x2 = min(image.shape[1], x + (iw - iw // 2))

            icon_y1 = (ih // 2) - (y - y1)
            icon_y2 = (ih // 2) + (y2 - y)
            icon_x1 = (iw // 2) - (x - x1)
            icon_x2 = (iw // 2) + (x2 - x)

            icon_resized = icon_to_draw[icon_y1:icon_y2, icon_x1:icon_x2]

            if icon_resized.shape[2] == 4:
                alpha_icon = icon_resized[:, :, 3] / 255.0
                for c in range(3):
                    overlay[y1:y2, x1:x2, c] = (
                        alpha_icon * icon_resized[:, :, c] + (1 - alpha_icon) * overlay[y1:y2, x1:x2, c]
                    )
            else:
                overlay[y1:y2, x1:x2] = icon_resized[:, :, :3]

        plt.figure(figsize=(8, 8))
        plt.imshow(overlay.astype(np.uint8))
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

def inference(auto_pointer: AutomaticPointor,
              test_image: np.ndarray,
              vis_output_path: str = None,
              mask_output_path: str = None
              ):
    masks, scores, logits = auto_pointer.predict(test_image)
    best_idx = int(np.argmax(scores))
    final_mask = masks[best_idx]
    # sav vis
    if vis_output_path:
        auto_pointer.save_vis(
            image=test_image,
            mask=final_mask,
            output_path=vis_output_path
        )

    if mask_output_path:
        masks_uint8 = (final_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_output_path, masks_uint8)

    return masks, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PerSAM2 Fine-tuning on PerSeg dataset structure.")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2.1_hiera_large.pt", help="Path to SAM2 checkpoint.")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 config.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root containing Images/ and Annotations/ folders.")
    parser.add_argument("--class_name", type=str, default=None, help="Specific class to run. If None, runs all found classes.")
    parser.add_argument("--ref_idx", type=str, default="00", help="Index of reference image (e.g., '00').")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save results.")
    parser.add_argument("--num_prompt_clusters", type=int, default=1,help="cluster on reference-level")

    args = parser.parse_args()

    images_root = os.path.join(args.data_root, "Images")
    if args.class_name:
        classes = [args.class_name]
    else:
        classes = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])
        print(f"======> Automatically found {len(classes)} classes: {classes}")

    for class_idx, class_name in enumerate(classes):
        print(f"\n\n[{class_idx+1}/{len(classes)}] Processing Class: ===> {class_name} <===")

        img_dir = os.path.join(args.data_root, "Images", class_name)
        mask_dir = os.path.join(args.data_root, "Annotations", class_name)
        ref_img_name = f"{args.ref_idx}.jpg"
        ref_mask_name = f"{args.ref_idx}.png"
        ref_img_path = os.path.join(img_dir, ref_img_name)
        ref_mask_path = os.path.join(mask_dir, ref_mask_name)

        if not os.path.exists(ref_img_path) or not os.path.exists(ref_mask_path):
            print(f"Skipping {class_name}: Reference files not found ({ref_img_name}/{ref_mask_name})")
            continue

        print(f"--> Initializing SAM2 for {class_name}...")
        persam = AutomaticPointor(args.sam2_checkpoint, args.model_cfg)

        print(f"--> Loading reference: {os.path.join(class_name, ref_img_name)}")
        ref_img = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)

        print(f"--> Setting reference (One-shot)...")
        persam.set_reference(ref_img, ref_mask)

        output_class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        test_images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        print(f"--> Found {len(test_images)} images. Starting inference for {class_name}...")

        for test_path in tqdm(test_images, desc=f"Inference ({class_name})"):
            img_name = os.path.basename(test_path)
            if img_name == ref_img_name: continue

            test_img = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)

            base_name = os.path.splitext(img_name)[0]
            vis_path = os.path.join(output_class_dir, base_name + "_vis.jpg")
            mask_path = os.path.join(output_class_dir, base_name + ".png")

            inference(
                persam,
                test_img,
                vis_output_path=vis_path,
                mask_output_path=mask_path
            )

    print("\n======> All classes processed.")

import os
import glob
import argparse
from typing import Dict, Optional, Tuple

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
        self.visual_prompt: Optional[torch.Tensor] = None
        self.ref_feats_for_clustering: Optional[torch.Tensor] = None
        self.ref_mask_for_clustering: Optional[torch.Tensor] = None
        self.num_prompt_clusters = 1
        self.cluster_agg_method = "mean"
        self.last_points: Optional[torch.Tensor] = None
        self.last_labels: Optional[torch.Tensor] = None
        self.target_embedding : Optional[torch.Tensor] = None
        self.target_feat : Optional[torch.Tensor] = None

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
        self.ref_feats_for_clustering = feats.permute(0, 2, 3, 1).detach()
        mask_feat = F.interpolate(processed_mask.float(), size=(Hf, Wf), mode="bilinear", align_corners=False)
        self.ref_mask_for_clustering = (mask_feat > 0.5).squeeze(1).bool().detach()
        b, hf, wf, c = self.ref_feats_for_clustering.shape
        target_embeddings = []
        for i in range(b):
            feat_b = self.ref_feats_for_clustering[i]  # [Hf, Wf, C]
            mask_b = self.ref_mask_for_clustering[i]  # [Hf, Wf]
            
            target_feat_pixels = feat_b[mask_b] # [N, C]
            if target_feat_pixels.shape[0] == 0:
                target_embedding_b = feat_b.mean([0, 1]).unsqueeze(0) # [1, C]
            else:
                target_embedding_b = target_feat_pixels.mean(0).unsqueeze(0) # [1, C]
            
            target_embeddings.append(target_embedding_b)
        
        self.target_embedding = torch.cat(target_embeddings, dim=0).unsqueeze(1) # [B, 1, C]

        self.visual_prompt = self.extract_visual_prompt(ref_features, processed_mask)
        self.target_feat = self.target_embedding 

    def predict(self, test_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.visual_prompt is None:
            raise ValueError("Reference not set. Call set_reference() first.")
        self.predictor.set_image(test_image)
        cached_features = self.predictor._features
        orig_hw = self.predictor._orig_hw[0]
        test_feat_embed = cached_features["image_embed"] # [B, C, Hf, Wf]
        b, c, hf, wf = test_feat_embed.shape

        norm_test_feat = F.normalize(test_feat_embed, p=2, dim=1)
        norm_test_feat_flat = norm_test_feat.view(b, c, hf * wf) # [B, C, Hf*Wf]

        sim = torch.bmm(self.target_feat, norm_test_feat_flat) # [B, 1, Hf*Wf]
        sim = sim.view(b, 1, hf, wf) # [B, 1, Hf, Wf]
        sim_up = self.predictor._transforms.postprocess_masks(
            sim,
            orig_hw=orig_hw,
        )  # [B,1,H,W]

        sim_orig = sim_up.squeeze(1)

        attn_sim_list = []
        for i in range(b):
            sim_b = sim_orig[i] # [orig_h, orig_w]
            sim_std = torch.std(sim_b)
            if sim_std == 0: 
                sim_std = 1.0
            sim_b = (sim_b - sim_b.mean()) / sim_std
            sim_b_64 = F.interpolate(sim_b.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
            # [1, 1, 4096]
            attn_sim_b = sim_b_64.sigmoid_().unsqueeze(0).flatten(3) 
            attn_sim_list.append(attn_sim_b)
        
        attn_sim = torch.cat(attn_sim_list, dim=0) # [B, 1, 4096]

        auto_point_coords, auto_point_labels = self.cal_point(
            cached_features, self.visual_prompt, orig_hw
        )

        self.last_points = auto_point_coords.clone()
        self.last_labels = auto_point_labels.clone()

        masks, scores, logits = self.predictor.predict(
            point_coords=auto_point_coords,
            point_labels=auto_point_labels,
            multimask_output=True,
            attn_sim=attn_sim,
            target_embedding=self.target_embedding
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
        features = ref_features["image_embed"]
        if features.dim() == 3:
            features = features.unsqueeze(0)
        B, C, Hf, Wf = features.shape

        mask = F.interpolate(ref_mask.float(), size=(Hf, Wf), mode="bilinear", align_corners=False)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).float()

        masked = features * mask
        area = mask.sum(dim=[2, 3], keepdim=True)
        mean_feat = masked.sum(dim=[2, 3], keepdim=True) / (area + 1e-6)

        min_val = masked.min() - 1.0
        masked_for_max = masked.clone()
        masked_for_max[mask.repeat(1, C, 1, 1) == 0] = min_val
        max_feat = masked_for_max.amax(dim=[2, 3], keepdim=True)

        alpha, beta = 1.0, 0.0
        prompt = alpha * mean_feat + beta * max_feat
        prompt_flat = prompt.view(B, C)
        prompt_norm = F.normalize(prompt_flat, p=2, dim=1).view(B, C, 1, 1)
        return prompt_norm

    def cal_point(self, test_features: Dict[str, torch.Tensor],
                                           visual_prompt: torch.Tensor,
                                           original_image_size: Tuple[int, int]):
        device = test_features["image_embed"].device
        B, C, Hf, Wf = test_features["image_embed"].shape
        orig_h, orig_w = original_image_size

        test_feat = F.normalize(test_features["image_embed"], p=2, dim=1)
        prompt_vec = visual_prompt
        if prompt_vec.dim() == 4:
            prompt_vec = prompt_vec.view(B, C)
        prompt_vec = F.normalize(prompt_vec, p=2, dim=1)

        n_clusters = max(1, int(getattr(self, "num_prompt_clusters", 1)))
        clustered_prompts = []
        if n_clusters > 1:
            for b in range(B):
                if self.ref_feats_for_clustering is None:
                    base = prompt_vec[b].unsqueeze(0).repeat(Hf * Wf, 1)
                    samples = (base + 0.001 * torch.randn_like(base, device=device)).cpu().numpy()
                else:
                    feat_b = self.ref_feats_for_clustering[b]  # [Hf, Wf, C]
                    mask_b = self.ref_mask_for_clustering[b]
                    samples = feat_b[mask_b].cpu().numpy()
                    if samples.shape[0] < n_clusters:
                        base = prompt_vec[b].unsqueeze(0).repeat(Hf * Wf, 1)
                        samples = (base + 0.001 * torch.randn_like(base, device=device)).cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(samples)
                centers = torch.tensor(kmeans.cluster_centers_, device=device, dtype=prompt_vec.dtype)
                centers = F.normalize(centers, p=2, dim=1)
                clustered_prompts.append(centers.unsqueeze(0))
            prompt_multi = torch.cat(clustered_prompts, dim=0)  # [B, n_clusters, C]
        else:
            prompt_multi = prompt_vec.unsqueeze(1)  # [B,1,C]

        sim_list = []
        for k in range(prompt_multi.shape[1]):
            p = prompt_multi[:, k, :].view(B, C, 1, 1)
            sim = F.cosine_similarity(test_feat, p, dim=1)
            sim_list.append(sim.unsqueeze(1))
        sim_map = torch.cat(sim_list, dim=1)  # [B, n_clusters, Hf, Wf]

        if getattr(self, "cluster_agg_method", "mean") == "max":
            sim_agg = sim_map.max(dim=1)[0]
        else:
            sim_agg = sim_map.mean(dim=1)

        sim_up_multi = F.interpolate(sim_map, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        sim_up_agg = F.interpolate(sim_agg.unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)

        auto_coords = []
        auto_labels = []
        for b in range(B):
            h, w = sim_up_agg[b].shape
            batch_coords = []
            batch_labels = []

            for k in range(sim_up_multi.shape[1]):
                sim_k = sim_up_multi[b, k]
                flat_k = sim_k.flatten()
                pos_idx = torch.argmax(flat_k)
                pos_y, pos_x = divmod(int(pos_idx.item()), w)
                pos_x = float(max(0, min(w - 1, pos_x)))
                pos_y = float(max(0, min(h - 1, pos_y)))
                batch_coords.append([pos_x, pos_y])
                batch_labels.append(1)

            flat_g = sim_up_agg[b].flatten()
            neg_idx = torch.argmin(flat_g)
            ny = int(neg_idx // w)
            nx = int(neg_idx %  w)
            neg_y, neg_x = ny, nx

            neg_x = float(max(0, min(w - 1, neg_x)))
            neg_y = float(max(0, min(h - 1, neg_y)))
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
            print(f"Warning: 'predict()' must be called before 'save_vis()'. "
                  f"Skipping visualization for {output_path}")
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
    parser.add_argument("--num_prompt_clusters", type=int, default=5,help="cluster on reference-level")

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
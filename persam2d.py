import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import glob
from tqdm import tqdm

from persam2_automatic_predictor import PerSAM2AutomaticPredictor
from sam2.persam2_image_predictor import SAM2ImagePredictor

# --- Loss Functions (Adapted from persam_f.py) ---
def calculate_dice_loss(inputs, targets, num_masks=1):
    # Compute the DICE loss, similar to generalized IOU for masks.
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def calculate_sigmoid_focal_loss(inputs, targets, num_masks=1, alpha: float = 0.25, gamma: float = 2):
    # Loss used in RetinaNet for dense detection
    prob = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob.flatten(1) * targets + (1 - prob.flatten(1)) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

# --- Phase 2: One-Shot Fine-Tuning ---
def train_persam2_f(
    auto_predictor: PerSAM2AutomaticPredictor,
    ref_image: np.ndarray,
    ref_mask: np.ndarray,
    iterations: int = 1000,
    lr: float = 1e-3
):
    # Fine-tunes only the visual prompt on the reference image
    print("======> Phase 1: Initializing Reference...")
    auto_predictor.set_reference(ref_image, ref_mask)
    
    device = auto_predictor.device
    predictor = auto_predictor.predictor
    model = predictor.model

    # 2. Prepare Ground Truth Mask for training
    gt_mask = torch.tensor(ref_mask).float().unsqueeze(0).to(device)
    # Ensure GT is 0/1
    gt_mask = (gt_mask > 0).float()

    # [cite_start]3. Setup learnable prompt [cite: 11]
    initial_prompt = auto_predictor.visual_prompt.clone().detach()
    finetuned_prompt = nn.Parameter(initial_prompt, requires_grad=True)
    
    # [cite_start]4. Optimizer for ONLY the prompt [cite: 12]
    optimizer = torch.optim.AdamW([finetuned_prompt], lr=lr, eps=1e-4)
    
    # [cite_start]5. Pre-calculate frozen features to keep loop efficient [cite: 14, 20]
    # We need raw features for the manual forward pass
    with torch.no_grad():
        # Ensure the predictor is set to the reference image (set_reference does this)
        ref_feat_raw = predictor._features["image_embed"]
        high_res_feats = predictor._features["high_res_feats"]
        
        # We need a spatial prompt to guide the decoder even during fine-tuning.
        # We use the auto-detected point on the reference image itself.
        h, w = ref_image.shape[:2]
        point_coords, point_labels = auto_predictor._calculate_similarity_and_get_point(
            predictor._features, initial_prompt, (h, w)
        )
        
        # Pre-encode the spatial prompt
        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )

    print(f"======> Phase 2: Start One-Shot Fine-Tuning for {iterations} iterations...")
    model.sam_mask_decoder.train(False) # Keep decoder in eval mode
    
    for i in tqdm(range(iterations)):
        optimizer.zero_grad()

        # [cite_start]Custom minimal forward pass allowing gradient flow [cite: 13]
        # [cite_start]1. Combine embeddings [cite: 14]
        current_embedding = ref_feat_raw + finetuned_prompt

        # [cite_start]2. Pass to decoder (bypassing @torch.no_grad of standard predict) [cite: 15]
        low_res_masks, _, _, _ = model.sam_mask_decoder(
            image_embeddings=current_embedding,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True, # Use multimask to allow network to find best fit
            repeat_image=False,
            high_res_features=high_res_feats,
        )
        
        # 3. Post-process masks for loss calculation
        # We need to upscale low_res_masks to original size to compare with GT
        masks = F.interpolate(
            low_res_masks,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        
        # Select the best mask if multimask output (simplified: often index 0 or 1 is best in PerSAM)
        # For stability in training, we often take the one with highest initial score or just all.
        # Here we simplify by taking the best matching mask to GT for loss to avoid mode collapse issues.
        # A simple robust way is to use the mask that is most confident or just average loss over plausible masks.
        # Let's use the first mask for simplicity in this minimal implementation, or best IoU.
        # Standard SAM usually puts best single-object mask at index 0 when multimask=False, 
        # but with multimask=True it returns 3. Let's use the one that matches GT best to optimize.
        
        # (Optional refined selection could go here, using simple index 0 for this example)
        main_mask_logits = masks[:, 0, :, :] 

        # [cite_start]4. Calculate Loss [cite: 16]
        dice_loss = calculate_dice_loss(main_mask_logits, gt_mask)
        focal_loss = calculate_sigmoid_focal_loss(main_mask_logits, gt_mask)
        loss = dice_loss + focal_loss

        loss.backward()
        optimizer.step()

        if i % 200 == 0:
             print(f"Iter {i}: Loss={loss.item():.4f} (Dice={dice_loss.item():.4f}, Focal={focal_loss.item():.4f})")

    print("======> Fine-tuning complete.")
    return finetuned_prompt.detach() # Lock the prompt 

# --- Phase 3: Fast Inference ---
    # output_path: str = None
def inference_persam2_f(
    auto_predictor: PerSAM2AutomaticPredictor,
    test_image: np.ndarray,
    finetuned_prompt: torch.Tensor,
    vis_output_path: str = None,
    mask_output_path: str = None

):
    """
    [cite_start]Run inference on a test image using the fine-tuned prompt[cite: 18].
    """
    # [cite_start]Override internal visual_prompt with our fine-tuned one [cite: 18]
    original_prompt = auto_predictor.visual_prompt
    auto_predictor.visual_prompt = finetuned_prompt

    # Run prediction
    masks, scores, logits = auto_predictor.predict(test_image)

    # Restore original prompt just in case
    auto_predictor.visual_prompt = original_prompt

    # modified 
    best_idx = np.argmax(scores)
    final_mask = masks[best_idx]

    if vis_output_path:
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(final_mask, plt.gca())
        # add click icon
        auto_point_coords, _ = auto_predictor._calculate_similarity_and_get_point(
            auto_predictor.predictor._features,
            finetuned_prompt,
            test_image.shape[:2],
        )
        x, y = int(auto_point_coords[0, 0, 0].item()), int(auto_point_coords[0, 0, 1].item())

        icon_path = "icon/click.png"
        if os.path.exists(icon_path):
            icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
            if icon is not None:
                ih, iw = icon.shape[:2]
                # compute top-left corner
                y1, y2 = max(0, y - ih // 2), min(test_image.shape[0], y + ih // 2)
                x1, x2 = max(0, x - iw // 2), min(test_image.shape[1], x + iw // 2)
                icon_resized = icon[: y2 - y1, : x2 - x1]

                # overlay icon (handle alpha)
                overlay = test_image.copy()
                if icon_resized.shape[2] == 4:
                    alpha = icon_resized[:, :, 3] / 255.0
                    for c in range(3):
                        overlay[y1:y2, x1:x2, c] = (
                            alpha * icon_resized[:, :, c] + (1 - alpha) * overlay[y1:y2, x1:x2, c]
                        )
                else:
                    overlay[y1:y2, x1:x2] = icon_resized
                plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(vis_output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    if mask_output_path:
        masks_uint8=(final_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_output_path,masks_uint8)

    return masks, scores

# --- Utils for visualization ---
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PerSAM2 Fine-tuning on PerSeg dataset structure.")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2.1_hiera_large.pt", help="Path to SAM2 checkpoint.")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 config.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root containing Images/ and Annotations/ folders.")
    parser.add_argument("--class_name", type=str, default=None, help="Specific class to run. If None, runs all found classes.")
    parser.add_argument("--ref_idx", type=str, default="00", help="Index of reference image (e.g., '00').")
    parser.add_argument("--iterations", type=int, default=1000, help="Fine-tuning iterations per class.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save results.")

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
        persam = PerSAM2AutomaticPredictor(args.sam2_checkpoint, args.model_cfg)

        print(f"--> Loading reference: {os.path.join(class_name, ref_img_name)}")
        ref_img = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)

        print(f"--> Starting fine-tuning for {class_name}...")
        finetuned_prompt = train_persam2_f(persam, ref_img, ref_mask, iterations=args.iterations)

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
            
            inference_persam2_f(
                persam, 
                test_img, 
                finetuned_prompt, 
                vis_output_path=vis_path, 
                mask_output_path=mask_path
            )

    print("\n======> All classes processed.")
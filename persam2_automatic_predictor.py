import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from sam2.build_sam import build_sam2
from sam2.persam2_image_predictor import SAM2ImagePredictor


def _unravel_index(indices: torch.Tensor, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, ...]:
    """Convert flat indices into unraveled coordinates."""
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim
    return tuple(reversed(coord))


class PerSAM2AutomaticPredictor:
    def __init__(self, sam2_model_path: str, model_config_path: str):
        """Initialize model and predictor."""
        self.model = build_sam2(model_config_path, sam2_model_path)
        self.predictor = SAM2ImagePredictor(self.model)
        self.visual_prompt: Optional[torch.Tensor] = None
        # modified by csq: init spatial_prompt part
        self.spatial_prompt: Optional[torch.Tensor] = None
        self.device = self.model.device
        self.model.eval()
        print(f"Model loaded to {self.device}")

    @torch.no_grad()
    def set_reference(self, ref_image: np.ndarray, ref_mask: np.ndarray) -> None:
        """Cache reference features and extract visual prompt."""
        self.predictor.set_image(ref_image)
        ref_features = self.predictor._features
        ref_mask_tensor = torch.as_tensor(ref_mask, dtype=torch.float, device=self.device)
        if len(ref_mask_tensor.shape) == 2:
            ref_mask_tensor = ref_mask_tensor.unsqueeze(0)
        if len(ref_mask_tensor.shape) == 3:
            ref_mask_tensor = ref_mask_tensor.unsqueeze(0)
        processed_ref_mask = self.predictor._transforms.transform_masks(ref_mask_tensor)
        self.visual_prompt = self._extract_visual_prompt(ref_features, processed_ref_mask)
        # ---modified by csq: add spatial_prompt part
        self.spatial_prompt = self._extract_spatial_prompt(processed_ref_mask)
        # ---end
        print("Reference set. Visual prompt calculated.")

    @torch.no_grad()
    def predict(self, test_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run automatic segmentation on a test image."""
        if self.visual_prompt is None:
            raise ValueError("set_reference() must be called first.")
        self.predictor.set_image(test_image)
        cached_test_features = self.predictor._features
        original_size = self.predictor._orig_hw[0]
        auto_point_coords, auto_point_labels = self._calculate_similarity_and_get_point(
            cached_test_features, self.visual_prompt, original_size
        )
        masks, iou, logits = self.predictor.predict(
            point_coords=auto_point_coords.cpu().numpy(),
            point_labels=auto_point_labels.cpu().numpy(),
            visual_prompt=self.visual_prompt,
            multimask_output=True,
        )
        return masks, iou, logits
    # ---modified by csq 1110
    def _extract_visual_prompt(
        self,
        ref_features: Dict[str, torch.Tensor],
        ref_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute visual prompt from reference mask and image features."""
        B = ref_mask.shape[0]
        features = ref_features["image_embed"]
        C, H_feat, W_feat = features.shape[1:]
        mask = F.interpolate(ref_mask, size=(H_feat, W_feat), mode="bilinear", align_corners=False)
        mask = (mask > 0.5).float()
        prompt = torch.sum(features * mask, dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-6)
        return prompt.reshape(B, C, 1, 1)
    # modified by csq: add spatial_prompt function
    def _extract_spatial_prompt(self, mask: torch.Tensor) -> torch.Tensor:
        # mask shape: [B, H, W] 或 [B, 1, H, W]
        if mask.ndim == 3:  # [B, H, W] -> [B,1,H,W]
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4:
            raise ValueError(f"Unexpected mask shape {mask.shape}, expected 3 or 4 dims")

        ref_features = self.predictor._features
        features = ref_features["image_embed"]  # [B, C, H_feat, W_feat]
        B, C, H_feat, W_feat = features.shape

        # 下采样 mask 到特征图大小
        mask_lowres = F.interpolate(mask.float(), size=(H_feat, W_feat), mode="bilinear", align_corners=False)

        # 可选二值化或归一化
        mask_lowres = mask_lowres / (mask_lowres.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-6)

        # 加权平均 pooling
        spatial_prompt = torch.sum(features * mask_lowres, dim=[2, 3]) / (mask_lowres.sum(dim=[2, 3]) + 1e-6)
        spatial_prompt = spatial_prompt.reshape(B, C, 1, 1)

        return spatial_prompt


    def _calculate_similarity_and_get_point(
        self,
        test_features: Dict[str, torch.Tensor],
        visual_prompt: torch.Tensor,
        original_image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute most similar feature location as point prompt."""
        orig_h, orig_w = original_image_size
        features = test_features["image_embed"]
        B, C, H_feat, W_feat = features.shape
        sim_map = F.cosine_similarity(features, visual_prompt, dim=1)
        sim_map_flat = sim_map.flatten(1)
        max_idx = torch.argmax(sim_map_flat, dim=1)
        y_feat, x_feat = _unravel_index(max_idx, (H_feat, W_feat))
        y_feat = y_feat.float()
        x_feat = x_feat.float()
        # scale_h = orig_h / H_feat
        # scale_w = orig_w / W_feat
        # y_orig = (y_feat + 0.5) * scale_h
        # x_orig = (x_feat + 0.5) * scale_w
        # --- modified by csq 1110
        # get model input resolution (after SAM2Transforms)
        model_res = self.predictor._transforms.resolution
        scale_h = model_res / orig_h
        scale_w = model_res / orig_w
        y_model = (y_feat + 0.5) * (model_res / H_feat)
        x_model = (x_feat + 0.5) * (model_res / W_feat)
        y_orig = y_model / scale_h
        x_orig = x_model / scale_w
        # --- end modified ---
        coords = torch.stack([x_orig, y_orig], dim=1)
        auto_point_coords = coords.unsqueeze(1)
        auto_point_labels = torch.ones(B, 1, device=self.device, dtype=torch.int)
        return auto_point_coords, auto_point_labels

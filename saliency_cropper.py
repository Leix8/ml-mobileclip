# saliency_crop.py
import cv2
import numpy as np
import torch 
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    _HAS_SAM = True
except ImportError:
    _HAS_SAM = False

try:
    from groundingdino.util.inference import load_model, predict
    _HAS_GDINO = True
except ImportError:
    _HAS_GDINO = False


class SaliencyCropper:
    def __init__(self, method="opencv", device="cuda",
                 sam_checkpoint=None, model_type="vit_b",
                 gdino_config="/home/leixu/workspace/workspace_leixu/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                 gdino_ckpt="/home/leixu/workspace/workspace_leixu/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth", text_prompt=None):
        """
        Args:
            method: "opencv", "sam", or "groundingdino"
            device: "cuda" or "cpu"
            sam_checkpoint: path to sam_vit_b.pth or sam_vit_h.pth.
            model_type: SAM model type (vit_b, vit_h, etc.)
            gdino_config: config file for Grounding DINO
            gdino_ckpt: checkpoint file for Grounding DINO
            text_prompt: text used for bbox guidance (Grounding DINO only)
        """
        self.method = method
        self.text_prompt = text_prompt
        self.device = device

        if method == "sam":
            if not _HAS_SAM:
                raise ImportError("SAM not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git")
            if sam_checkpoint is None:
                raise ValueError("sam_checkpoint must be provided for SAM.")
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

        elif method == "groundingdino":
            if not _HAS_GDINO:
                raise ImportError("GroundingDINO not installed. Run: pip install groundingdino-py")
            if gdino_config is None or gdino_ckpt is None or text_prompt is None:
                raise ValueError("gdino_config, gdino_ckpt and text_prompt must be provided for GroundingDINO.")
            self.gdino_model = load_model(gdino_config, gdino_ckpt)
        else:
            self.mask_generator = None

    def crop(self, img_bgr: np.ndarray):
        """Run saliency detection and return cropped image (BGR, bbox_norm)."""
        if self.method == "sam":
            return self._crop_sam(img_bgr)
        elif self.method == "groundingdino":
            return self._crop_gdino(img_bgr)
        else:
            return self._crop_opencv(img_bgr)

    def expand_bbox(self, x, y, w, h, img_w, img_h, scale=1.2):
        """Expand bbox by scale while keeping it inside image bounds."""
        cx, cy = x + w/2, y + h/2
        new_w, new_h = w * scale, h * scale
        x1 = int(max(0, cx - new_w/2))
        y1 = int(max(0, cy - new_h/2))
        x2 = int(min(img_w, cx + new_w/2))
        y2 = int(min(img_h, cy + new_h/2))
        return x1, y1, x2 - x1, y2 - y1

    def _crop_opencv(self, img_bgr):
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(img_bgr)
        if not success:
            return img_bgr, None
        saliency_map = (saliency_map * 255).astype("uint8")
        _, mask = cv2.threshold(saliency_map, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return img_bgr, None
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_h, img_w = img_bgr.shape[:2]
        x, y, w, h = self.expand_bbox(x, y, w, h, img_w, img_h, scale=1.1)
        return img_bgr[y:y+h, x:x+w], (x/img_w, y/img_h, w/img_w, h/img_h)

    def _crop_sam(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(img_rgb)
        if len(masks) == 0:
            return img_bgr, None
        largest_mask = max(masks, key=lambda x: x['area'])
        bx, by, bw, bh = largest_mask['bbox']
        x, y, w, h = self.expand_bbox(bx, by, bw, bh,
                                      img_w=img_bgr.shape[1], img_h=img_bgr.shape[0],
                                      scale=1.2)
        img_h, img_w = img_bgr.shape[:2]
        return img_bgr[y:y+h, x:x+w], (x/img_w, y/img_h, w/img_w, h/img_h)

    def _crop_gdino(self, img_bgr):
        """
        GroundingDINO crop guided by text. Robust:
        - handles normalized + unordered corners
        - retries with looser thresholds / prompt variants
        - ranks boxes by score*area to avoid tiny parts
        - clamps/validates; falls back to full image
        Returns: (crop_bgr, bbox_norm_xywh)
        """
        img_h, img_w = img_bgr.shape[:2]

        # ---- to tensor [3,H,W], float in [0,1] then ImageNet normalize ----
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std  # what DINO expects

        # ---- try several prompts & thresholds for recall ----
        prompts = [self.text_prompt,
                f"a {self.text_prompt}" if self.text_prompt else None,
                f"a photo of {self.text_prompt}" if self.text_prompt else None]
        prompts = [p for p in prompts if p]

        thresholds = [(0.30, 0.25), (0.20, 0.20), (0.10, 0.10)]

        best_xyxy = None
        for cap in prompts:
            for box_th, txt_th in thresholds:
                try:
                    boxes, logits, phrases = predict(
                        model=self.gdino_model,
                        image=t,                # torch.Tensor [3,H,W], normalized
                        caption=cap.lower(),
                        box_threshold=box_th,
                        text_threshold=txt_th,
                        device=self.device
                    )
                except Exception:
                    continue

                if len(boxes) == 0:
                    continue

                # tensors -> numpy
                boxes_np = boxes.cpu().numpy()              # [N, 4], normalized xyxy
                if torch.is_tensor(logits):
                    scores = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                else:
                    scores = np.asarray(logits).reshape(-1)

                # reorder corners, scale to pixels, clamp, validate
                x1n, y1n, x2n, y2n = (boxes_np[:, 0], boxes_np[:, 1],
                                    boxes_np[:, 2], boxes_np[:, 3])
                x1 = (np.minimum(x1n, x2n) * img_w).astype(int)
                y1 = (np.minimum(y1n, y2n) * img_h).astype(int)
                x2 = (np.maximum(x1n, x2n) * img_w).astype(int)
                y2 = (np.maximum(y1n, y2n) * img_h).astype(int)

                x1 = np.clip(x1, 0, img_w - 1); x2 = np.clip(x2, 0, img_w - 1)
                y1 = np.clip(y1, 0, img_h - 1); y2 = np.clip(y2, 0, img_h - 1)

                w = (x2 - x1); h = (y2 - y1)
                valid = (w > 5) & (h > 5)
                if not np.any(valid):
                    continue

                # prefer big & confident boxes (avoid “tiny paw”)
                area = w * h
                idx_valid = np.where(valid)[0]
                pick = idx_valid[np.argmax(scores[idx_valid] * area[idx_valid])]

                best_xyxy = (int(x1[pick]), int(y1[pick]), int(x2[pick]), int(y2[pick]))
                break  # stop on first valid pick for this prompt
            if best_xyxy:
                break

        # ---- fallback: full image ----
        if not best_xyxy:
            return img_bgr, (0.0, 0.0, 1.0, 1.0)

        x1, y1, x2, y2 = best_xyxy
        w,  h  = x2 - x1, y2 - y1

        # expand slightly for context
        x1, y1, w, h = self.expand_bbox(x1, y1, w, h, img_w, img_h, scale=1.15)

        crop = img_bgr[y1:y1+h, x1:x1+w]
        bbox_norm = (x1 / img_w, y1 / img_h, w / img_w, h / img_h)
        return crop, bbox_norm
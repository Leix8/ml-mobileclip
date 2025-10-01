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
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, infer_device
    _HAS_GDINO = True
except ImportError:
    _HAS_GDINO = False

class SaliencyCropper:
    def __init__(self, method="opencv", device="cuda",
                 sam_checkpoint=None, model_type="vit_b", text_prompt=None):
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
        self.device = device
        if method == "sam":
            if not _HAS_SAM:
                raise ImportError("SAM not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git")
            if sam_checkpoint is None:
                raise ValueError("sam_checkpoint must be provided for SAM.")
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

        elif method == "gdino":
            if not _HAS_GDINO:
                raise ImportError("GroundingDINO not installed.")
            self.gdino_model_id = "IDEA-Research/grounding-dino-tiny"
            self.gdino_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_model_id).to(self.device)
            self.text_prompt = text_prompt  
        
        elif method == "opencv":
            self.mask_generator = None

    def crop(self, img_bgr: np.ndarray):
        """Run saliency detection and return cropped image (BGR, bbox_norm)."""
        if self.method == "sam":
            return self._crop_sam(img_bgr)
        elif self.method == "gdino":
            return self._crop_gdino(img_bgr)
        elif self.method == "opencv":
            return self._crop_opencv(img_bgr)
        else:
            return self._crop_base(img_bgr)

    def expand_bbox(self, x, y, w, h, img_w, img_h, scale=1.2):
        """Expand bbox by scale while keeping it inside image bounds."""
        cx, cy = x + w/2, y + h/2
        new_w, new_h = w * scale, h * scale
        x1 = int(max(0, cx - new_w/2))
        y1 = int(max(0, cy - new_h/2))
        x2 = int(min(img_w, cx + new_w/2))
        y2 = int(min(img_h, cy + new_h/2))
        return x1, y1, x2 - x1, y2 - y1

    def _crop_base(self, img_bgr):
        return img_bgr, (0.0, 0.0, 1.0, 1.0)
    
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
        x, y, w, h = self.expand_bbox(x, y, w, h, img_w, img_h, scale=1.2)
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

        prompts = [self.text_prompt]
        prompts = [p for p in prompts if p]

        inputs = self.gdino_processor(images=img_rgb, text=prompts, return_tensors="pt").to(self.gdino_model.device)
        with torch.no_grad():
            outputs = self.gdino_model(**inputs)
        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes= [img_rgb.shape[:2]]
        )

        best_xyxy = (0.0, 0.0, 1.0, 1.0)
        best_score = -1
        for result in results:
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                if score > best_score:
                    best_score = score
                    # Convert tensor to list
                    best_xyxy = [int(x) for x in box.tolist()]  # [x1, y1, x2, y2]
                    
        x1, y1, x2, y2 = best_xyxy
        w,  h  = x2 - x1, y2 - y1

        # expand slightly for context
        x1, y1, w, h = self.expand_bbox(x1, y1, w, h, img_w, img_h, scale=1.2)

        crop = img_bgr[y1:y1+h, x1:x1+w]
        bbox_norm = (x1 / img_w, y1 / img_h, w / img_w, h / img_h)
        return crop, bbox_norm
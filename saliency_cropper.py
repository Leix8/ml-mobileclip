# saliency_crop.py
import cv2
import numpy as np

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    _HAS_SAM = True
except ImportError:
    _HAS_SAM = False


class SaliencyCropper:
    def __init__(self, method="opencv", device="cuda", sam_checkpoint=None, model_type="vit_b"):
        """
        Args:
            method: "opencv" (fast, light) or "sam" (accurate, heavy).
            device: "cuda" or "cpu" for SAM.
            sam_checkpoint: path to sam_vit_b.pth or sam_vit_h.pth.
            model_type: SAM model type (vit_b, vit_h, etc.)
        """
        self.method = method

        if method == "sam":
            if not _HAS_SAM:
                raise ImportError("SAM not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git")
            if sam_checkpoint is None:
                raise ValueError("sam_checkpoint must be provided for SAM.")
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
        else:
            self.mask_generator = None

    def crop(self, img_bgr: np.ndarray) -> np.ndarray:
        """Run saliency detection and return cropped image (BGR)."""
        if self.method == "sam":
            return self._crop_sam(img_bgr)
        else:
            return self._crop_opencv(img_bgr)

    def expand_bbox(self, x, y, w, h, img_w, img_h, scale=1.2):
        """
        Expand bbox by a scale factor while keeping it inside image bounds.
        """
        cx, cy = x + w/2, y + h/2  # center
        new_w, new_h = w * scale, h * scale

        # clamp to image bounds
        x1 = int(max(0, cx - new_w/2))
        y1 = int(max(0, cy - new_h/2))
        x2 = int(min(img_w, cx + new_w/2))
        y2 = int(min(img_h, cy + new_h/2))

        return x1, y1, x2 - x1, y2 - y1
        
    def _crop_opencv(self, img_bgr):
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(img_bgr)
        if not success:
            return img_bgr
        saliency_map = (saliency_map * 255).astype("uint8")
        _, mask = cv2.threshold(saliency_map, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return img_bgr
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_h, img_w = img_bgr.shape[:2]
        x, y, w, h = self.expand_bbox(x, y, w, h, img_w, img_h, scale=scale)

        return img_bgr[y:y+h, x:x+w]

    def _crop_sam(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(img_rgb)
        if len(masks) == 0:
            return img_bgr
        largest_mask = max(masks, key=lambda x: x['area'])
        bx, by, bw, bh = largest_mask['bbox']
        x, y, w, h = self.expand_bbox(
                bx, by, bw, bh,
                img_w = img_bgr.shape[1], img_h = img_bgr.shape[0],
                scale = 1.2
            )
        return img_bgr[y:y+h, x:x+w]

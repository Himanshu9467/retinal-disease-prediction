"""
gradcam.py — Grad-CAM Explainability for CVD Classification Models
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import CLASS_NAMES


class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd(m, i, o): self.activations = o.detach()
        def bwd(m, gi, go): self.gradients = go[0].detach()
        self._hooks.append(self.target_layer.register_forward_hook(fwd))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._hooks: h.remove()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced non-finite logits during GradCAM")

        flat_logits = logits.float().view(logits.size(0), -1)
        if flat_logits.size(1) == 1:
            disease_score = flat_logits[0, 0]
            if target_class is None:
                target_class = int(torch.sigmoid(disease_score).item() >= 0.5)
            score = disease_score if int(target_class) == 1 else -disease_score
        else:
            if target_class is None:
                target_class = int(flat_logits.argmax(dim=1).item())
            score = flat_logits[0, int(target_class)]

        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture activations and gradients")
        if self.gradients.ndim != 4 or self.activations.ndim != 4:
            raise RuntimeError("GradCAM target layer must produce 4D feature maps")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam).squeeze().cpu().numpy()
        cam    -= cam.min()
        cam    /= (cam.max() + 1e-8)
        return cam, target_class


def overlay_cam(img_pil, cam, alpha=0.5):
    img_np      = np.array(img_pil.convert("RGB"))
    h, w        = img_np.shape[:2]
    if cam is None or not np.isfinite(cam).all():
        raise RuntimeError("Invalid GradCAM heatmap")
    cam_resized = cv2.resize(cam, (w, h))
    cam_resized = _mask_to_retinal_field(img_np, cam_resized)
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap     = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlaid    = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(overlaid)


def _mask_to_retinal_field(img_np, cam):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
    masked = cam * mask
    masked -= masked.min()
    masked /= masked.max() + 1e-8
    return masked


def _last_conv_layer(module):
    for child in reversed(list(module.children())):
        if isinstance(child, torch.nn.Conv2d):
            return child
        found = _last_conv_layer(child)
        if found is not None:
            return found
    return None


def _module_path(root, path):
    module = root
    for part in path:
        module = module[part] if isinstance(part, int) else getattr(module, part)
    return module


def get_target_layer(model, model_name):
    name = model_name.lower()
    base_model = getattr(model, "model", model)
    if name in {"cnn", "custom_cnn", "customcnn"}:
        return base_model.features[-2]
    elif name == "resnet":
        return base_model.layer4[-1]
    elif name == "efficientnet":
        return _module_path(base_model.features, [6, -1])
    elif name in {"efficientnet", "mobilenet", "mobilenetv3", "mobilenetv3_large"}:
        return base_model.features[-1]
    else:
        raise ValueError(f"Grad-CAM not supported for '{name}'. Use ViT Attention Rollout.")


def visualise_gradcam(model, image_path, target_layer,
                      img_size=224, save_path=None, predicted_class=None, confidence=None):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img_pil   = Image.open(image_path).convert("RGB")
    input_t   = transform(img_pil).unsqueeze(0)
    device    = next(model.parameters()).device
    input_t   = input_t.to(device)

    cam_gen   = GradCAM(model, target_layer)
    cam, cls  = cam_gen.generate(input_t)
    cam_gen.remove_hooks()
    overlaid  = overlay_cam(img_pil, cam)

    cls_name  = predicted_class if predicted_class else CLASS_NAMES[cls]
    title_str = f"Prediction: {cls_name}"
    if confidence:
        title_str += f" ({confidence:.1%} confidence)"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_pil.resize((img_size, img_size)))
    axes[0].set_title("Original Retinal Image")
    axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    axes[2].imshow(overlaid.resize((img_size, img_size)))
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.suptitle(f"Grad-CAM Explainability — {title_str}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return cam, overlaid, cls


class ViTAttentionRollout:
    def __init__(self, model, discard_ratio=0.9):
        self.model         = model
        self.discard_ratio = discard_ratio
        self.attentions    = []
        self._hooks        = []
        base_model = getattr(self.model, "model", self.model)
        encoder = getattr(base_model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            raise ValueError("ViT attention rollout is not supported by this model")
        for layer in layers:
            self._hooks.append(
                layer.self_attention.register_forward_hook(
                    lambda m, i, o: self._capture_attention(o)
                )
            )

    def _capture_attention(self, output):
        tensor = output[1] if isinstance(output, tuple) and len(output) > 1 else output
        if torch.is_tensor(tensor):
            self.attentions.append(tensor.detach().cpu())

    def remove_hooks(self):
        for h in self._hooks: h.remove()

    def generate(self, input_tensor):
        self.attentions = []
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        if not self.attentions:
            raise RuntimeError("ViT attention maps were not captured")

        result = torch.eye(self.attentions[0].size(-1))
        for attn in self.attentions:
            attn_fused = attn.mean(dim=1)
            flat = attn_fused.view(attn_fused.size(0), -1)
            _, idx = flat.topk(int(flat.size(-1) * self.discard_ratio), dim=-1, largest=False)
            flat.scatter_(1, idx, 0)
            attn_fused = flat.view(attn_fused.size())
            attn_fused = attn_fused + torch.eye(attn_fused.size(-1))
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn_fused[0], result)

        mask = result[0, 1:]
        n    = int(mask.size(0) ** 0.5)
        mask = mask.reshape(n, n).numpy()
        mask -= mask.min()
        mask /= (mask.max() + 1e-8)
        return mask

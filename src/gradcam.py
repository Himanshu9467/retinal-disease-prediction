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

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

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
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap     = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlaid    = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(overlaid)


def get_target_layer(model, model_name):
    name = model_name.lower()
    base_model = getattr(model, "model", model)
    if name in {"cnn", "custom_cnn", "customcnn"}:
        return base_model.features[12]
    elif name == "resnet":
        return getattr(base_model.layer4[-1], "conv3", base_model.layer4[-1].conv2)
    elif name == "efficientnet":return base_model.features[-1]
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

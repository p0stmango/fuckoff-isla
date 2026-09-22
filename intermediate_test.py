import torch
from PIL import Image
import numpy as np
from dataset import AUSynthDataset, val_transforms, ALL_SPEEDS
from model import load as load_model, get_device
from eot import eot_batch
from patch_attack import apply_patch, make_circular_mask

device = get_device()
model = load_model("surrogate.pt").to(device)
model.eval()

patch_01 = torch.load("patch.pt")  # or whatever checkpoint
# if you only have the PNG:
#patch_01 = torch.tensor(np.array(Image.open("patch_step_2850.png")).astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)

mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
patch_norm = (patch_01.to(device) - mean) / std
mask = make_circular_mask(48, device)

ds = AUSynthDataset("./data/synthetic_aus_signs", transform=val_transforms, split="val")
img, label = ds[0]  # grab a 5km/h sign
imgs = img.unsqueeze(0).to(device)

# Apply patch + EOT and visualise
patched = apply_patch(imgs, patch_norm, mask, randomise_placement=True)
patched_01 = patched * std + mean
patched_eot = eot_batch(patched_01)

pred = model((patched_eot - mean)/std).argmax(1).item()
print(f"True: {ALL_SPEEDS[label]} km/h → Predicted: {ALL_SPEEDS[pred]} km/h")

# Save the visual
arr = patched_eot.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()
Image.fromarray((arr*255).astype(np.uint8)).save("test_overlay.png")
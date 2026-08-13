import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from medigan import Generators

# ==========================================
# 1. CUSTOM cGAN ARCHITECTURE & LOADER
# ==========================================
class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=4, embed_size=100, img_channels=1):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, embed_size)

        self.model = nn.Sequential(
            self._block(z_dim + embed_size, 1024, kernel_size=4, stride=1, padding=0),
            self._block(1024, 512, kernel_size=4, stride=2, padding=1),
            self._block(512, 256, kernel_size=4, stride=2, padding=1),
            self._block(256, 128, kernel_size=4, stride=2, padding=1),
            self._block(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        noise = noise.unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, c], dim=1)
        return self.model(x)


@st.cache_resource
def load_custom_generator(weights_path):
    device = torch.device("cpu")
    model = ConditionalGenerator(z_dim=100, num_classes=4, embed_size=100, img_channels=1)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        return model, device
    else:
        return None, device


def generate_custom_images(num_images, class_idx, weights_path):
    model, device = load_custom_generator(weights_path)
    if model is None:
        st.error(f"Checkpoint file not found at `{weights_path}`. Please verify the weight path.")
        return []

    images = []
    with torch.no_grad():
        for _ in range(num_images):
            noise = torch.randn(1, 100, device=device)
            label = torch.tensor([class_idx], dtype=torch.long, device=device)
            fake_tensor = model(noise, label).detach().cpu()

            # Denormalize from [-1, 1] to [0, 1]
            fake_tensor = (fake_tensor + 1) / 2
            fake_tensor = fake_tensor.squeeze(0)  # Shape: (1, 128, 128)

            pil_img = to_pil_image(fake_tensor).convert("RGB")
            images.append(pil_img)

    return images


# ==========================================
# 2. MEDIGAN GENERATION FUNCTION
# ==========================================
def torch_images(num_images, model_id):
    generators = Generators()
    dataloader = generators.get_as_torch_dataloader(
        model_id=model_id,
        install_dependencies=False,
        num_samples=num_images,
        prefetch_factor=None,
    )

    images = []
    for batch_idx, data_dict in enumerate(dataloader):
        image_list = []
        for i in data_dict:
            if "sample" in i:
                sample = data_dict.get("sample")
                if sample.dim() == 4:
                    sample = sample.squeeze(0).permute(2, 0, 1)

                sample = to_pil_image(sample).convert("RGB")
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ])
                sample = transform(sample)
                image_list.append(sample)

            if "mask" in i:
                mask = data_dict.get("mask")
                if mask.dim() == 4:
                    mask = mask.squeeze(0).permute(2, 0, 1)
                mask = to_pil_image(mask).convert("RGB")
                mask = transform(mask)
                image_list.append(mask)

        Grid = make_grid(image_list, nrow=2)

        if Grid.dim() == 4:
            Grid = Grid.squeeze(0)
            if Grid.size(-1) == 1:
                Grid = Grid.squeeze(-1)
            else:
                raise ValueError("Expected a single channel (grayscale) image.")

        img = torchvision.transforms.ToPILImage()(Grid)
        images.append(img)

        # Enforce exact batch limit to override MEDIGAN defaults
        if len(images) >= num_images:
            break

    return images[:num_images]


# ==========================================
# 3. STREAMLIT INTERFACE (ANALYTICS LAYOUT)
# ==========================================
medigan_model_ids = [
    "00007_INPAINT_BRAIN_MRI",
    "00021_CYCLEGAN_BRAIN_MRI_T1_T2",
]

CUSTOM_WEIGHTS_PATH = "Brain_Tumor_cGAN_Checkpoints/gen_epoch_150.pth"


def main():
    # Force a wide layout for better data visualization
    st.set_page_config(layout="wide", page_title="MRI Generator Dashboard")
    st.title("Brain MRI Image Generation")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.image("Images/GANs.jpg",width=350)
    with col2:
        st.image("Images/brain.jpg",width=350)

    # Define layout: Controls (left) and Display panel (right) equally split 50/50
    col_controls, col_display = st.columns(2)

    # --- LEFT PANEL: CONFIGURATION ---
    with col_controls:
        st.subheader("Configuration Parameters")

        generator_source = st.radio(
            "Select Architecture Engine",
            ["Custom cGAN (Brain Tumor)", "MEDIGAN Library"]
        )

        num_images = st.number_input(
            "Target Output Size", min_value=1, max_value=7, value=1, step=1
        )

        # Render conditional controls based on the selected framework
        if generator_source == "Custom cGAN (Brain Tumor)":
            class_map = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}
            selected_class = st.selectbox("Select Tumor Target Class", list(class_map.values()))
            class_idx = [k for k, v in class_map.items() if v == selected_class][0]

            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("Initialize Custom Pipeline", type="primary", use_container_width=True)
        else:
            model_id = st.selectbox("Select MEDIGAN Registry ID", medigan_model_ids)

            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("Initialize External Pipeline", type="primary", use_container_width=True)

    # --- RIGHT PANEL: OUTPUT DISPLAY ---
    with col_display:
        st.subheader("Visual Output Data")
        # Create an empty placeholder container to inject images into later
        display_window = st.container()

    # --- INJECT GENERATION RESULTS INTO THE RIGHT PANEL ---
    with display_window:
        if generate_btn:
            if generator_source == "Custom cGAN (Brain Tumor)":
                with st.spinner("Synthesizing MRI tensors..."):
                    images = generate_custom_images(num_images, class_idx, CUSTOM_WEIGHTS_PATH)

                    if images:
                        # Render multiple output images side-by-side within the right half
                        grid_cols = st.columns(len(images))
                        for i, (col, img) in enumerate(zip(grid_cols, images)):
                            with col:
                                st.image(img, caption=f"Custom cGAN — Class: {selected_class}",
                                         use_container_width=True)

            else:
                with st.spinner(f"Pulling pre-trained weights from {model_id}..."):
                    images = torch_images(num_images, model_id)

                    if images:
                        # Render multiple output images side-by-side within the right half
                        grid_cols = st.columns(len(images))
                        for i, (col, img) in enumerate(zip(grid_cols, images)):
                            with col:
                                st.image(img, caption=f"MEDIGAN Model: {model_id}", use_container_width=True)
        else:
            st.info("System idle. Awaiting configuration input from the control panel.")


if __name__ == "__main__":
    main()

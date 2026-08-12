import os
import gradio as gr
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
            self._block(z_dim + embed_size, 1024, 4, 1, 0),
            self._block(1024, 512, 4, 2, 1),
            self._block(512, 256, 4, 2, 1),
            self._block(256, 128, 4, 2, 1),
            self._block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
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


def load_custom_generator(weights_path):
    # Dynamically map to CUDA if the space provisions a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalGenerator(z_dim=100, num_classes=4, embed_size=100, img_channels=1)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)  # Ensure the model itself is moved to the GPU
        model.eval()
        return model, device
    return None, device


def generate_custom_images(num_images, class_idx, weights_path):
    model, device = load_custom_generator(weights_path)
    if model is None:
        raise gr.Error(f"Checkpoint not found at {weights_path}")

    images = []
    with torch.no_grad():
        for _ in range(int(num_images)):
            noise = torch.randn(1, 100, device=device)
            label = torch.tensor([class_idx], dtype=torch.long, device=device)
            fake_tensor = model(noise, label).detach().cpu()
            fake_tensor = (fake_tensor + 1) / 2
            fake_tensor = fake_tensor.squeeze(0)
            pil_img = to_pil_image(fake_tensor).convert("RGB")
            images.append(pil_img)
    return images


# ==========================================
# 2. MEDIGAN GENERATION FUNCTION
# ==========================================
def torch_images(num_images, model_id):
    generators = Generators()
    dataloader = generators.get_as_torch_dataloader(
        model_id=model_id, install_dependencies=True, num_samples=int(num_images), prefetch_factor=None
    )
    images = []
    for batch_idx, data_dict in enumerate(dataloader):
        image_list = []
        for i in data_dict:
            if "sample" in i:
                sample = data_dict.get("sample")
                if sample.dim() == 4: sample = sample.squeeze(0).permute(2, 0, 1)
                sample = to_pil_image(sample).convert("RGB")
                sample = torchvision.transforms.ToTensor()(sample)
                image_list.append(sample)

        Grid = make_grid(image_list, nrow=2)
        if Grid.dim() == 4:
            Grid = Grid.squeeze(0)
            if Grid.size(-1) == 1: Grid = Grid.squeeze(-1)

        img = torchvision.transforms.ToPILImage()(Grid)
        images.append(img)
        if len(images) >= num_images: break
    return images[:int(num_images)]


# ==========================================
# 3. GRADIO ROUTING LOGIC (WITH GPU DECORATOR)
# ==========================================
CUSTOM_WEIGHTS_PATH = "Brain_Tumor_cGAN_Checkpoints/gen_epoch_150.pth"


# The decorator must wrap the function directly called by the Gradio event

def run_pipeline(engine, num_images, target_class, medigan_id):
    if engine == "Custom cGAN (Brain Tumor)":
        class_map = {'Glioma': 0, 'Meningioma': 1, 'No Tumor': 2, 'Pituitary': 3}
        class_idx = class_map[target_class]
        return generate_custom_images(num_images, class_idx, CUSTOM_WEIGHTS_PATH)
    else:
        return torch_images(num_images, medigan_id)


def toggle_inputs(engine):
    if engine == "Custom cGAN (Brain Tumor)":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


# ==========================================
# 4. GRADIO INTERFACE (ANALYTICS LAYOUT)
# ==========================================
theme = gr.themes.Monochrome(
    primary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(theme=theme, title="MRI Generator Dashboard") as demo:
    gr.Markdown("# Brain MRI Image Generation")
    gr.Markdown("---")

    with gr.Row():
        # --- LEFT PANEL: CONFIGURATION ---
        with gr.Column(scale=1):
            gr.Markdown("### Configuration Parameters")

            engine_radio = gr.Radio(
                choices=["Custom cGAN (Brain Tumor)", "MEDIGAN Library"],
                value="Custom cGAN (Brain Tumor)",
                label="Select Architecture Engine"
            )

            num_images_input = gr.Number(
                value=1, minimum=1, maximum=7, step=1,
                label="Target Output Size"
            )

            target_class_dropdown = gr.Dropdown(
                choices=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
                value='Glioma',
                label="Select Tumor Target Class",
                visible=True
            )

            medigan_id_dropdown = gr.Dropdown(
                choices=["00007_INPAINT_BRAIN_MRI", "00021_CYCLEGAN_BRAIN_MRI_T1_T2"],
                value="00007_INPAINT_BRAIN_MRI",
                label="Select MEDIGAN Registry ID",
                visible=False
            )

            generate_btn = gr.Button("Initialize Pipeline", variant="primary")

        # --- RIGHT PANEL: VISUAL OUTPUT ---
        with gr.Column(scale=1):
            gr.Markdown("### Visual Output Data")
            output_gallery = gr.Gallery(
                label="Generated MRI Tensors",
                show_label=False,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )

    # Event Listeners
    engine_radio.change(
        fn=toggle_inputs,
        inputs=[engine_radio],
        outputs=[target_class_dropdown, medigan_id_dropdown]
    )

    generate_btn.click(
        fn=run_pipeline,
        inputs=[engine_radio, num_images_input, target_class_dropdown, medigan_id_dropdown],
        outputs=[output_gallery]
    )

if __name__ == "__main__":
    demo.launch()
# MRI cGAN Image Generation Dashboard

A robust, interactive deep learning pipeline for synthesizing medical imaging data. This application provides a web-based dashboard to generate synthetic Brain MRI scans using both custom-trained Conditional Generative Adversarial Networks (cGANs) and pre-trained models from the MEDIGAN library.

Built with real-world implementation in mind, this project demonstrates end-to-end model deployment, tensor processing, and dynamic visual analytics without relying on rigid theoretical frameworks.

##  Features

*   **Dual-Engine Architecture:**
    *   **Custom cGAN:** A natively trained conditional GAN capable of synthesizing MRIs across four specific tumor classifications (Glioma, Meningioma, No Tumor, Pituitary) from random noise vectors.
    *   **MEDIGAN Integration:** Direct access to external pre-trained medical imaging models (e.g., Inpainting, CycleGANs) for broader synthetic data generation.
*   **Interactive UI:** A streamlined, analytics-focused Streamlit dashboard for real-time inference and visualization.
*   **Cloud-Ready Pipeline:** Configured with specific dependency overrides and dynamic monkey-patching to allow heavy PyTorch/OpenCV rendering inside restricted cloud environments.

##  Tech Stack

*   **Deep Learning Framework:** PyTorch, Torchvision
*   **Medical Imaging Library:** MEDIGAN
*   **Web Framework:** Streamlit
*   **Image Processing:** OpenCV (Headless), Albumentations, PIL

##  Project Structure

```text
MRI_cGAN_Image_Generation/
├── app.py                                  # Main Streamlit application and UI layout
├── requirements.txt                        # Strict dependency pins for cloud deployment
├── Brain_Tumor_cGAN_Checkpoints/
│   └── gen_epoch_150.pth                   # Custom cGAN model weights (Git LFS)
├── Images/
│   ├── GANs.jpg                            # Dashboard UI asset
│   └── brain.jpg                           # Dashboard UI asset
└── README.md

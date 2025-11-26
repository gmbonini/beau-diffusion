# Operations and Maintenance Guide

This document outlines the procedures for provisioning models, updating the Gradio application, and maintaining the compute instance.

## 1. Model Provisioning

The pipeline relies on specific checkpoints for Qwen, Flux, and RealVis.

### Automated Provisioning
The `setup_project.sh` script is designed to automatically download the required model weights and LoRAs. Ensure you have sufficient disk space before running the setup.

### Manual Provisioning
If manual setup is required, ensure the following models are placed in the `models/` directory:
* **LLM:** Qwen 2.5VL:7b and Qwen 2.5:7b
* **Flux:** Flux Schnell
* **RealVis:** RealVisXL_V5.0 + artificialguybr/3DRedmond-V1 LoRA
* **3D Model:** TRELLIS weights

## 2. Updating the Gradio Application

To update the web interface and pipeline logic:

1. **Stop the current process:**
   Terminate the running `run_pipeline.sh` script.

2. **Pull the latest code:**
   
       git pull origin main

3. **Update dependencies:**
   If `requirements.txt` has changed, reinstall dependencies:

       pip install -r requirements.txt

   > **Critical Note:** The models currently rely on specific PyTorch versions defined in the `setup_project.sh` script. **Do not modify or upgrade these versions manually**, as the pipeline is validated strictly against this environment.

4. **Restart the application:**
   
       ./run_pipeline.sh

## 3. Instance Maintenance

### GPU Driver Updates
Since this project runs on an **NVIDIA A100**, keeping CUDA drivers up to date is critical.
1. Check current driver version: `nvidia-smi`
2. If an update is required, follow the official NVIDIA data center driver installation guide.
3. After driver updates, a system reboot is often required.
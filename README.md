# Text-to-3D Generation Pipeline

This project implements an automated pipeline for generating 3D meshes from text prompts. It integrates Large Language Models (LLMs) for prompt refinement, intelligent image model selection, and TRELLIS for 3D mesh reconstruction.

## Workflow Overview

The system operates through the following stages:

1. **Prompt Input & Refinement**
   The user provides an initial text prompt. This is processed and refined by **Qwen 2.5VL:7b** to optimize it for image generation.

2. **Model Decision & Generation**
   Based on the content of the refined prompt, the system automatically selects the appropriate text-to-image model:
   * **Flux Schnell**: Selected for scenes, landscapes, buildings, and complex scenarios.
   * **RealVisXL_V5.0** (with `artificialguybr/3DRedmond-V1 LoRA`): Selected for characters, simple objects, and animals.

3. **Validation Loop**
   The generated images undergo a multiview validation process:
   * **Undesired results:** The system regenerates the images using a different random seed.
   * **User changes:** If the user modifies the prompt, the workflow restarts at the refinement stage.
   * **Positive result:** The workflow proceeds to the 3D generation stage.

4. **3D Mesh Generation**
   Approved images are processed by **TRELLIS** to convert multiview data into a 3D mesh, followed by a final mesh validation step.

## Hardware Specifications

This pipeline has been tested and executed on the following hardware configuration:
* **CPU:** AMD EPYC (16 Cores)
* **GPU:** NVIDIA A100

## Setup and Usage

### Installation
To set up the environment and install necessary dependencies, run the setup script:

    ./setup_project.sh

### Running the Pipeline
After installation, verify the configuration in the run script and execute the pipeline:

    ./run_pipeline.sh

## Maintenance

### Updating the Application
To update the Gradio interface and project code to the latest version, verify the steps in the `MAINTENANCE_GUIDE.md` file. Generally, this involves pulling the latest changes and restarting the service.

### Instance Updates
Regular updates to the compute instance (drivers and OS) are recommended to ensure compatibility with the NVIDIA A100 GPU. Refer to the operations guide for detailed instructions.
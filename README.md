# Text-to-3D Generation Pipeline

This project implements an automated pipeline for generating 3D meshes from text prompts. It integrates Large Language Models (LLMs) for prompt refinement, intelligent image model selection, and TRELLIS for 3D mesh reconstruction.

## Workflow Overview

This project implements an automated end-to-end pipeline that converts natural language descriptions into high-quality 3D meshes. The system intelligently refines prompts, generates consistent multiview images, and reconstructs them into 3D models.

##  Architecture

The system consists of several key components:

- **FastAPI Backend**: RESTful API for text-to-multiview and multiview-to-mesh generation
- **MV-Adapter**: Text-to-multiview image generation using Stable Diffusion XL
- **TRELLIS**: Microsoft's multiview-to-3D mesh reconstruction pipeline
- **Ollama Integration**: LLM-powered prompt refinement and image quality validation
- **MySQL Database**: Feedback storage and multiview image management

##  Workflow

The pipeline operates through the following stages:

1. **Prompt Input & Refinement**
   - User provides an initial text prompt
   - **Qwen 2.5:7b** (via Ollama) refines the prompt and generates negative prompts
   - Context-aware prompt formatting for optimal 3D generation

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
   - **TRELLIS** processes the multiview images
   - Generates 3D Gaussian Splatting representation
   - Converts to mesh format (GLB, PLY)
   - Creates preview videos and frame captures

5. **Feedback Collection**
   - User feedback is stored in MySQL database
   - Multiview images are saved for future reference
   - Automatic cleanup maintains storage limits (500 images max)

##  Requirements

### Hardware
- **GPU**: NVIDIA A100 (or compatible CUDA-capable GPU with 40GB+ VRAM)
- **CPU**: AMD EPYC or equivalent (16+ cores recommended)
- **RAM**: 64GB+ recommended
- **Storage**: 100GB+ free space

### Software
- Python 3.10
- CUDA 12.1
- MySQL Server
- Conda (for environment management)

##  Installation

### Automated Setup (Vast.ai A100)

For Vast.ai instances, use the automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
- Create a Conda environment with Python 3.10
- Install CUDA Toolkit 12.1 and build tools
- Install PyTorch 2.5.1 with CUDA 12.1 support
- Clone MV-Adapter and TRELLIS repositories
- Install all Python dependencies
- Set up Ollama and download required models
- Configure MySQL database

### Manual Setup

1. **Create Conda Environment**
   ```bash
   conda create -n beau python=3.10
   conda activate beau
   ```

2. **Install CUDA Toolkit**
   ```bash
   conda install -y -c conda-forge "cuda-toolkit=12.1.*" "cuda-nvcc=12.1.*" gcc=11 gxx=11 cmake ninja
   ```

3. **Install PyTorch**
   ```bash
   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
     --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Clone Dependencies**
   ```bash
   cd /path/to/direcotry  # or your parent directory
   git clone https://github.com/huanngzh/MV-Adapter
   git clone https://github.com/microsoft/TRELLIS
   cd TRELLIS
   git submodule init
   git submodule update
   chmod +x setup.sh
   ./setup.sh --basic --xformers --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
   ```

5. **Install Python Dependencies**
   ```bash
   cd /path/to/repo
   pip install -r requirements.txt
   ```

6. **Set Up Ollama**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull qwen2.5vl:7b
   ollama pull qwen2.5:7b
   ```

7. **Configure MySQL**
   ```bash
   # Start MySQL service
   service mysql start
   
   # Create database and user
   mysql -e "CREATE DATABASE IF NOT EXISTS text_to_model_db;"
   mysql -e "CREATE USER IF NOT EXISTS 'admin'@'localhost' IDENTIFIED BY 'admin';"
   mysql -e "GRANT ALL PRIVILEGES ON text_to_model_db.* TO 'admin'@'localhost';"
   mysql -e "FLUSH PRIVILEGES;"
   
   # Apply schema
   mysql -uadmin -padmin text_to_model_db < sql/init_db/db_schema.sql
   ```

8. **Set Environment Variables**
   ```bash
   export ATTN_BACKEND=xformers
   export OLLAMA_URL=http://127.0.0.1:11200
   export TEXT_MODEL=qwen2.5:7b
   export VISION_MODEL=qwen2.5vl:7b
   ```

## 🏃 Running the Pipeline

### Quick Start

Use the provided script to start all services:

```bash
./run_pipeline.sh
```

This starts:
- Ollama server (on port 11200)
- FastAPI server (default port 8000)
- Demo chat interface (if available)

### Options

```bash
# Skip Ollama (if already running)
./run_pipeline.sh --ignoreOllama

# Skip API (if already running)
./run_pipeline.sh --ignoreApi

# Skip chat interface
./run_pipeline.sh --ignoreChat
```

### Manual Start

```bash
# Start Ollama
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_NOHISTORY=true OLLAMA_NUM_PARALLEL=1 \
  OLLAMA_GPU_OVERHEAD=10240 ollama serve &

# Start API
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start chat interface (if available)
python demo_chat.py
```

## 📡 API Endpoints

### POST `/t2mv/generate`

Generate multiview images from a text prompt.

**Parameters:**
- `ref_prompt` (string): The text prompt describing the desired object/scene
- `neg_prompt` (string): Negative prompt to avoid unwanted features
- `randomize` (bool, default: false): Use random seed if true, fixed seed if false
- `inference_steps` (int, default: 45): Number of diffusion steps

**Response:**
- Returns a ZIP file containing 6 multiview images (view_00.jpg through view_05.jpg)

**Example:**
```bash
curl -X POST "http://localhost:8000/t2mv/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_prompt": "a cute orange cat",
    "neg_prompt": "blurry, low detail, artifacts",
    "randomize": false,
    "inference_steps": 45
  }' \
  --output views.zip
```

### POST `/mv2m/generate`

Convert multiview images to 3D mesh.

**Parameters:**
- `files` (multipart/form-data): List of image files (6 views recommended)

**Response:**
```json
{
  "glb_b64": "base64_encoded_glb_file",
  "ply_b64": "base64_encoded_ply_file",
  "mp4_b64": "base64_encoded_preview_video",
  "filenames": {
    "glb": "mesh_<session_id>.glb",
    "ply": "mesh_<session_id>.ply",
    "mp4": "preview_<session_id>.mp4",
    "frame": "frame_<session_id>.jpg"
  },
  "sizes": {
    "glb": 1234567,
    "ply": 2345678,
    "mp4": 3456789,
    "frame": 456789
  },
  "frame_path": "/path/to/frame.jpg"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/mv2m/generate" \
  -F "files=@view_00.jpg" \
  -F "files=@view_01.jpg" \
  -F "files=@view_02.jpg" \
  -F "files=@view_03.jpg" \
  -F "files=@view_04.jpg" \
  -F "files=@view_05.jpg"
```

### POST `/feedback/save`

Save user feedback and associated multiview images.

**Parameters:**
- `is_positive` (int): 1 for positive feedback, 0 for negative
- `original_prompt` (string): Original user prompt
- `refined` (string, optional): Refined prompt used for generation
- `chat` (string, optional): Chat conversation history
- `negative_prompt` (string, optional): Negative prompt used
- `video_frame_path` (string, optional): Path to preview frame
- `multiview_files` (files, optional): Multiview images to save
- `step` (string, optional): Step in workflow (IMAGE, VIDEO, REGENERATE)

**Response:**
```json
{
  "status": "success",
  "feedback_id": 123
}
```

##  Project Structure

```
beau-diffusion/
├── api.py                 # FastAPI application and endpoints
├── mvadapter_t2mv.py      # MV-Adapter pipeline wrapper
├── trellis_mv2m.py        # TRELLIS pipeline wrapper
├── ollama.py              # Ollama integration for LLM operations
├── run_pipeline.sh        # Script to start all services
├── setup.sh               # Automated setup script
├── requirements.txt       # Python dependencies
├── sql/
│   ├── db_connector.py    # MySQL database connector
│   ├── docker-compose.yml # Docker setup for MySQL (optional)
│   └── init_db/
│       └── db_schema.sql  # Database schema
└── README.md              # This file
```

##  Configuration

### Database Configuration

Edit `sql/db_connector.py` to change database settings:

```python
DB_CONFIG = {
    'host': 'localhost',
    'database': 'text_to_model_db',
    'user': 'admin',
    'password': 'admin'
}
```

### Ollama Configuration

Set environment variables:

```bash
export OLLAMA_URL=http://127.0.0.1:11200
export TEXT_MODEL=qwen2.5:7b
export VISION_MODEL=qwen2.5vl:7b
```

### Model Paths

Update paths in `mvadapter_t2mv.py` and `trellis_mv2m.py`:

```python
MVADAPTER_PATH = "/workspace/MV-Adapter"
TRELLIS_PATH = "/workspace/TRELLIS"
```

## Database Schema

### `feedback` Table
- `id`: Primary key
- `step`: Workflow step (IMAGE, VIDEO, REGENERATE)
- `created_at`: Timestamp
- `is_positive`: Boolean feedback flag
- `original_prompt`: User's original prompt
- `refined_prompt`: LLM-refined prompt
- `chat`: Conversation history
- `negative_prompt`: Negative prompt used
- `video_frame_url`: Path to preview frame

### `multiviews` Table
- `id`: Primary key
- `feedback_id`: Foreign key to feedback table
- `image_url`: Path to multiview image file

## 🧪 Testing

Test individual components:

```bash
# Test Ollama prompt refinement
python -c "from ollama import prepare_prompts; print(prepare_prompts('a cute cat'))"

# Test API endpoints
curl http://localhost:8000/docs  # OpenAPI documentation
```

## Logging

Logs are written to:
- `api.log`: FastAPI application logs
- `ollama.log`: Ollama integration logs
- `gradio.log`: Gradio interface logs (if used)

Log rotation is configured (1 MB for API, 10 MB for Ollama).

##  Troubleshooting

### CUDA Out of Memory
- Reduce batch size or image resolution
- Enable VAE slicing (already enabled by default)
- Use fewer inference steps

### Ollama Connection Errors
- Ensure Ollama is running: `curl http://127.0.0.1:11200/api/tags`
- Check OLLAMA_URL environment variable
- Verify models are downloaded: `ollama list`

### Database Connection Issues
- Verify MySQL is running: `service mysql status`
- Check credentials in `sql/db_connector.py`
- Ensure database exists: `mysql -uadmin -padmin -e "SHOW DATABASES;"`

### Model Loading Errors
- Verify MV-Adapter and TRELLIS are cloned correctly
- Check paths in Python files match your directory structure
- Ensure all submodules are initialized (TRELLIS)

##  Dependencies

Key dependencies:
- **PyTorch 2.5.1** (CUDA 12.1)
- **Diffusers**: Hugging Face diffusion models
- **xformers 0.0.29**: Optimized attention mechanisms
- **FastAPI**: Web framework
- **TRELLIS**: Microsoft's 3D reconstruction pipeline
- **MV-Adapter**: Multiview generation adapter
- **Ollama**: Local LLM inference

See `requirements.txt` for complete list.

## Used base repos

- **MV-Adapter**: [huanngzh/MV-Adapter](https://github.com/huanngzh/MV-Adapter)
- **TRELLIS**: [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)
- **Ollama**: [ollama/ollama](https://github.com/ollama/ollama)
- **RealVisXL**: SG161222/RealVisXL_V4.0

## 📧 Contact

### Updating the Application
To update the Gradio interface and project code to the latest version, verify the steps in the `MAINTENANCE_GUIDE.md` file. Generally, this involves pulling the latest changes and restarting the service.

### Instance Updates
Regular updates to the compute instance (drivers and OS) are recommended to ensure compatibility with the NVIDIA A100 GPU. Refer to the operations guide for detailed instructions.

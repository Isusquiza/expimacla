# Before running:
#   pip install timm open_clip_torch transformers datasets tqdm pandas pillow
#
# User Guide :
#   1. Edit DATASET_NAME y MODEL_NAME en Section 1.
#   2. Execute: python extract_embeddings.py
#   3. To execute in batch manner: active MODEL_BATCH ate the end of script.
# ============================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1 — MAIN CONFIGURATION  ← EDIT HERE
# ============================================================================

# Dataset to process
# Options: 'UCM' | 'DTD' | 'CALTECH101' | 'CALTECH256' | 'SUN397' | 'CIFAR100' | 'FOOD101'
DATASET_NAME = 'DTD'

# Select the model to extract embeddings with. Must be one of the keys in FeatureExtractorFactory.MODEL_REGISTRY.
MODEL_NAME = 'dino_v2_large'

# Bases paths for saving embeddings and temporary data. Adjust if needed.
BASE_DIR = 'your_base_path_here'  
DATA_DIR = './data'

# Parameters for embedding extraction
BATCH_SIZE           = 32     
SEED                 = 42     # IMPORTANT: Don´t Change — must match training script seeds
TRAIN_RATIO          = 0.70
VAL_RATIO            = 0.15
TEST_RATIO           = 0.15
NUM_TIMING_BATCHES   = 50     
ONLY_MEASURE_TIME    = False  # If True, runs a single batch through the model to measure inference time without saving embeddings (useful for benchmarking).
NORMALIZE_EMBEDDINGS = False  # If True, applies L2 normalization to the extracted embeddings (recommended for cosine similarity in downstream tasks).
USE_OFFICIAL_TRAIN_ONLY = False   #If True = use official train split


# ============================================================================
# SECTION 2 — DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    'UCM': {
        'save_dir':    os.path.join(BASE_DIR, 'UCM'),
        'prefix':      'ucm',
        'num_classes': 21,
        'description': 'UC Merced Land Use (21 clases, 2,100 imgs)',
        'source':      'huggingface', 'hf_repo': 'blanchon/UC_Merced',
    },
    'DTD': {
        'save_dir':    os.path.join(BASE_DIR, 'DTD'),
        'prefix':      'dtd',
        'num_classes': 47,
        'description': 'Describable Textures (47 clases, 5,640 imgs)',
        'source':      'torchvision_splits',
    },
    'CALTECH256': {
        'save_dir':    os.path.join(BASE_DIR, 'CALTECH256'),
        'prefix':      'caltech256',
        'num_classes': 257,
        'description': 'Caltech-256 (257 clases, ~30,607 imgs)',
        'source':      'torchvision',
    },
    'SUN397': {
        'save_dir':    os.path.join(BASE_DIR, 'SUN397'),
        'prefix':      'sun397',
        'num_classes': 397,
        'description': 'SUN-397 Scene Recognition (397 clases, ~108,754 imgs)',
        'source':      'huggingface', 'hf_repo': '1aurent/SUN397',
    },
    'CIFAR100': {
        'save_dir':    os.path.join(BASE_DIR, 'CIFAR100'),
        'prefix':      'cifar100',
        'num_classes': 100,
        'description': 'CIFAR-100 (100 clases, 60,000 imgs, 32×32→224)',
        'source':      'torchvision_splits',
    },
    'FOOD101': {
        'save_dir':    os.path.join(BASE_DIR, 'FOOD101'),
        'prefix':      'food101',
        'num_classes': 101,
        'description': 'Food-101 (101 clases, 101,000 imgs)',
        'source':      'torchvision_splits',
    },
}

config         = DATASET_CONFIG[DATASET_NAME]
SAVE_DIR       = config['save_dir']
PREFIX         = config['prefix']
MODEL_SAVE_DIR = os.path.join(SAVE_DIR, MODEL_NAME)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================================
# SECTION 3 — DISPOSITIVE CONFIGURATION
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print(f"  EXTRACTION OF EMBEDDINGS — {MODEL_NAME.upper()}  —  {DATASET_NAME}")
print("=" * 80)

if torch.cuda.is_available():
    gpu_name      = torch.cuda.get_device_name(0)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_mem_free  = gpu_mem_total - torch.cuda.memory_allocated() / 1e9
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print(f"\n  GPU:        {gpu_name}")
    print(f"  VRAM:       {gpu_mem_free:.1f} / {gpu_mem_total:.1f} GB disponible")
else:
    gpu_name = 'CPU'
    print(f"\n  Without GPU — the extraction will be significantly slower.")

print(f"  Dataset:    {DATASET_NAME} — {config['description']}")
print(f"  Saving:     {MODEL_SAVE_DIR}")


# ============================================================================
# SECTION 4 — FEATURE EXTRACTOR FACTORY
# ============================================================================

class FeatureExtractorFactory:
    """
    Factory centralized to 35+ extractores de características.

    Diseño:
    ─────────────────────────────────────────────────────────────────────────
    • MODEL_REGISTRY  — diccionario único de metadatos (dim, dependencia).
      Añadir un modelo nuevo = una entrada en el registry + un método _build_*.

    • create()        — único punto de entrada. Llama al builder correspondiente,
      aplica eval() + requires_grad=False automáticamente, y verifica la
      dimensión con una pasada de prueba.

    • _freeze()       — helper interno, evita repetir 4 líneas en cada builder.

    • list_models()   — imprime tabla de modelos disponibles.

    Todos los builders retornan el modelo listo para inferencia en `device`.
    """

    # ── Registro central ─────────────────────────────────────────────────────
    # Clave  → (dimensión_embedding, dependencia_pip)
    MODEL_REGISTRY: dict = {
        # Vision Transformers
        'vit_b_16':           (768,  'torchvision'),
        'vit_l_16':           (1024, 'torchvision'),
        'vit_h_14':           (1280, 'pip install timm'),
        # DINOv2
        'dino_v2_small':      (384,  'pip install timm'),
        'dino_v2_base':       (768,  'pip install timm'),
        'dino_v2_large':      (1024, 'pip install timm'),
        'dino_v2_giant':      (1536, 'pip install timm'),
        # DINO v1
        'dino_v1_vits8':      (384,  'torch.hub (auto)'),
        'dino_v1_vitb8':      (768,  'torch.hub (auto)'),
        'dino_v1_vitb16':     (768,  'torch.hub (auto)'),
        # CLIP
        'clip_vit_b32':       (512,  'pip install open_clip_torch'),
        'clip_vit_l14':       (768,  'pip install open_clip_torch'),
        'clip_jina_v2':       (1024, 'pip install sentence-transformers'), 
        'jina_emb_v4':        (2048, 'pip install transformers>=4.52.0 peft>=0.15.2'),
    }

    @classmethod
    def list_models(cls) -> None:
        """Print all models available with dimension and dependency."""
        print(f"\n  {'Model':<22} {'Dim':>6}  Dependency")
        print(f"  {'-'*60}")
        current_family = ''
        family_map = {
            'vit':     'Vision Transformers',
            'dino_v2': 'DINOv2  (Meta 2023)',
            'dino_v1': 'DINO v1 (Meta 2021)',
            'clip':    'CLIP / OpenCLIP',
            'siglip':  'SigLIP  (Google 2023)',
            'mae':     'MAE     (Meta 2021)',
            'mocov3':  'MoCo v3 (Meta 2021)',
            'resnet':  'ConvNets',
            'effici':  'ConvNets',
            'convn':   'ConvNets',
            'swin':    'Swin Transformer',
            'beit':    'BEiT    (Microsoft 2021)',
            'eva02':   'EVA-02  (BAAI 2023)',
            'intern':  'InternViT (Shanghai AI Lab 2024)',
            'sam':     'SAM Vision Encoder (Meta 2023)',
        }
        for name, (dim, dep) in cls.MODEL_REGISTRY.items():
            prefix = next((v for k, v in family_map.items() if name.startswith(k)), '')
            if prefix != current_family:
                current_family = prefix
                print(f"\n  # {prefix}")
            print(f"  {name:<22} {dim:>6}  {dep}")
        print()

    @classmethod
    def get_embedding_dim(cls, name: str) -> int:
        name = name.lower()
        if name not in cls.MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' not registered. Use list_models() to see options.")
        return cls.MODEL_REGISTRY[name][0]

    @classmethod
    def create(cls, name: str, device: torch.device) -> tuple:
        """
        Create the feature extractor, freeze it, verify the dimension, and return it.

        Returns:
            (model, actual_embedding_dim)
        """
        name = name.lower()
        if name not in cls.MODEL_REGISTRY:
            cls.list_models()
            raise ValueError(f"Model '{name}' not supported. See list above.")

        builder = getattr(cls, f'_build_{name}', None)
        if builder is None:
            raise NotImplementedError(
                f"Builder '_build_{name}' not implemented.\n"
                f"Add a static method with that name to FeatureExtractorFactory."
            )

        print(f"\n  Loading: {name} ...")
        model = builder(device)
        cls._freeze(model)

        # IMPORTANT: Use correct image size for each model family
        if 'intern_vit' in name:
            img_size = 448
        elif 'dino_v2' in name:
            img_size = 518   # patch14: 37 × 14 = 518
        else:
            img_size = 224
        with torch.no_grad():
            dummy = torch.randn(2, 3, img_size, img_size).to(device)
            out   = model(dummy)
        actual_dim = out.shape[-1]

        n_params = sum(p.numel() for p in model.parameters())
        print(f" {name}  |  dim={actual_dim}  |  params={n_params:,}  |  device={device}")
        return model, actual_dim


    @staticmethod
    def _freeze(model: nn.Module) -> None:
        """eval() + deactivate gradients. Load once of create()."""
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    # =========================================================================
    # BUILDERS — Vision Transformers
    # =========================================================================

    @staticmethod
    def _build_vit_b_16(device):
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()
        return m.to(device)

    @staticmethod
    def _build_vit_l_16(device):
        from torchvision.models import vit_l_16, ViT_L_16_Weights
        m = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()
        return m.to(device)

    @staticmethod
    def _build_vit_h_14(device):
        import timm
        return timm.create_model('vit_huge_patch14_224', pretrained=True, num_classes=0).to(device)

    # =========================================================================
    # BUILDERS — DINOv2 
    # =========================================================================

    @staticmethod
    def _build_dino_v2_small(device):
        import timm
        return timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0).to(device)

    @staticmethod
    def _build_dino_v2_base(device):
        import timm
        return timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0).to(device)

    @staticmethod
    def _build_dino_v2_large(device):
        import timm
        return timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, num_classes=0).to(device)

    @staticmethod
    def _build_dino_v2_giant(device):
        import timm
        return timm.create_model('vit_giant_patch14_dinov2.lvd142m', pretrained=True, num_classes=0).to(device)

    # =========================================================================
    # BUILDERS — DINO v1  (via torch.hub)
    # =========================================================================

    @staticmethod
    def _build_dino_v1_vits8(device):
        return torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=True).to(device)

    @staticmethod
    def _build_dino_v1_vitb8(device):
        return torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=True).to(device)

    @staticmethod
    def _build_dino_v1_vitb16(device):
        return torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True).to(device)

    # =========================================================================
    # BUILDERS — CLIP  (OpenAI / OpenCLIP)
    # Note: we extract only the visual encoder and wrap it in a lightweight wrapper
    # =========================================================================

    @staticmethod
    def _build_clip_vit_b32(device):
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        return _CLIPVisualWrapper(m).to(device)

    @staticmethod
    def _build_clip_vit_l14(device):
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        return _CLIPVisualWrapper(m).to(device)

    @staticmethod
    def _build_clip_jina_v1(device):
        raise RuntimeError(
            "clip_jina_v1 is not available: the model jinaai/jina-clip-v1 has "
            "a bug in its EVA (eva_model.py) code that is incompatible "
            "with PyTorch 2.x + accelerate in Colab/GPU. "
            "Use 'clip_jina_v2' or 'jina_emb_v4' instead."
        )

    @staticmethod
    def _build_clip_jina_v2(device):
        from sentence_transformers import SentenceTransformer
        from PIL import Image as _PIL

        st_model = SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True)
        st_model = st_model.to(device)

        class JinaV2Wrapper(nn.Module):
            def __init__(self, model, dev):
                super().__init__()
                self.st     = model
                self.device = dev

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x * 0.5 + 0.5
                x = (x * 255).clamp(0, 255).byte().cpu()
                pil_images = [
                    _PIL.fromarray(x[i].permute(1, 2, 0).numpy(), mode='RGB')
                    for i in range(x.shape[0])
                ]
                embeddings = self.st.encode(
                    pil_images,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                return embeddings.float().to(self.device)

        wrapper = JinaV2Wrapper(st_model, device)
        wrapper.eval()
        for param in wrapper.parameters():
            param.requires_grad = False
        return wrapper

    @staticmethod
    def _build_jina_emb_v4(device):
        from transformers import AutoModel
        from PIL import Image as _PIL

        model = AutoModel.from_pretrained(
            'jinaai/jina-embeddings-v4',
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model = model.to(device)
        model.eval()

        class JinaEmbV4Wrapper(nn.Module):
            def __init__(self, m, dev):
                super().__init__()
                self.model  = m
                self.device = dev

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x * 0.5 + 0.5
                x = (x * 255).clamp(0, 255).byte().cpu()
                pil_images = [
                    _PIL.fromarray(x[i].permute(1, 2, 0).numpy(), mode='RGB')
                    for i in range(x.shape[0])
                ]
                with torch.no_grad():
                    embeddings = self.model.encode_image(
                        images=pil_images,
                        task='retrieval',
                    )
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.from_numpy(embeddings)
                return embeddings.float().to(self.device)

        wrapper = JinaEmbV4Wrapper(model, device)
        wrapper.eval()
        for param in wrapper.parameters():
            param.requires_grad = False
        return wrapper
    
# ============================================================================
# SECTION 5 — WRAPPERS TO MODELS WITH NON-STANDARD APIS
# ============================================================================

class _CLIPVisualWrapper(nn.Module):
    """
    Extrae el encoder visual de un modelo CLIP o SigLIP y normaliza la salida.
    Maneja los distintos formatos de retorno (tensor 2D, 3D, o tupla).
    """
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.visual(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 3:       # [B, T, D] → tomar token CLS
            out = out[:, 0]
        return out


class _JinaCLIPWrapper(nn.Module):
    """
    OBSOLETO — mantenido para compatibilidad pero NO usado por clip_jina_v1/v2.
    Los builders de Jina definen su wrapper inline (patron que funciona en PyTorch 2.x).
    """
    # CLIP normalization stats (los mismos que usa get_transforms para 'clip_*')
    _CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
    _CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    def __init__(self, jina_model, device):
        super().__init__()
        self.model   = jina_model.to(device)   # mover aqui, NUNCA fuera del __init__
        self._device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deshacer CLIP normalization -> [0, 1]
        mean = self._CLIP_MEAN.to(x.device).view(1, 3, 1, 1)
        std  = self._CLIP_STD.to(x.device).view(1, 3, 1, 1)
        x_01 = (x * std + mean).clamp(0, 1)

        # Convertir a PIL
        x_u8 = (x_01 * 255).byte().cpu()
        pil_imgs = [
            Image.fromarray(x_u8[i].permute(1, 2, 0).numpy(), mode='RGB')
            for i in range(x_u8.shape[0])
        ]

        with torch.no_grad():
            emb = self.model.encode_image(pil_imgs)

        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        return emb.float().to(self._device)


class _InternViTWrapper(nn.Module):
    """
    InternViT retorna last_hidden_state [B, T, D].
    Aplica average pooling sobre tokens de patch (excluye CLS en pos 0).
    """
    def __init__(self, intern_model):
        super().__init__()
        self.model = intern_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x).last_hidden_state  # [B, T, D]
        return out[:, 1:].mean(dim=1).float()                # avg pool patches


# ============================================================================
# SECTION 6 — TRANSFORMATIONS FOR MODEL FAMILY 
# ============================================================================

def get_transforms(model_name: str) -> transforms.Compose:
    """
    Retorn correct transformations according the family model..
    The most use ImageNet mean/std + 224x224.
    Exceptions (verified with official code/documentation):
      DINOv2 (dino_v2_*)  -> 518x518  (patch14: 37 patches x 14 px)
      InternViT            -> 448x448
      CLIP/SigLIP          -> mean/std propios de OpenAI
      Jina CLIP            -> mean=0.5/std=0.5 
    """
    MEAN_IMAGENET = [0.485, 0.456, 0.406]
    STD_IMAGENET  = [0.229, 0.224, 0.225]
    MEAN_CLIP     = [0.48145466, 0.4578275,  0.40821073]
    STD_CLIP      = [0.26862954, 0.26130258, 0.27577711]

    name = model_name.lower()

    if 'intern_vit' in name:
        size, mean, std = 448, MEAN_IMAGENET, STD_IMAGENET
    elif 'dino_v2' in name:
        size, mean, std = 518, MEAN_IMAGENET, STD_IMAGENET
    elif 'dino_v1' in name:
        size, mean, std = 224, MEAN_IMAGENET, STD_IMAGENET
    elif 'clip_jina' in name or 'jina_emb' in name:
        size, mean, std = 224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif any(k in name for k in ('clip', 'siglip')):
        size, mean, std = 224, MEAN_CLIP, STD_CLIP
    else:
        size, mean, std = 224, MEAN_IMAGENET, STD_IMAGENET

    class _ToRGB:
        def __call__(self, img):
            return img.convert('RGB') if img.mode != 'RGB' else img

    return transforms.Compose([
        _ToRGB(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ============================================================================
# SECTION 7 — DATASETS PYTORCH
# ============================================================================

class HuggingFaceDataset(Dataset):
 
    def __init__(self, hf_data, indices=None, transform=None):
        self.hf_data   = hf_data
        self.indices   = [int(i) for i in indices] if indices is not None else list(range(len(hf_data)))
        self.transform = transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        item  = self.hf_data[int(self.indices[idx])]
        image = item['image']
        label = item['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def _load_hf_dataset(hf_repo: str):
    """Descarga un dataset de HuggingFace con fallback automático."""
    from datasets import load_dataset
    print(f"\n  Descargando '{hf_repo}' desde HuggingFace ...")
    try:
        ds = load_dataset(hf_repo, cache_dir='./hf_cache')
    except Exception:
        ds = load_dataset(hf_repo, trust_remote_code=True, cache_dir='./hf_cache')
    split_key = 'train' if 'train' in ds else list(ds.keys())[0]
    return ds[split_key]


def _stratified_split(labels: np.ndarray, val_r: float, test_r: float, seed: int):
    """Split estratificado 70/15/15 reproducible."""
    indices  = np.arange(len(labels))
    tr_idx, tmp = train_test_split(indices, test_size=val_r + test_r,
                                   stratify=labels, random_state=seed)
    va_idx, te_idx = train_test_split(tmp, test_size=test_r / (val_r + test_r),
                                      stratify=labels[tmp], random_state=seed)
    return tr_idx, va_idx, te_idx


def load_dataset_splits(transform):
    """
    Load dataset configured (DATASET_NAME) y return:
        train_ds, val_ds, test_ds
    Respect official splits where exist; use 70/15/15 estratified in the rest.
    """
    from torchvision.datasets import DTD, Caltech101, Caltech256, CIFAR100, Food101

    cfg  = DATASET_CONFIG[DATASET_NAME]
    src  = cfg.get('source', '')

    # ── HuggingFace datasets (UCM, SUN397) ───────────────────────────────────
    if src == 'huggingface':
        hf_data    = _load_hf_dataset(cfg['hf_repo'])
        all_labels = np.array(hf_data['label'])
        tr_idx, va_idx, te_idx = _stratified_split(all_labels, VAL_RATIO, TEST_RATIO, SEED)

        np.savez(os.path.join(SAVE_DIR, f'{PREFIX}_split_indices.npz'),
                 train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx, seed=SEED)

        n_cls = (hf_data.features['label'].num_classes
                 if hasattr(hf_data.features['label'], 'num_classes')
                 else len(set(all_labels)))
        _print_split_summary(len(tr_idx), len(va_idx), len(te_idx), len(hf_data), n_cls, 'HuggingFace')

        return (HuggingFaceDataset(hf_data, tr_idx, transform),
                HuggingFaceDataset(hf_data, va_idx, transform),
                HuggingFaceDataset(hf_data, te_idx, transform))

    # ── Datasets splits OFICIALS (DTD, CIFAR100, FOOD101) ───────────────
    if DATASET_NAME == 'DTD':
        tr = DTD(root=DATA_DIR, split='train', download=True, transform=transform)
        va = DTD(root=DATA_DIR, split='val',   download=True, transform=transform)
        te = DTD(root=DATA_DIR, split='test',  download=True, transform=transform)
        _print_split_summary(len(tr), len(va), len(te), len(tr)+len(va)+len(te), 47, 'oficiales')
        return tr, va, te

    if DATASET_NAME == 'CIFAR100':
        from torchvision.datasets import CIFAR100 as C100
        tr_full = C100(root=DATA_DIR, train=True,  download=True, transform=transform)
        te      = C100(root=DATA_DIR, train=False, download=True, transform=transform)
        tr_labels = np.array(tr_full.targets)
        tr_idx, va_idx = train_test_split(np.arange(len(tr_full)), test_size=0.1,
                                          stratify=tr_labels, random_state=SEED)
        _print_split_summary(len(tr_idx), len(va_idx), len(te),
                             len(tr_full)+len(te), 100, 'oficiales (val=10% de train)')
        return Subset(tr_full, tr_idx), Subset(tr_full, va_idx), te

    if DATASET_NAME == 'FOOD101':
        tr_full = Food101(root=DATA_DIR, split='train', download=True, transform=transform)
        te      = Food101(root=DATA_DIR, split='test',  download=True, transform=transform)

        n_te_official = len(te)

        if hasattr(tr_full, '_labels'):
            tr_labels = np.array(tr_full._labels)
        else:
            tr_labels = np.array([tr_full[i][1]
                                   for i in tqdm(range(len(tr_full)), desc='  Labels Food101')])

        indices_path = os.path.join(SAVE_DIR, f'{PREFIX}_split_indices.npz')

        if USE_OFFICIAL_TRAIN_ONLY:
     
            tr_idx = np.arange(len(tr_full))
            va_idx = np.array([], dtype=np.int64)

            np.savez(indices_path,
                     train_idx        = tr_idx,
                     val_idx          = va_idx,
                     test_is_official = np.array([True]),
                     n_test_official  = np.array([n_te_official]),
                     seed             = np.array([SEED]),
                     mode             = np.array(['official_train_only']))
            print(f'  Split indices guardados: {indices_path}')
            print(f'  Modo: OFFICIAL_TRAIN_ONLY — train={len(tr_idx):,}  val=0  test={n_te_official:,} (oficial)')

            _print_split_summary(len(tr_idx), 0, n_te_official,
                                 len(tr_full) + n_te_official, 101,
                                 'oficiales — train completo (sin val)')
            return tr_full, None, te   # val=None indica "sin val"

        else:

            if os.path.exists(indices_path):
                saved = np.load(indices_path, allow_pickle=True)
                if ('mode' in saved and str(saved['mode']) == "['val_from_train']") or \
                   ('train_idx' in saved and 'val_idx' in saved and len(saved['val_idx']) > 0):
                    tr_idx = saved['train_idx']
                    va_idx = saved['val_idx']
                    print(f'  Split indices load from: {indices_path}')
                    print(f'     train={len(tr_idx):,}  val={len(va_idx):,}  '
                          f'test={n_te_official:,} (oficial)')
                else:
                    print(f'  Split indices exist in different mode → recreating ')
                    tr_idx, va_idx = None, None
            else:
                tr_idx, va_idx = None, None

            if tr_idx is None:
                tr_idx, va_idx = train_test_split(
                    np.arange(len(tr_full)), test_size=0.10,
                    stratify=tr_labels, random_state=SEED)
                np.savez(indices_path,
                         train_idx        = tr_idx,
                         val_idx          = va_idx,
                         test_is_official = np.array([True]),
                         n_test_official  = np.array([n_te_official]),
                         seed             = np.array([SEED]),
                         mode             = np.array(['val_from_train']))
                print(f'  Split indices guardados: {indices_path}')

            print(f'  Test oficial FIJO: {n_te_official:,} imágenes (Food101 split=test)')
            print(f'  Val reproducible : {len(va_idx):,} imágenes '
                  f'(10% del train oficial, SEED={SEED})')
            _print_split_summary(len(tr_idx), len(va_idx), n_te_official,
                                 len(tr_full) + n_te_official, 101,
                                 f'oficiales — val=10% del train (SEED={SEED})')
            return Subset(tr_full, tr_idx), Subset(tr_full, va_idx), te

    if DATASET_NAME == 'CALTECH101':
        full = Caltech101(root=DATA_DIR, target_type='category', download=True, transform=transform)
        all_labels = np.array([full[i][1] for i in tqdm(range(len(full)), desc='  Labels')])
    else:  # CALTECH256
        full = Caltech256(root=DATA_DIR, download=True, transform=transform)
        all_labels = np.array([full[i][1] for i in tqdm(range(len(full)), desc='  Labels')])

    tr_idx, va_idx, te_idx = _stratified_split(all_labels, VAL_RATIO, TEST_RATIO, SEED)
    np.savez(os.path.join(SAVE_DIR, f'{PREFIX}_split_indices.npz'),
             train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx, seed=SEED)
    _print_split_summary(len(tr_idx), len(va_idx), len(te_idx), len(full),
                         len(np.unique(all_labels)), '70/15/15 estratificado')
    return Subset(full, tr_idx), Subset(full, va_idx), Subset(full, te_idx)


def _print_split_summary(n_tr, n_va, n_te, n_total, n_cls, split_type):
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  {DATASET_NAME:<63}│
  ├─────────────────────────────────────────────────────────────────┤
  │  Train  : {n_tr:>8,}   Val : {n_va:>7,}   Test : {n_te:>7,}              │
  │  Total  : {n_total:>8,}   Clases : {n_cls:>3}                             │
  │  Splits : {split_type:<53}│
  └─────────────────────────────────────────────────────────────────┘""")


# ============================================================================
# SECTION 8 — EXTRACT EMBEDDINGS WITH TIMING
# ============================================================================

def extract_embeddings(dataset, model, desc="Extracting embeddings", normalize=NORMALIZE_EMBEDDINGS):
    """
    Extracts embeddings from a complete dataset and measures times per batch.
    Omits the first batch as GPU/cache warming.

    Returns:
        embeddings : np.ndarray  [N, D]  float32
        labels     : np.ndarray  [N]     int64
        timing     : dict with detailed latency metrics
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=torch.cuda.is_available())

    all_emb, all_lbl          = [], []
    t_total, t_infer, t_xfer = [], [], []
    n_images, warmup_done     = 0, False

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {desc}"):

            _sync()
            t0 = time.perf_counter()

            images_gpu = images.to(device)
            _sync(); t1 = time.perf_counter()

            out = model(images_gpu)
            _sync(); t2 = time.perf_counter()

            if normalize:
                out = F.normalize(out.float(), p=2, dim=1)

            all_emb.append(out.cpu().float().numpy())
            lbl = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            all_lbl.append(lbl)
            _sync(); t3 = time.perf_counter()

            if warmup_done:
                bs = images.size(0)
                t_total.append(t3 - t0);  t_infer.append(t2 - t1);  t_xfer.append(t1 - t0)
                n_images += bs
            else:
                warmup_done = True

    emb    = np.concatenate(all_emb).astype(np.float32)
    labels = np.concatenate(all_lbl).astype(np.int64)

    t_tot = np.array(t_total); t_inf = np.array(t_infer); t_xf = np.array(t_xfer)

    timing = {
        'dataset': DATASET_NAME, 'model': MODEL_NAME, 'device': str(device),
        'batch_size': BATCH_SIZE, 'n_images': n_images, 'embedding_dim': emb.shape[1],
        'per_image_total_ms':         _ms(t_tot.sum(), n_images),
        'per_image_inference_ms':     _ms(t_inf.sum(), n_images),
        'per_image_transfer_ms':      _ms(t_xf.sum(),  n_images),
        'inference_mean_ms':          t_inf.mean() * 1000,
        'inference_std_ms':           t_inf.std()  * 1000,
        'inference_p50_ms':           np.percentile(t_inf, 50) * 1000,
        'inference_p95_ms':           np.percentile(t_inf, 95) * 1000,
        'inference_p99_ms':           np.percentile(t_inf, 99) * 1000,
        'throughput_total_imgs_s':    n_images / t_tot.sum() if t_tot.sum() > 0 else 0,
        'throughput_inference_imgs_s':n_images / t_inf.sum() if t_inf.sum() > 0 else 0,
        'pct_inference':              (t_inf.sum() / t_tot.sum()) * 100 if t_tot.sum() > 0 else 0,
        'pct_transfer':               (t_xf.sum()  / t_tot.sum()) * 100 if t_tot.sum() > 0 else 0,
    }
    return emb, labels, timing


def measure_time_only(dataset, desc="Benchmark"):
    """
    Measure inference times without saving embeddings.
    Uses a dummy model (only timing), requires the global model to be loaded.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=torch.cuda.is_available())

    t_inf, n_images, warmed = [], 0, False
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"  {desc}",
                               total=min(NUM_TIMING_BATCHES + 5, len(loader))):
            images = images.to(device)
            _sync(); t0 = time.perf_counter()
            _         = model(images)
            _sync(); t1 = time.perf_counter()
            if warmed:
                t_inf.append(t1 - t0); n_images += images.size(0)
                if len(t_inf) >= NUM_TIMING_BATCHES: break
            else:
                warmed = True

    t_inf = np.array(t_inf)
    return {
        'dataset': DATASET_NAME, 'model': MODEL_NAME, 'device': str(device),
        'batch_size': BATCH_SIZE, 'n_images': n_images,
        'per_image_inference_ms':     _ms(t_inf.sum(), n_images),
        'inference_mean_ms':          t_inf.mean() * 1000,
        'inference_p95_ms':           np.percentile(t_inf, 95) * 1000,
        'throughput_inference_imgs_s':n_images / t_inf.sum() if t_inf.sum() > 0 else 0,
    }


def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def _ms(total_s, n): return (total_s / n) * 1000 if n > 0 else 0.0


def print_timing(timing: dict, title="TIMING OF INFERENCE"):
    print(f"""
  {'='*65}
  {title}
  {'='*65}
  Model    :  {timing.get('model', MODEL_NAME)}
  Dataset   :  {timing.get('dataset', DATASET_NAME)}
  Dispositive: {timing.get('device', str(device))}
  Batch size:  {timing.get('batch_size', BATCH_SIZE)}
  Dim emb   :  {timing.get('embedding_dim', '—')}
  N images:  {timing.get('n_images', 0):,}
  {'─'*65}
  TIME FOR IMAGE EMBEDDINGS
    Total (I/O + inferencia) : {timing.get('per_image_total_ms', 0):.4f} ms
    Only inferencia          : {timing['per_image_inference_ms']:.4f} ms
    Transferency CPU→GPU    : {timing.get('per_image_transfer_ms', 0):.4f} ms
  STATISTICS (per batch)
    Mean  : {timing.get('inference_mean_ms', 0):.4f} ms
    Std   : {timing.get('inference_std_ms',  0):.4f} ms
    P50   : {timing.get('inference_p50_ms',  0):.4f} ms
    P95   : {timing.get('inference_p95_ms',  0):.4f} ms
    P99   : {timing.get('inference_p99_ms',  0):.4f} ms
  THROUGHPUT
    Total     : {timing.get('throughput_total_imgs_s',    0):.1f} img/s
    Inferencia: {timing.get('throughput_inference_imgs_s',0):.1f} img/s
  {'='*65}
""")


# ============================================================================
# SECTION 9 — VERIFICATION OF EXISTING EMBEDDINGS
# ============================================================================

def check_embeddings_exist():
    # In USE_OFFICIAL_TRAIN_ONLY mode, val is not extracted → we don't require it
    required_splits = ['train', 'test'] if USE_OFFICIAL_TRAIN_ONLY else ['train', 'val', 'test']
    files = (
        [f'{PREFIX}_{MODEL_NAME}_embeddings_{s}.npy' for s in required_splits] +
        [f'{PREFIX}_{MODEL_NAME}_labels_{s}.npy'     for s in required_splits]
    )
    missing  = [f for f in files if not os.path.exists(os.path.join(MODEL_SAVE_DIR, f))]
    existing = [f for f in files if     os.path.exists(os.path.join(MODEL_SAVE_DIR, f))]
    return len(missing) == 0, existing, missing


# ============================================================================
# SECTION 10 — MAIN PIPELINE
# ============================================================================

def main():
    global model 

    print(f"\n  Modelo  : {MODEL_NAME}  (dim={FeatureExtractorFactory.get_embedding_dim(MODEL_NAME)})")
    print(f"  Dataset : {DATASET_NAME} — {config['description']}")
    _official_split_ds = ('DTD', 'CIFAR100', 'FOOD101')
    if DATASET_NAME in _official_split_ds:
        if DATASET_NAME == 'FOOD101':
            _split_desc = ('train=75,750 oficial + test=25,250 oficial  '
                           f'| val=10% del train (SEED={SEED})'
                           if not USE_OFFICIAL_TRAIN_ONLY
                           else 'train=75,750 oficial + test=25,250 oficial  | val=no (OFFICIAL_TRAIN_ONLY)')
        else:
            _split_desc = 'splits oficiales del dataset'
    else:
        _split_desc = f'{TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%} estratificado (SEED={SEED})'
    print(f"  Seed    : {SEED}  |  Split: {_split_desc}")

    # ── Verify existing embeddings ──────────────────────────────────────
    emb_exist, exist_f, missing_f = check_embeddings_exist()

    print(f"\n  {'='*65}")
    print("  VERIFICATION OF EMBEDDINGS")
    print(f"  {'='*65}")

    if emb_exist:
        print(f"  Already exist in: {MODEL_SAVE_DIR}")
        for f in exist_f:
            mb = os.path.getsize(os.path.join(MODEL_SAVE_DIR, f)) / 1024**2
            print(f"      {f}  ({mb:.1f} MB)")
        mode = "ONLY MEASURE TIME" if ONLY_MEASURE_TIME else "RE-EXTRACT"
        print(f"\n  Mode: {mode}")
    else:
        print(f"  Missing files → EXTRACT")
        for f in missing_f:
            print(f"      {f}")

    transform = get_transforms(MODEL_NAME)

    # ── Load model  ─────────────────────────────────────────────────────────
    print(f"\n  {'='*65}")
    print(f"  LOADING MODEL: {MODEL_NAME}")
    print(f"  {'='*65}")
    t_load = time.time()
    model, actual_dim = FeatureExtractorFactory.create(MODEL_NAME, device)
    print(f"  Load: {time.time() - t_load:.2f}s")

    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"\n  {'='*65}")
    print(f"  PREPARING DATASET: {DATASET_NAME}")
    print(f"  {'='*65}")
    train_ds, val_ds, test_ds = load_dataset_splits(transform)
    val_is_empty = val_ds is None or len(val_ds) == 0
    n_val_display = 0 if val_is_empty else len(val_ds)
    print(f"\n  Train={len(train_ds):,}  Val={n_val_display:,}  Test={len(test_ds):,}")
    if val_is_empty:
        print(f"  Val empty (USE_OFFICIAL_TRAIN_ONLY=True): "
              f"train embeddings contain the complete official train ({len(train_ds):,} imgs)")
        print(f"     The downstream classifiers can use train.npy directly "
              f"as the complete training set.")

    # ── Extract or measure ───────────────────────────────────────────────────────
    print(f"\n  {'='*65}")
    all_timings = {}

    if emb_exist and ONLY_MEASURE_TIME:
        print("  MEASURING TIMES (embeddings already exist)")
        print(f"  {'='*65}")
        splits_to_run = [('train', train_ds), ('test', test_ds)]
        if not val_is_empty:
            splits_to_run.insert(1, ('val', val_ds))
        for split_name, ds in splits_to_run:
            t = measure_time_only(ds, f"{split_name.capitalize()} [{MODEL_NAME}]")
            all_timings[split_name] = t
            print(f"  {split_name}: {t['per_image_inference_ms']:.4f} ms/img  "
                  f"| {t['throughput_inference_imgs_s']:.1f} img/s")
    else:
        print("  EXTRAYENDO EMBEDDINGS")
        print(f"  {'='*65}")
        results = {}
        splits_to_run = [('train', train_ds), ('test', test_ds)]
        if not val_is_empty:
            splits_to_run.insert(1, ('val', val_ds))
        for split_name, ds in splits_to_run:
            emb, lbl, timing = extract_embeddings(
                ds, model, desc=f"{split_name.capitalize()} [{MODEL_NAME}]"
            )
            results[split_name]     = (emb, lbl)
            all_timings[split_name] = timing
            print_timing(timing, f"TIMING — {split_name.upper()}")

        # ── Guardar embeddings ────────────────────────────────────────────────
        print(f"\n  {'='*65}")
        print("  SAVING EMBEDDINGS")
        print(f"  {'='*65}")
        for split_name, (emb, lbl) in results.items():
            emb_path = os.path.join(MODEL_SAVE_DIR, f'{PREFIX}_{MODEL_NAME}_embeddings_{split_name}.npy')
            lbl_path = os.path.join(MODEL_SAVE_DIR, f'{PREFIX}_{MODEL_NAME}_labels_{split_name}.npy')
            np.save(emb_path, emb)
            np.save(lbl_path, lbl)
            print(f"  {split_name:<6}: {emb.shape}  → {emb_path}")
        if val_is_empty:
            print(f"  val: not extracted (USE_OFFICIAL_TRAIN_ONLY=True)")
            print(f"     → For downstream: use train.npy as the complete training set.")
            print(f"       If you need val, change USE_OFFICIAL_TRAIN_ONLY=False and re-extract.")
        print(f"\n  L2 Normalization: {'Yes' if NORMALIZE_EMBEDDINGS else 'No'}")

    # ── Save summary of times ────────────────────────────────────────────
    avg_inf_ms = np.mean([t.get('per_image_inference_ms', 0) for t in all_timings.values()])
    avg_thr    = np.mean([t.get('throughput_inference_imgs_s', 0) for t in all_timings.values()])
    avg_tot_ms = np.mean([t.get('per_image_total_ms', avg_inf_ms) for t in all_timings.values()])

    summary = {
        'dataset': DATASET_NAME, 'model': MODEL_NAME, 'device': str(device),
        'gpu_name': gpu_name, 'embedding_dim': actual_dim,
        'n_params': sum(p.numel() for p in model.parameters()),
        'batch_size': BATCH_SIZE, 'normalize_l2': NORMALIZE_EMBEDDINGS,
        'avg_inference_per_image_ms': avg_inf_ms,
        'avg_total_per_image_ms':     avg_tot_ms,
        'avg_throughput_imgs_per_sec':avg_thr,
        'seed': SEED,
    }

    summary_path = os.path.join(MODEL_SAVE_DIR, f'{PREFIX}_{MODEL_NAME}_timing_summary.csv')
    splits_path  = os.path.join(MODEL_SAVE_DIR, f'{PREFIX}_{MODEL_NAME}_timing_splits.csv')
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(list(all_timings.values())).to_csv(splits_path, index=False)

    print(f"""
  {'='*65}
  PROCESS COMPLETED
  {'='*65}
  Modelo    : {MODEL_NAME}
  Dataset   : {DATASET_NAME}
  Guardado  : {MODEL_SAVE_DIR}
  Dim       : {actual_dim}
  Parámetros: {summary['n_params']:,}
  Latencia  : {avg_inf_ms:.4f} ms/imagen
  Throughput: {avg_thr:.1f} img/s
  {'='*65}
""")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  GPU memory released.")


# ============================================================================
# SECTION 11 — EXTRACTION ON BATCH PROCESS  (various models in sequence)
# ============================================================================

def run_batch(model_list: list, dataset_name: str = DATASET_NAME):
    """
    Extracts embeddings with various models in sequence over the same dataset.

    Example:
        run_batch(['dino_v2_large', 'siglip_so400m', 'eva02_large', 'clip_vit_l14'])
    """
    global MODEL_NAME, MODEL_SAVE_DIR, DATASET_NAME
    DATASET_NAME = dataset_name

    results = []
    for i, m_name in enumerate(model_list, 1):
        print(f"\n{'#'*80}")
        print(f"  MODEL {i}/{len(model_list)}: {m_name}")
        print(f"{'#'*80}")

        MODEL_NAME     = m_name
        MODEL_SAVE_DIR = os.path.join(DATASET_CONFIG[DATASET_NAME]['save_dir'], m_name)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

        try:
            main()
            results.append({'model': m_name, 'status': 'OK'})
        except Exception as e:
            import traceback
            print(f"\n  Error with {m_name}:")
            traceback.print_exc()
            results.append({'model': m_name, 'status': f'ERROR: {e}'})

    print(f"\n{'='*80}")
    print("  BATCH SUMMARY")
    print(f"{'='*80}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':

    # ── Option A: Only one model (config SECTION 1) ────────────────
    # main()

    # ── Option B: Various names in secuency  ─────────────────────────────────
    
    run_batch([
         'dino_v2_large',      # SOTA embeddings          dim=1024
          'clip_vit_b32',       # CLIP classic             dim=768
          'vit_b_16',
    
    ])
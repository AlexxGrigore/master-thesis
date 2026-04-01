"""
Environment sanity check — verifies CUDA, PyTorch, ARTIST, and PAINT are
correctly installed and the GPU is accessible.
"""
import sys

print("=" * 60)
print("ENVIRONMENT SANITY CHECK")
print("=" * 60)

# --- Python ---
print(f"\nPython: {sys.version}")

# --- PyTorch + CUDA ---
print("\n[PyTorch]")
import torch
print(f"  Version:        {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version:   {torch.version.cuda}")
    print(f"  Device count:   {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    t = torch.tensor([1.0]).cuda()
    print(f"  Tensor on GPU:  {t.device}  ✓")
else:
    print("  WARNING: CUDA not available — running on CPU only!")

# --- ARTIST ---
print("\n[ARTIST]")
try:
    import artist
    print(f"  Imported OK  ✓")
    from artist.scenario.scenario import Scenario
    from artist.util import config_dictionary
    print(f"  Core imports OK  ✓")
except Exception as e:
    print(f"  ERROR: {e}")

# --- PAINT ---
print("\n[PAINT]")
try:
    import paint
    print(f"  Imported OK  ✓")
    import paint.util.paint_mappings as paint_mappings
    print(f"  paint.util.paint_mappings OK  ✓")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

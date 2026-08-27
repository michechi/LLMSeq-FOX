import sys
from pathlib import Path

# Ensure `src` resolves to repro/src (the legacy top-level src/ package would
# otherwise shadow it when pytest runs from the repository root).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathlib import Path


def prefer_fusion_checkpoint(path: str, preferred_name: str) -> str:
    if not path:
        return ""
    base = Path(path)
    preferred = base.with_name(preferred_name)
    return str(preferred) if preferred.exists() else str(base)

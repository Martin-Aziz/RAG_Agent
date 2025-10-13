"""Artifact logging utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union


class ArtifactType(str, Enum):
    TEXT = "text"
    JSON = "json"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"


@dataclass
class Artifact:
    name: str
    artifact_type: ArtifactType
    data: Union[str, bytes]
    metadata: Dict[str, str] = field(default_factory=dict)

    def persist(self, directory: Union[str, Path]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        extension = self._extension()
        target = directory / f"{self.name}{extension}"
        mode = "wb" if isinstance(self.data, bytes) else "w"
        with target.open(mode) as file:
            file.write(self.data)
        return target

    def _extension(self) -> str:
        mapping = {
            ArtifactType.TEXT: ".txt",
            ArtifactType.JSON: ".json",
            ArtifactType.IMAGE: ".img",
            ArtifactType.AUDIO: ".wav",
            ArtifactType.VIDEO: ".mp4",
            ArtifactType.BINARY: ".bin",
        }
        return mapping.get(self.artifact_type, ".dat")


class ArtifactLogger:
    """Minimal artifact logger that keeps artifacts in memory and optionally persists."""

    def __init__(self) -> None:
        self._artifacts: List[Artifact] = []

    def log(self, artifact: Artifact) -> None:
        self._artifacts.append(artifact)

    def log_text(self, name: str, text: str, metadata: Optional[Dict[str, str]] = None) -> Artifact:
        artifact = Artifact(name=name, artifact_type=ArtifactType.TEXT, data=text, metadata=metadata or {})
        self.log(artifact)
        return artifact

    def artifacts(self) -> List[Artifact]:
        return list(self._artifacts)

    def clear(self) -> None:
        self._artifacts.clear()

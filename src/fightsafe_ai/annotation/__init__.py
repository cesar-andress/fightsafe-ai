"""
Manual **ground-truth** labels for evaluation (safety events in combat-sports video).

- :mod:`fightsafe_ai.annotation.schema` — Pydantic types and event vocabulary
- :mod:`fightsafe_ai.annotation.loader` — read / write JSON
- :mod:`fightsafe_ai.annotation.validator` — file validation and optional overlap notes
- :mod:`fightsafe_ai.annotation.llm_assist` — Ollama drafts for *manual* labelling (not ground truth)

CLI: ``fightsafe annotate-template``, ``fightsafe validate-annotations``,
``fightsafe suggest-annotations``.
See ``docs/annotation.md``. TapKO (submission / vulnerability track): ``tapko_schema``, ``docs/tapko_annotation.md``.
"""

from fightsafe_ai.annotation.llm_assist import (
    run_pipeline_suggest_annotations,
)
from fightsafe_ai.annotation.loader import (
    load_annotation_file,
    new_empty_template,
    save_annotation_file,
)
from fightsafe_ai.annotation.schema import (
    ANNOTATION_FORMAT_VERSION,
    AnnotationDocument,
    EventAnnotation,
    EventType,
    parse_annotation_dict,
)
from fightsafe_ai.annotation.tapko_schema import (
    EXAMPLE_DOCUMENT_FULL,
    EXAMPLE_DOCUMENT_MINIMAL,
    TAPKO_ANNOTATION_FORMAT_VERSION,
    OcclusionLevel,
    TapkoAnnotation,
    TapkoAnnotationDocument,
    TapkoAnnotationStatus,
    TapkoEventType,
    Visibility,
    parse_tapko_dict,
    parse_tapko_json,
    tapko_json_schema,
    validate_tapko_json,
)
from fightsafe_ai.annotation.validator import (
    format_overlap_warnings,
    is_valid_annotation_file,
    validate_annotation_file,
)


__all__ = [
    "ANNOTATION_FORMAT_VERSION",
    "EXAMPLE_DOCUMENT_FULL",
    "EXAMPLE_DOCUMENT_MINIMAL",
    "TAPKO_ANNOTATION_FORMAT_VERSION",
    "AnnotationDocument",
    "EventAnnotation",
    "EventType",
    "OcclusionLevel",
    "TapkoAnnotation",
    "TapkoAnnotationDocument",
    "TapkoAnnotationStatus",
    "TapkoEventType",
    "Visibility",
    "format_overlap_warnings",
    "is_valid_annotation_file",
    "load_annotation_file",
    "new_empty_template",
    "parse_annotation_dict",
    "parse_tapko_dict",
    "parse_tapko_json",
    "run_pipeline_suggest_annotations",
    "save_annotation_file",
    "tapko_json_schema",
    "validate_annotation_file",
    "validate_tapko_json",
]

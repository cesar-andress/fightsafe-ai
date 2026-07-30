"""
Compatibility re-exports for the QA report API.

Implementation lives in :mod:`fightsafe_ai.qa.validators` (``run_quality_checks``,
:class:`QualityReport`, :func:`write_qa_report_json`, etc.).
"""

from __future__ import annotations

from fightsafe_ai.qa.validators import (
    QualityCheckResult,
    QualityReport,
    Status,
    quality_report_to_dict,
    run_quality_checks,
    write_qa_report_json,
)


__all__ = [
    "QualityCheckResult",
    "QualityReport",
    "Status",
    "quality_report_to_dict",
    "run_quality_checks",
    "write_qa_report_json",
]

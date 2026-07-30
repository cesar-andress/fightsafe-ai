"""
Human- and machine-readable reports from a FightSafe AI pipeline run directory.
"""

from fightsafe_ai.reports.html import generate_html_report
from fightsafe_ai.reports.markdown import generate_markdown_report
from fightsafe_ai.reports.outputs import write_all_default_reports
from fightsafe_ai.reports.summary import build_summary_dict, generate_summary_json
from fightsafe_ai.reports.validate import (
    missing_report_artifacts,
    report_prereq_error_message,
)


__all__ = [
    "build_summary_dict",
    "generate_html_report",
    "generate_markdown_report",
    "generate_summary_json",
    "missing_report_artifacts",
    "report_prereq_error_message",
    "write_all_default_reports",
]

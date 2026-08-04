from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class CourseRecord(BaseModel):
    student_id: Optional[str] = None
    term: Optional[str] = None
    course_code: str
    course_title: Optional[str] = None
    credits: float = 0.0
    grade: Optional[str] = None
    grading_basis: Optional[str] = None  # e.g. 'GRD', 'P/F', 'EXEMPT'
    flags: List[str] = Field(default_factory=list)  # e.g. ['transfer', 'repeat']


class GpaPolicy(BaseModel):
    grade_points: Dict[str, float]
    excluded_grading_bases: List[str] = Field(default_factory=lambda: ["P/F", "PASS/FAIL", "EXEMPT", "AUDIT"])
    excluded_grades: List[str] = Field(default_factory=lambda: ["P", "S", "U", "W", "I", "AU", "EX", "NR"])
    fail_grades_count_in_gpa: List[str] = Field(default_factory=lambda: ["F", "WF"])
    transfer_flags: List[str] = Field(default_factory=lambda: ["transfer", "ap", "exempt"])
    repeat_policy: Literal["all_attempts", "latest_attempt", "highest_grade"] = "latest_attempt"
    include_zero_credit_in_gpa: bool = False
    count_pass_fail_credits_as_earned: bool = True


class CourseAudit(BaseModel):
    index: int
    course_code: str
    term: Optional[str]
    credits: float
    grade: Optional[str]
    included_in_gpa: bool
    reason: str


class GpaResult(BaseModel):
    gpa: Optional[float]
    total_quality_points: float
    gpa_credits: float
    attempted_credits: float
    earned_credits: float
    included_courses: int
    excluded_courses: int
    audits: List[CourseAudit]


class ParseResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int

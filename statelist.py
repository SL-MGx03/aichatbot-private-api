class CourseItem(TypedDict):
    course_code: str
    course_name: str
    credits: float
    grade: str


class UniversityGPAState(TypedDict):
    file_bytes: bytes                  
    custom_rules_prompt: str           
    degree_type: str                   
    
    markdown_table: str
    json_data: Dict[str, Any]
    
    extracted_courses: List[CourseItem] 
    user_confirmed: bool
    
    total_completed_credits: float
    target_credits: int
    remaining_credits: float
    calculated_gpa: float
    
    final_analysis_report: str          
    error: Optional[str]

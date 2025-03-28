from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = 'Suraj'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=7.5, description='CGPA of student')

new_student = {'age': 28, 'email': 'abc@gmail.com'}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])  # 28

student_json = student.model_dump_json()
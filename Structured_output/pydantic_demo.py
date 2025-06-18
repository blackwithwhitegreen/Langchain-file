from pydantic import BaseModel,Field, EmailStr
from typing import Optional

class Student(BaseModel):

    name : str = 'Rohit' # Set default value
    #name : str # we can do this type also but, here it is not set as default.
    age: Optional[int] = None
    email: EmailStr # Builtin Validation
    cgpa:float = Field(gt=0,lt=9) # Here we can add the default value also --> default = 5,Discription = ......

new_student = {'age':32,'email':'abd@go.com','cgpa':8}

student  = Student(**new_student)# As we can see this is a Pydantic object, but we can save it as dictionay as well desired form

# we can make it dictionay also
student_dict = dict(student)

print(student_dict['age'])

#Similary we can make it as json function also
studnet_json = student.model_dump_json()

print(studnet_json)
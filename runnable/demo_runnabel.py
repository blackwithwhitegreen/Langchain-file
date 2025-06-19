# This file is demo file there is no use of any real model,langchain and anything
import random

class NakliLLM:
    def __init__(self):
        print("LLM Created")

    def predict(self,prompt):
        response_list = [
            'Delhi is the capital of India.',
            'IPL is a cricket Game',
            'AI wil destroy JS one day.'
        ]
        return {'response':random.choice(response_list)}
    
class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)
    

template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length','topic']

)

template.format({'length':'short','topic':'india'})

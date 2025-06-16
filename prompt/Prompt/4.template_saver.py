from langchain_core.prompts import PromptTemplate

# Code for saving the prompt template in desired format.
template = PromptTemplate(
    template = """
Please summarize the poem titled "{poem_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Thematic Elements:  
- Highlight the central themes, emotional tone, and symbolic meanings.  
- Discuss notable literary devices such as metaphors, imagery, or rhyme scheme.  

2. Interpretation Aids:  
- Use relatable analogies or cultural references to enhance understanding.  
- Where appropriate, interpret abstract or figurative language clearly.  

If certain details cannot be reliably inferred from the poem, please respond with:  
"Relevant details could not be determined based on the available content."  

Ensure the summary is accurate, insightful, and aligned with the selected style and length.
""",

input_variables = ['poem_input', 'style_input', 'length_input'],
validate_template = True

)

template.save('template.json')
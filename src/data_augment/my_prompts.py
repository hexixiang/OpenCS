system_prompt_response = """
You are a helpful, precise but picky assistant for checking the quality 
of the [response] to a given [instruction]. 
"""

system_prompt_instruction = """
You are a helpful, precise but picky assistant for checking the quality 
of the [instruction] to a given [response]. 
"""

user_prompt_response = """
[instruction]: {Instruction}
[response]: {Response}
We would like you to answer several questions related to the quality 
of the [response] to the given [instruction]. 
1. Why this [response] is not good for the given [instruction]? Analysis 
based on the Helpfulness, Relevance, Accuracy, Level of Details and Structure. 
2. Based on the reason you provided, please generate a better 
[response] while preserving the same content. To achieve that, you may 
want to adjust the level of details, add bullet points, or use 
comprehensive words, etc. Your answer should be in the format of as below: 
<format>
Step 1: The response is not good for the given instruction because … 
Step 2: The Better response is
```json
{
    "better_response":[better response]
}
```
</format>
"""

user_prompt_instruction = """
[instruction]: {Instruction}
[response]: {Response}
We would like you to answer several questions related to the quality 
of the [instruction] to the given [response]. 
1. Why this [instruction] is not good for the given [response]? Analysis 
based on the Clarity, Specificity, Completeness, and Relevance. 
2. Based on the reason you provided, please generate a better 
[instruction] while preserving the same content. To achieve that, you may 
want to adjust the clarity, add specific details, or use more precise 
languages, etc. Your answer should be in the format of as below, do not remove or edit "### Instruction:" and "### Input:": 
<format>
Step 1: The instruction is not good for the given response because … 
Step 2: The Better instruction is
```json
{
    "better_instruction":[better instruction]
}
```
</format>
"""
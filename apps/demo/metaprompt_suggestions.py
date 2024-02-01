# edited from metaprompt_ddai (general version of latest run_evolution tag POC metaprompt; not hooked up)
# combine with suggestions text in clarifications_metaprompt_pl_tag_poc

# FIXME hardcoded
INSTRUCTION_TEMPLATE = "<assistant_instruction>"
ALL_EXAMPLES_TEMPLATE = "<high_information_gain_examples>"

METAPROMPT_TEMPLATE = """
An assistant has been given the instruction below. Given an input, the assistant is expected to provide an output based on the instruction.

** Begin Assistant Instruction **
<assistant_instruction>
** End Assistant Instruction **

A supervisor has provided examples of inputs and desirable outputs for each example.

Here are some examples provided by the supervisor, the assistant response, and the assistant justification for the response.

** Begin Examples **
<high_information_gain_examples>
** End Examples **

Given the instruction and the examples above, consider that:
- The assistant may have biases or preconceptions regarding certain topics which need to be overcome by emphasizing certain aspects of the instruction.
- There may be lack of clarity in the intent of the instruction.
- It is possible the supervisor examples are incorrect or misleading, given the intended instruction.
- Some combination of all of the above may be true.

Suggest a question that the assistant could ask the supervisor to improve the assistant's performance.
Multiple different suggestions are possible given the considerations above.

Suggestions could include:
- Possible changes to the instructions: general clarifications, additions, deletions, how to handle specific cases
- Possible errors in the labels that the supervisor has provided on examples.

Make detailed suggestions that could have the greatest impact in increasing the assistant's performance. Consider the assistant justification in possible errors.
Provide the relevant transcript number(s) given above. 

Provide justifications for your suggestions, but do not reference the assistant or supervisor in your suggested questions, simply ask the question.
Output your response in a numbered list with the following format:
1. <title>: <question>. <explanation>. Examples(s): <example_number1,example_number2 | All | None>
"""

SUMMARIZATION_EXAMPLE_TEMPLATE = """
Example <example_number>
-- Start input text to summarize --
<example_text_to_summarize>
-- End input text to summarize --

-- Start reference summary provided by supervisor --
<reference_summary>
-- End reference summary provided by supervisor --

-- Start assistant summary --
<assistant_summary>
-- End assistant summary --

-- Start assistant justification --
<assistant_justification>
-- End assistant justification --

"""

GENERAL_EXAMPLE_TEMPLATE = """
Example <example_number>
-- Start input --
<example_text>
-- End input --

-- Start reference answer provided by supervisor --
<reference_answer>
-- End reference answer provided by supervisor --

-- Start assistant answer --
<assistant_answer>
-- End assistant answer --

-- Start assistant justification --
<assistant_justification>
-- End assistant justification --

"""

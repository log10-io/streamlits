import os
import streamlit as st
from suggestions import ReferenceExample, InstructionType, Suggestions

max_num_reference_examples = 5
mock_out_oai_calls = False
# expected_passcode = os.environ["DDS_PASSCODE"]


def process_inputs(
    initial_prompt,
    target_llm,
    reference_input_examples,
    reference_output_examples,
    user_email,
):
    print(initial_prompt)
    print(target_llm)
    print(user_email)

    all_reference_examples = []
    for i_example, (input_text, reference_text) in enumerate(
        zip(reference_input_examples, reference_output_examples), start=1
    ):
        all_reference_examples.append(
            ReferenceExample(
                id_=i_example,
                input_text=input_text,
                reference_output=reference_text,
            )
        )

    if mock_out_oai_calls:
        suggestions = "1. test suggestion blah blah"
    else:
        suggestions_generator = Suggestions(
            initial_prompt=initial_prompt,
            reference_examples=all_reference_examples,
            target_model=target_llm,
            instruction_type=InstructionType.OTHER,  # placeholder for future, maybe
        )
        suggestions = suggestions_generator.generate_suggestions(
            optimizer_model="gpt-4-0613",
            optimizer_temperature=1.0,
            optimizer_seed=0,
            log10_tags=["dd-suggestions", "optimizer"],
        )
    print(suggestions)

    st.session_state["suggestions"] = suggestions
    # TODO send email

    return suggestions


### Start form

## Passcode to gate access
# passcode = st.text_input("Enter passcode: ")

# Prompt (task description)
initial_prompt = st.text_area("Enter your task description (prompt):")

# Target model
target_llm = st.selectbox("Target Model", ["gpt-3.5-turbo-16k-0613", "gpt-4-0613"])

# Reference examples
reference_input_examples = []
reference_output_examples = []

ref_idx = 0
form_key = f"ref_{ref_idx}"
with st.form(key=form_key):
    st.write(f"Reference Example {ref_idx + 1}")
    reference_input = st.text_area("Reference Input")
    reference_output = st.text_area("Reference Output")
    exist_more_examples = st.selectbox("More examples?", ["no", "yes"])
    st.form_submit_button("Next")
    reference_input_examples.append(reference_input)
    reference_output_examples.append(reference_output)

while exist_more_examples == "yes" and ref_idx < max_num_reference_examples - 1:
    ref_idx += 1
    form_key = f"ref_{ref_idx}"
    with st.form(key=form_key):
        st.write(f"Reference Example {ref_idx + 1}")
        reference_input = st.text_area("Reference Input")
        reference_output = st.text_area("Reference Output")
        if ref_idx < max_num_reference_examples - 1:
            exist_more_examples = st.selectbox("More examples?", ["no", "yes"])
        st.form_submit_button("Next")
        reference_input_examples.append(reference_input)
        reference_output_examples.append(reference_output)

if (
    exist_more_examples == "no"
    or len(reference_input_examples) == max_num_reference_examples
) and reference_output_examples[-1] != "":
    # if passcode == expected_passcode:
    user_email = st.text_input(
        "Enter your email and we will send you the results when they are ready"
    )
    if st.button(
        "Submit",
        on_click=lambda: process_inputs(
            initial_prompt,
            target_llm,
            reference_input_examples,
            reference_output_examples,
            user_email,
        ),
    ):
        st.write("\n**Suggestions:**\n")
        st.write(st.session_state["suggestions"])
    # else:
    #    st.write("Invalid passcode")

import json
from dataclasses import dataclass
from enum import Enum
from metaprompt_suggestions import (
    METAPROMPT_TEMPLATE,
    INSTRUCTION_TEMPLATE,
    ALL_EXAMPLES_TEMPLATE,
    SUMMARIZATION_EXAMPLE_TEMPLATE,
    GENERAL_EXAMPLE_TEMPLATE,
)
from log10.load import log10_session, OpenAI
from typing import Any, List, Dict, Optional


tag_justification_instruction = """Provide a justification for the selected tag.
Give a RFC8259-compliant JSON response in the following format: {"answer": "answer", "justification": "justification"}.
"""

summarization_justification_instruction = """Provide justifications for why each part of the summary is important. Give a RFC8259-compliant JSON response in the following format: {"answer": "summary", "justification": "justification"}."""

general_justification_instruction = """Provide a justification for your answer. Give a RFC8259-compliant JSON response in the following format: {"answer": "answer", "justification": "justification"}."""

client = OpenAI()

# CONCERNS
# 1. instrumenting prompt with justification
#    - somehow degrades prompt, since slapping on w/o adapting to contents of prompt
#    - will confuse target model if already asking for justification


@dataclass
class ReferenceExample:
    id_: int  # from 1 to N
    input_text: str
    reference_output: str


@dataclass
class Example:
    reference_example: ReferenceExample
    initial_prompt_completion: str
    initial_prompt_justification: str  # ?? more complex data structure here?


class InstructionType(Enum):
    OTHER = 0
    TAGGING = 1
    SUMMARIZATION = 2


@dataclass
class Suggestions:
    """
    inputs
    - initial_prompt
    - few shot examples (sample, reference output)
    - target model to optimize for
    - instruction type (tagging, summarization); start with narrowed scope

    algo
    1. instrument prompt with justifications
        - depending on instruction type, justification is different (justify inclusion/exclusion, justify label)
        - (detect if justification already included? assume not/also formatting...test what happens if ask for justification twice)
    2. get output for selected model
    3. for tagging, point to ddai algo (most incorrect samples)
    4. for summarization, point to samples where difference in embedding space is the largest
        - for each... I don't remember what this thought was
        - to start, just random or all samples
    """

    initial_prompt: str
    reference_examples: List[ReferenceExample]
    target_model: str
    instruction_type: InstructionType
    # TODO target model temperature, seed, top_k, top_p?

    def __post_init__(self):
        self._add_justification_to_prompt()
        self._get_target_model_outputs()

    def _add_justification_to_prompt(self):
        """Assumes initial_prompt does not already specify outputing justification
        Creates self.instrumented_prompt
        """
        if self.instruction_type == InstructionType.TAGGING:
            justification_instruction = tag_justification_instruction
        elif self.instruction_type == InstructionType.SUMMARIZATION:
            justification_instruction = summarization_justification_instruction
        else:
            justification_instruction = general_justification_instruction
        self.instrumented_prompt = (
            self.initial_prompt + "\n" + justification_instruction
        )
        print("INSTRUMENTED PROMPT:")
        print(self.instrumented_prompt)

    def _get_target_model_outputs(
        self,
        log10_tags: Optional[List[str]] = ["dd-suggestions"],
        max_num_retries: Optional[int] = 3,
    ):
        """Creates list of Example -> self.examples"""
        # what to do about temperature and seed?
        self.examples = []
        for ref_example in self.reference_examples:
            messages = [
                {"role": "system", "content": self.instrumented_prompt},
                {"role": "user", "content": ref_example.input_text},
            ]
            n_try = 0
            output = None
            justification = None
            while n_try < max_num_retries:
                try:
                    completion = self._get_completion(
                        self.target_model,
                        messages,
                        temperature=0.2,
                        seed=0,
                        log10_tags=log10_tags,
                    )
                    json_output = json.loads(completion)
                    # FIXME answer, justification keys hardcoded in templates, could be bug-prone
                    output = json_output["answer"]
                    justification = json_output["justification"]
                except (json.JSONDecodeError, KeyError) as e:
                    n_try += 1
                else:
                    break
            example = Example(
                reference_example=ref_example,
                initial_prompt_completion=output,
                initial_prompt_justification=justification,
            )
            self.examples.append(example)

    def _select_examples(self):
        # select only "incorrect" examples
        # for tagging, most incorrect
        # for summarization, all that fit, most incorrect are farthest away semantically
        # if not fit in context, select most different ones (future)

        # TODO determine how many examples fit in context window
        context_window = 8192  # FIXME hardcoded for gpt-4-0613

        # adapted from suggest_clarifications_pl_tag_poc select_incorrect_examples
        # most_inaccurate method
        if self.instruction_type == InstructionType.TAGGING:
            selected_examples = []
            for example in self.examples:
                prediction = example.initial_prompt_completion
                ground_truth = example.reference_example.reference_output
                # TODO save logprobs, select most inaccurate from both labels
                if prediction != ground_truth:
                    selected_examples.append(example)
        # elif self.instruction_type == InstructionType.SUMMARIZATION:
        else:  # all other types for now
            # FIXME to start, just take all samples
            # skip examples that have None completion, justification
            selected_examples = [
                ex
                for ex in self.examples
                if ex.initial_prompt_completion is not None
                and ex.initial_prompt_justification is not None
            ]
        return selected_examples

    def _create_examples_text(self, examples: List[Example]) -> str:
        all_example_strs = []
        if self.instruction_type == InstructionType.SUMMARIZATION:
            for example in examples:
                example_str = (
                    SUMMARIZATION_EXAMPLE_TEMPLATE.replace(
                        "<example_number>", str(example.reference_example.id_)
                    )
                    .replace(
                        "<example_text_to_summarize>",
                        example.reference_example.input_text,
                    )
                    .replace(
                        "<reference_summary>",
                        example.reference_example.reference_output,
                    )
                    .replace("<assistant_summary>", example.initial_prompt_completion)
                    .replace(
                        "<assistant_justification>",
                        example.initial_prompt_justification,
                    )
                )
                all_example_strs.append(example_str)
        else:
            # assume TAGGING can use general template
            # SUMMARIZATION template is the same except specific use of word "summary"
            for example in examples:
                example_str = (
                    GENERAL_EXAMPLE_TEMPLATE.replace(
                        "<example_number>", str(example.reference_example.id_)
                    )
                    .replace(
                        "<example_text>",
                        example.reference_example.input_text,
                    )
                    .replace(
                        "<reference_answer>",
                        example.reference_example.reference_output,
                    )
                    .replace("<assistant_answer>", example.initial_prompt_completion)
                    .replace(
                        "<assistant_justification>",
                        example.initial_prompt_justification,
                    )
                )
                all_example_strs.append(example_str)
        return "\n".join(all_example_strs)

    def _create_metaprompt(self, selected_examples):
        metaprompt = METAPROMPT_TEMPLATE.replace(
            INSTRUCTION_TEMPLATE, self.initial_prompt
        )
        metaprompt = metaprompt.replace(
            ALL_EXAMPLES_TEMPLATE, self._create_examples_text(selected_examples)
        )
        return metaprompt

    def _get_completion(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        seed: int,
        log10_tags: List[str],
    ):
        params = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "seed": seed,
        }
        with log10_session(tags=log10_tags):
            api_response = client.chat.completions.create(**params)
        completion = api_response.choices[0].message.content
        return completion

    def generate_suggestions(
        self,
        optimizer_model: str,
        optimizer_temperature: float,
        optimizer_seed: int,
        log10_tags: Optional[List[str]] = ["dd-suggestions"],
    ) -> str:
        incorrect_examples = self._select_examples()
        metaprompt = self._create_metaprompt(incorrect_examples)
        messages = [{"role": "system", "content": metaprompt}]
        suggestions = self._get_completion(
            optimizer_model,
            messages,
            optimizer_temperature,
            optimizer_seed,
            log10_tags,
        )
        return suggestions

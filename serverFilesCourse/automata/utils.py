from shared_utils import strings_of_length_at_most_n
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
from automata.fa.fa import FA
from random import sample
from typing import Tuple, List, Any, Union, Optional
from typing_extensions import assert_never
from shared_utils import replace_empty

LATEX_EPSILON = r"\varepsilon"


def elem_to_latex(elem: str) -> str:
    return elem if elem else LATEX_EPSILON


def check_dfa(
    submitted_dfa: DFA, reference_dfa: DFA, max_length_to_check: int
) -> Tuple[List[str], List[str]]:
    """
    Parameters
      - submitted_dfa: DFA submitted by the student
      - reference_dfa: Reference DFA for this problem
      - max_length_to_check: Maximum length to check regex string for feedback
    Return value
      - Return a pair of lists of strings: false_positives, false_negatives
    Exceptions
      - Throw ValueError if input symbols don't match or if DFAs are equivalent
    """

    if submitted_dfa.input_symbols != reference_dfa.input_symbols:
        raise ValueError("Input symbols for submitted DFA don't match reference")

    # Brute Force Check
    false_positives: List[str] = []
    false_negatives: List[str] = []

    for bitstring in strings_of_length_at_most_n(
        0, max_length_to_check, alphabet=submitted_dfa.input_symbols
    ):
        accepted_by_reference_DFA = reference_dfa.accepts_input(bitstring)
        accepted_by_submitted_DFA = submitted_dfa.accepts_input(bitstring)

        if not accepted_by_reference_DFA and accepted_by_submitted_DFA:
            false_positives.append(bitstring)
        elif accepted_by_reference_DFA and not accepted_by_submitted_DFA:
            false_negatives.append(bitstring)

    if false_positives or false_negatives:
        return false_positives, false_negatives

    # Graph Based Check
    counter = submitted_dfa.find_counterexample(reference_dfa)
    if counter is None:
        raise ValueError("DFAs are equivalent.")
    elif submitted_dfa.accepts_input(counter):
        false_positives.append(counter)
    else:
        false_negatives.append(counter)
    return false_positives, false_negatives


def states_to_string(obj: Any) -> str:
    if isinstance(obj, set):
        if not obj:
            return "∅"

        # Weird highlighting, but code inside braces is indeed run
        return (
            f"{{{', '.join(states_to_string(item) for item in sorted(obj, key=str))}}}"
        )

    elif isinstance(obj, tuple):
        if not obj:
            raise ValueError("Tuple shouldn't be empty")
        return f"({', '.join(states_to_string(item) for item in obj)})"

    return replace_empty(str(obj))


def sample_input_strings(
    max_input_string_len: int, num_rand_choices: int, fa: FA
) -> Tuple[List[str], List[str]]:
    """
    Samples accepted and not accepted input strings for the given fa. Converts
    for display on the frontend.
    """

    # Get all accepted and non-accepted strings of length at most n
    accepted = []
    not_accepted = []

    for x in strings_of_length_at_most_n(
        1, max_input_string_len, alphabet=fa.input_symbols
    ):
        if fa.accepts_input(x):
            accepted.append(x)
        else:
            not_accepted.append(x)

    # Next, do random sampling based on the number of accepted and rejected strings
    sampled_accepted = []
    sampled_not_accepted = []

    if len(accepted) < (num_rand_choices // 2):
        sampled_accepted = accepted
        sampled_not_accepted = sample(not_accepted, num_rand_choices - len(accepted))

    elif len(not_accepted) < (num_rand_choices // 2 + num_rand_choices % 2):
        sampled_accepted = sample(accepted, num_rand_choices - len(not_accepted))
        sampled_not_accepted = not_accepted

    else:
        sampled_accepted = sample(accepted, num_rand_choices // 2)
        sampled_not_accepted = sample(
            not_accepted, num_rand_choices // 2 + num_rand_choices % 2
        )

    # Always include the empty string
    if fa.accepts_input(""):
        sampled_accepted.append(LATEX_EPSILON)
    else:
        sampled_not_accepted.append(LATEX_EPSILON)

    # Return the result
    return sampled_accepted, sampled_not_accepted


def get_equiv_dfa(fsm: Union[DFA, NFA]) -> DFA:
    if isinstance(fsm, NFA):
        return DFA.from_nfa(fsm)
    elif isinstance(fsm, DFA):
        return fsm

    assert_never(fsm)


def generate_dfa_feedback_string(
    student_equiv_dfa: DFA,
    reference_equiv_dfa: DFA,
    max_length_to_check: int,
    student_input_name: str,
) -> str:
    """
    Generate a feedback string for use by externally graded questions. The
    'language' here is defined by reference_equiv_dfa.
    """

    res = []

    false_positives, false_negatives = check_dfa(
        student_equiv_dfa, reference_equiv_dfa, max_length_to_check
    )

    assert false_positives or false_negatives

    if false_positives:
        res.append(
            f"Here are some strings matched by your {student_input_name}"
            " which are not in the language:"
        )

        for x in false_positives[:max_length_to_check]:
            res.append(replace_empty(x))

        # Add blank line between false positives and false negatives, if both exist
        if false_negatives:
            res.append("")

    if false_negatives:
        res.append(
            "Here are some strings in the language which"
            f" aren't matched by your {student_input_name}:"
        )

        for x in false_negatives[:max_length_to_check]:
            res.append(replace_empty(x))

    return "\n".join(res)


def generate_dfa_feedback_html(
    student_equiv_dfa: DFA,
    reference_equiv_dfa: DFA,
    max_length_to_check: int,
    student_input_name: str,
    *,
    original_student_fa: Optional[FA] = None,
) -> str:
    """
    Generate feedback html for elements. The 'language' here is defined by
    reference_equiv_dfa.
    """

    def latex_prepare_first_n_list(elements: List[str], n: int) -> List[str]:
        "Format a list of strings for display as HTML"

        string_list = ["<ul>\n"]
        string_list.extend(
            f"<li>${elem_to_latex(elem)}$</li>\n" for elem in elements[:n]
        )
        string_list.append("</ul>")
        return string_list

    false_positives, false_negatives = check_dfa(
        student_equiv_dfa, reference_equiv_dfa, max_length_to_check
    )

    assert false_positives or false_negatives
    feedback_string_list = []

    if false_positives:
        feedback_string_list.append(
            f"<p>Here are some strings matched by your {student_input_name} which are not in the language:</p>"
        )
        feedback_string_list.extend(
            latex_prepare_first_n_list(false_positives, max_length_to_check)
        )

        if original_student_fa is not None:
            target_str = false_positives[0]

            input_path, was_acepted = original_student_fa.get_input_path(target_str)

            # Assertion here to make sure this works as expected. TODO remove later.
            assert was_acepted

            # Case where we accept immeditely
            if not input_path:
                assert target_str == ""

                feedback_string_list.append(
                    f"<p>For instance, the string ${elem_to_latex(target_str)}$ was accepted without taking any transitions.</p>"
                )
            else:
                feedback_string_list.append(
                    f"<p>For instance, here's the sequence of states taken to accept the input ${elem_to_latex(target_str)}$:</p>"
                )

                state_sequence_list = ["$$", input_path[0][0]]

                for _, to_state, symbol in input_path:
                    state_sequence_list.append(
                        rf" \xrightarrow{{{elem_to_latex(symbol)}}} "
                    )
                    state_sequence_list.append(str(to_state))

                state_sequence_list.append("$$")

                feedback_string_list.append("".join(state_sequence_list))

    if false_negatives:
        feedback_string_list.append(
            f"<p>Here are some strings in the language which aren't matched by your {student_input_name}:</p>"
        )
        feedback_string_list.extend(
            latex_prepare_first_n_list(false_negatives, max_length_to_check)
        )

    return "".join(feedback_string_list)


def compute_partial_credit(
    student_equiv_dfa: DFA,
    reference_equiv_dfa: DFA,
    *,
    word_limit_to_check: Optional[int] = None,
) -> float:
    """
    Computes the approximate density difference between student_equiv_dfa and reference_equiv_dfa.
    Assumes input DFAs are minimal. Used for giving partial credit to students for incorrect answers.
    See section 3.3 for details: https://www.cis.upenn.edu/~alur/Ijcai13.pdf
    """

    if word_limit_to_check is None:
        word_limit_to_check = 2 * len(reference_equiv_dfa.states)

    # Raise exception here to prevent really slow grading / weird freakouts
    if word_limit_to_check > 32:
        raise ValueError(f"Word limit to check {word_limit_to_check} too high.")

    difference_dfa = student_equiv_dfa ^ reference_equiv_dfa

    res = 0.0
    for n in range(word_limit_to_check + 1):
        difference_frac = difference_dfa.count_words_of_length(n) / max(
            reference_equiv_dfa.count_words_of_length(n), 1
        )
        res += difference_frac

    similarity_score = min(1.0, res / (word_limit_to_check + 1))

    return 1.0 - similarity_score

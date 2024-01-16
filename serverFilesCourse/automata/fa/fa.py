#!/usr/bin/env python3
"""Classes and methods for working with all finite automata."""

import abc
from typing import Any, Dict, Generator, List, NoReturn, Set, Tuple

import automata.base.exceptions as exceptions

FAStateT = Any

class FA(metaclass=abc.ABCMeta):
    """An abstract base class for all finite automata."""

    # Add slots to speed up class
    __slots__ = ['states', 'input_symbols', 'transitions', 'initial_state', 'final_states']

    states: Set[FAStateT]
    input_symbols: Set[str]
    transitions: Dict[FAStateT, Any]
    initial_state: FAStateT
    final_states: Set[FAStateT]

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initialize a complete automaton."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self) -> None:
        """Raise exception if this automaton is not internally consistent."""
        raise NotImplementedError

    @abc.abstractmethod
    def read_input_stepwise(self, input_str: str) -> Generator[FAStateT, None, None]:
        """Return a generator that yields each step while reading input."""
        raise NotImplementedError

    def read_input(self, input_str: str) -> FAStateT:
        """
        Check if the given string is accepted by this automaton.

        Return the automaton's final configuration if this string is valid.
        """
        *_, last_config = self.read_input_stepwise(input_str)
        return last_config

    def accepts_input(self, input_str: str) -> bool:
        """Return True if this automaton accepts the given input."""
        try:
            self.read_input(input_str)
            return True
        except exceptions.RejectionException:
            return False

    def _validate_input_symbols(self) -> None:
        """Raise an error if any input symbols are not just one character."""
        for input_symbol in self.input_symbols:
            if not isinstance(input_symbol, str) or len(input_symbol) != 1:
                raise exceptions.InvalidSymbolError(
                    f"'{input_symbol}' is not a valid input symbol")

    def _validate_initial_state(self) -> None:
        """Raise an error if the initial state is invalid."""
        if self.initial_state not in self.states:
            raise exceptions.InvalidStateError(
                f'{self.initial_state} is not a valid initial state')

    def _validate_initial_state_transitions(self) -> None:
        """Raise an error if the initial state has no transitions defined."""
        if self.initial_state not in self.transitions:
            raise exceptions.MissingStateError(
                f'initial state {self.initial_state} has no transitions defined')

    def _validate_transition_start_states(self) -> None:
        """Raise an error if transition start states are missing."""
        for state in self.states:
            if state not in self.transitions:
                raise exceptions.MissingStateError(
                    f'transition start state {state} is missing')

    def _validate_final_states(self) -> None:
        """Raise an error if any final states are invalid."""
        invalid_states = self.final_states - self.states
        if invalid_states:
            states_str = ', '.join(str(state) for state in invalid_states)
            raise exceptions.InvalidStateError(
                f'final states are not valid ({states_str})')

    def __delattr__(self, name: str) -> NoReturn:
        "Set custom delattr to make class immutable."
        raise AttributeError(f'This {type(self).__name__} is immutable')

    def __setattr__(self, name: str, val: Any) -> NoReturn:
        "Set custom setattr to make class immutable."
        raise AttributeError(f'This {type(self).__name__} is immutable')

    @abc.abstractmethod
    def get_input_path(
        self, input_str: str
    ) -> Tuple[List[Tuple[FAStateT, FAStateT, str]], bool]:
        """Calculate the path taken by input."""

        raise NotImplementedError(
            f"get_input_path is not implemented for {self.__class__}"
        )

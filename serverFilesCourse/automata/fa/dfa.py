#!/usr/bin/env python3
"""Classes and methods for working with deterministic finite automata."""

from __future__ import annotations
from itertools import count, chain, product
from collections import deque
from random import randint, shuffle
from typing_extensions import assert_never
from typing import (
    Dict, Set, Generator, Deque, Tuple, Type,
    cast, TypedDict, List, FrozenSet, Optional, Literal, overload, Union
)
from enum import IntEnum
from automata.base.partition import PartitionRefinement

import copy
import networkx as nx
import pygraphviz
import automata.base.exceptions as exceptions
import shared_utils as su
import automata.fa.fa as fa
import automata.fa.nfa as nfa
import array

class DFAJsonDict(TypedDict):
    "A class with type signatures for the dfa json dict"
    states: List[DFAStateT]
    input_symbols: List[str]
    transitions: DFATransitionsT
    initial_state: DFAStateT
    final_states: List[DFAStateT]

class OriginEnum(IntEnum):
    SELF = 0
    OTHER = 1

DFAStateT = fa.FAStateT

GraphT = Dict[DFAStateT, Set[DFAStateT]]
DFAPathT = Dict[str, DFAStateT]
DFATransitionsT = Dict[DFAStateT, DFAPathT]
DFAStatePairT = Tuple[DFAStateT, OriginEnum]

WitnessDictT = Optional[Dict[Tuple[DFAStatePairT, DFAStatePairT], Tuple[DFAStatePairT, DFAStatePairT, str]]]

class DFA(fa.FA):
    """A deterministic finite automaton."""

    # Add extra slots from base class
    __slots__ = ['allow_partial', '_count_cache']

    transitions: DFATransitionsT
    allow_partial: bool
    _count_cache: Dict[DFAStateT, array.array[int]]

    def __init__(self,
                 *,
                 states: Set[DFAStateT],
                 input_symbols: Set[str],
                 transitions: DFATransitionsT,
                 initial_state: DFAStateT,
                 final_states: Set[DFAStateT],
                 allow_partial: bool = False) -> None:
        """Initialize a complete DFA. Use weird initialization because of immutability"""

        object.__setattr__(self, "states", states.copy())
        object.__setattr__(self, "input_symbols", input_symbols.copy())
        object.__setattr__(self, "transitions", copy.deepcopy(transitions))
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "final_states", final_states.copy())
        object.__setattr__(self, "allow_partial", allow_partial)
        object.__setattr__(self, "_count_cache", {
                state: array.array("Q", [1] if state in self.final_states else [0])
                for state in self.states
            }
        )

        self.validate()


    def find_counterexample(self, other_DFA: DFA) -> Optional[str]:
        return self._equality_check(other_DFA, True)

    @overload
    def _equality_check(self, other_DFA: DFA, compute_counterexample: Literal[False]) -> bool: ...

    @overload
    def _equality_check(self, other_DFA: DFA, compute_counterexample: Literal[True]) -> Optional[str]: ...


    def _equality_check(self, other_DFA: DFA, compute_counterexample: bool) -> Union[bool, Optional[str]]:
        """
        Finds the minimal distinguishing string between this DFA and other_DFA.

        Uses the witness algorithm from page 48 of https://scholarworks.rit.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=7944&context=theses
        combined with https://arxiv.org/abs/0907.5058

        If compute_counterexample is True, concretely computes the witness string (if it exists), otherwise
        just returns a boolean
        """

        def is_final_state(state_pair: DFAStatePairT) -> bool:
            state, origin_enum = state_pair

            if origin_enum is OriginEnum.SELF:
                return state in self.final_states

            elif origin_enum is OriginEnum.OTHER:
                return state in other_DFA.final_states

            assert_never(origin_enum)

        def transition(state_pair: DFAStatePairT, symbol: str) -> DFAStatePairT:
            state, origin_enum = state_pair

            if origin_enum is OriginEnum.SELF:
                return (self.transitions[state][symbol], origin_enum)

            elif origin_enum is OriginEnum.OTHER:
                return (other_DFA.transitions[state][symbol], origin_enum)

            assert_never(origin_enum)

        witness: WitnessDictT = dict() if compute_counterexample else None

        # Get new initial states
        initial_state_a = (self.initial_state, OriginEnum.SELF)
        initial_state_b = (other_DFA.initial_state, OriginEnum.OTHER)

        # Get data structures
        state_sets = nx.utils.union_find.UnionFind([initial_state_a, initial_state_b])
        pair_stack: Deque[Tuple[DFAStatePairT, DFAStatePairT]] = deque()

        # Do union find
        state_sets.union(initial_state_a, initial_state_b)
        pair_stack.append((initial_state_a, initial_state_b))

        while pair_stack:
            # FIFO ordering actually matters here
            q_a, q_b = pair_stack.popleft()

            if is_final_state(q_a) ^ is_final_state(q_b):
                if witness is not None:
                    witness_str = []
                    key = (q_a, q_b)
                    while key != (initial_state_a, initial_state_b):
                        qka, qkb, symbol = witness[key]
                        witness_str.append(symbol)
                        key = (qka, qkb)
                    return "".join(witness_str[::-1])

                return False

            for symbol in self.input_symbols:
                r_a = state_sets[transition(q_a, symbol)]
                r_b = state_sets[transition(q_b, symbol)]

                if r_a != r_b:
                    state_sets.union(r_a, r_b)
                    pair_stack.append((r_a, r_b))

                    if witness is not None:
                        witness[(r_a, r_b)] = (q_a, q_b, symbol)

        if compute_counterexample:
            return None

        return True

    def __eq__(self, other: object) -> bool:
        """
        Return True if two DFAs are equivalent.
        """

        # Must be another DFA and have equal alphabets
        if not isinstance(other, DFA) or self.input_symbols != other.input_symbols:
            return NotImplemented

        return self._equality_check(other, False)


    def __le__(self, other: DFA) -> bool:
        """Return True if this DFA is a subset of (or equal to) another DFA."""
        if isinstance(other, DFA):
            return self.issubset(other)
        else:
            raise NotImplementedError

    def __ge__(self, other: DFA) -> bool:
        """Return True if this DFA is a superset of another DFA."""
        if isinstance(other, DFA):
            return self.issuperset(other)
        else:
            raise NotImplementedError

    def __lt__(self, other: DFA) -> bool:
        """Return True if this DFA is a strict subset of another DFA."""
        if isinstance(other, DFA):
            return self <= other and self != other
        else:
            raise NotImplementedError

    def __gt__(self, other: DFA) -> bool:
        """Return True if this DFA is a strict superset of another DFA."""
        if isinstance(other, DFA):
            return self >= other and self != other
        else:
            raise NotImplementedError

    def __sub__(self, other: DFA) -> DFA:
        """Return a DFA that is the difference of this DFA and another DFA."""
        if isinstance(other, DFA):
            return self.difference(other)
        else:
            raise NotImplementedError

    def __or__(self, other: DFA) -> DFA:
        """Return the union of this DFA and another DFA."""
        if isinstance(other, DFA):
            return self.union(other)
        else:
            raise NotImplementedError

    def __and__(self, other: DFA) -> DFA:
        """Return the intersection of this DFA and another DFA."""
        if isinstance(other, DFA):
            return self.intersection(other)
        else:
            raise NotImplementedError

    def __xor__(self, other: DFA) -> DFA:
        """Return the symmetric difference of this DFA and another DFA."""
        if isinstance(other, DFA):
            return self.symmetric_difference(other)
        else:
            raise NotImplementedError

    def __invert__(self) -> DFA:
        """Return the complement of this DFA and another DFA."""
        return self.complement()

    def copy(self) -> DFA:
        """Create a deep copy of the automaton."""
        return DFA(
            states = self.states,
            input_symbols = self.input_symbols,
            transitions = self.transitions,
            initial_state = self.initial_state,
            final_states = self.final_states,
            allow_partial = self.allow_partial
        )

    def _validate_transition_missing_symbols(self, start_state: DFAStateT, paths: DFAPathT) -> None:
        """Raise an error if the transition input_symbols are missing."""
        if self.allow_partial:
            return
        for input_symbol in self.input_symbols:
            if input_symbol not in paths:
                raise exceptions.MissingSymbolError(
                    f"state {start_state} is missing transitions for symbol {input_symbol}")

    def _validate_transition_invalid_symbols(self, start_state: DFAStateT, paths: DFAPathT) -> None:
        """Raise an error if transition input symbols are invalid."""
        for input_symbol in paths.keys():
            if input_symbol not in self.input_symbols:
                raise exceptions.InvalidSymbolError(
                    f"state {start_state} has invalid transition symbol {input_symbol}")

    def _validate_transition_end_states(self, start_state: DFAStateT, paths: DFAPathT) -> None:
        """Raise an error if transition end states are invalid."""
        for end_state in paths.values():
            if end_state not in self.states:
                raise exceptions.InvalidStateError(
                    f"end state {end_state} for transition on {start_state} is not valid")

    def _validate_transitions(self, start_state: DFAStateT, paths: DFAPathT) -> None:
        """Raise an error if transitions are missing or invalid."""
        self._validate_transition_missing_symbols(start_state, paths)
        self._validate_transition_invalid_symbols(start_state, paths)
        self._validate_transition_end_states(start_state, paths)

    def validate(self) -> None:
        """Raise exception if this DFA is not internally consistent."""
        self._validate_input_symbols()
        self._validate_transition_start_states()
        for start_state, paths in self.transitions.items():
            self._validate_transitions(start_state, paths)
        self._validate_initial_state()
        self._validate_final_states()


    def _get_next_current_state(self, current_state: DFAStateT, input_symbol: str) -> DFAStateT:
        """
        Follow the transition for the given input symbol on the current state.

        Raise an error if the transition does not exist.
        """
        if input_symbol in self.transitions[current_state]:
            return self.transitions[current_state][input_symbol]
        else:
            raise exceptions.RejectionException(
                f"{input_symbol} is not a valid input symbol")

    def _check_for_input_rejection(self, current_state: DFAStateT) -> None:
        """Raise an error if the given config indicates rejected input."""
        if current_state not in self.final_states:
            raise exceptions.RejectionException(
                f"the DFA stopped on a non-final state ({current_state})")

    def read_input_stepwise(self, input_str: str, *, ignore_rejection: bool = False) -> Generator[DFAStateT, None, None]:
        """
        Check if the given string is accepted by this DFA.

        Yield the current configuration of the DFA at each step.
        """
        current_state = self.initial_state

        yield current_state
        for input_symbol in input_str:
            current_state = self._get_next_current_state(
                current_state, input_symbol)
            yield current_state

        if not ignore_rejection:
            self._check_for_input_rejection(current_state)

    def read_input_from_state(self, state: DFAStateT, input: str) -> DFAStateT:
        """
        Read in an input from a valid state of the DFA

        Return the resulting state after reading the input.
        """

        new_dfa = DFA(
            states=self.states,
            input_symbols=self.input_symbols,
            transitions=self.transitions,
            initial_state=state,
            final_states=self.states
        )

        return new_dfa.read_input(input)

    def _get_digraph(self) -> nx.DiGraph:
        """Return a digraph corresponding to this DFA with transition symbols ignored"""
        return nx.DiGraph([
            (start_state, end_state)
            for start_state, transition in self.transitions.items()
            for end_state in transition.values()
        ])

    def get_reachable_states(self) -> Set[DFAStateT]:
        """Compute the states which are reachable from the initial state."""

        visited_set: Set[DFAStateT] = set()
        queue: Deque[DFAStateT] = Deque()

        queue.append(self.initial_state)
        visited_set.add(self.initial_state)

        while queue:
            state = queue.popleft()

            for chr in self.input_symbols:
                next_state = self.transitions[state][chr]

                if next_state not in visited_set:
                    visited_set.add(next_state)
                    queue.append(next_state)

        return visited_set

    def minify(self, retain_names: bool = False) -> DFA:
        """
        Create a minimal DFA which accepts the same inputs as this DFA.

        First, non-reachable states are removed.
        Then, similiar states are merged using Hopcroft's Algorithm.
            retain_names: If True, merged states retain names.
                          If False, new states will be named 0, ..., n-1.
        """

        # Compute reachable states and final states
        reachable_states = self.get_reachable_states()
        reachable_final_states = self.final_states & reachable_states

        # Set up partition data structure
        eq_classes = PartitionRefinement(reachable_states)
        refinement = eq_classes.refine(reachable_final_states)

        final_states_id = refinement[0][0] if refinement else eq_classes.get_set_ids()[0]

        # Precompute back map for transitions
        transition_back_map: Dict[str, Dict[DFAStateT, List[DFAStateT]]] = {
            symbol: {
                end_state: list()
                for end_state in reachable_states
            }
            for symbol in self.input_symbols
        }

        for start_state, path in self.transitions.items():
            if start_state in reachable_states:
                for symbol, end_state in path.items():
                    if end_state in reachable_states:
                        transition_back_map[symbol][end_state].append(start_state)

        origin_dicts = tuple(transition_back_map.values())
        processing = {final_states_id}

        while processing:
            active_set = tuple(eq_classes.get_set_by_id(processing.pop()))

            for origin_dict in origin_dicts:
                states_that_move_into_active_state = chain.from_iterable(
                    origin_dict[end_state] for end_state in active_set
                )

                # Refine set partition by states moving into current active one
                new_eq_class_pairs = eq_classes.refine(states_that_move_into_active_state)

                for (YintX_id, YdiffX_id) in new_eq_class_pairs:
                    if YdiffX_id in processing:
                        processing.add(YintX_id)
                    else:
                        if len(eq_classes.get_set_by_id(YintX_id)) <= len(eq_classes.get_set_by_id(YdiffX_id)):
                            processing.add(YintX_id)
                        else:
                            processing.add(YdiffX_id)


        # now eq_classes are good to go, make them a list for ordering
        eq_class_name_pairs: List[Tuple[DFAStateT, Set[DFAStateT]]] = (
            [(frozenset(eq), eq) for eq in eq_classes.get_sets()] if retain_names else
            list(enumerate(eq_classes.get_sets()))
        )


        # need a backmap to prevent constant calls to index
        back_map = {
            state: name
            for name, eq in eq_class_name_pairs
            for state in eq
        }

        new_input_symbols = self.input_symbols
        new_states = set(back_map.values())
        new_initial_state = back_map[self.initial_state]
        new_final_states = {back_map[acc] for acc in reachable_final_states}
        new_transitions: DFATransitionsT = {
            name: {
                letter: back_map[self.transitions[next(iter(eq))][letter]]
                for letter in self.input_symbols
            }
            for name, eq in eq_class_name_pairs
        }

        return DFA(
            states=new_states,
            input_symbols=new_input_symbols,
            transitions=new_transitions,
            initial_state=new_initial_state,
            final_states=new_final_states,
        )

    def _cross_product(self, other: DFA, final_states: Set[DFAStateT]) -> DFA:
        """
        Creates a new DFA which is the cross product of DFAs self and other
        with the given set of final states.
        """
        assert self.input_symbols == other.input_symbols
        new_states = set(product(self.states, other.states))

        new_transitions: DFATransitionsT = {
            (state_a, state_b): {
                symbol: (transitions_a[symbol], transitions_b[symbol])
                for symbol in self.input_symbols
            }
            for (state_a, transitions_a), (state_b, transitions_b) in
            product(self.transitions.items(), other.transitions.items())
        }

        new_initial_state = (self.initial_state, other.initial_state)

        return DFA(
            states=cast(Set[DFAStateT], new_states),
            input_symbols=self.input_symbols,
            transitions=new_transitions,
            initial_state=new_initial_state,
            final_states=final_states
        )

    def union(self, other: DFA, *, retain_names: bool = False, minify: bool = True) -> DFA:
        """
        Takes as input two DFAs M1 and M2 which
        accept languages L1 and L2 respectively.
        Returns a DFA which accepts the union of L1 and L2.
        """

        new_final_states = {
            (state_a, state_b)
            for state_a, state_b in product(self.states, other.states)
            if (state_a in self.final_states or state_b in other.final_states)
        }

        new_dfa = self._cross_product(other, new_final_states)

        if minify:
            return new_dfa.minify(retain_names=retain_names)

        return new_dfa

    def intersection(self, other: DFA, *, retain_names: bool = False, minify: bool = True) -> DFA:
        """
        Takes as input two DFAs M1 and M2 which
        accept languages L1 and L2 respectively.
        Returns a DFA which accepts the intersection of L1 and L2.
        """

        new_final_states = set(product(self.final_states, other.final_states))
        new_dfa = self._cross_product(other, new_final_states)

        if minify:
            return new_dfa.minify(retain_names=retain_names)
        return new_dfa

    def difference(self, other: DFA, *, retain_names: bool = False, minify: bool = True) -> DFA:
        """
        Takes as input two DFAs M1 and M2 which
        accept languages L1 and L2 respectively.
        Returns a DFA which accepts the difference of L1 and L2.
        """

        new_final_states = set(product(self.final_states, other.states - other.final_states))
        new_dfa = self._cross_product(other, new_final_states)

        if minify:
            return new_dfa.minify(retain_names=retain_names)
        return new_dfa

    def symmetric_difference(self, other: DFA, *, retain_names: bool = False, minify: bool = True) -> DFA:
        """
        Takes as input two DFAs M1 and M2 which
        accept languages L1 and L2 respectively.
        Returns a DFA which accepts the symmetric difference of L1 and L2.
        """

        new_final_states = {
            (state_a, state_b)
            for state_a, state_b in product(self.states, other.states)
            if (state_a in self.final_states) ^ (state_b in other.final_states)
        }
        new_dfa = self._cross_product(other, new_final_states)

        if minify:
            return new_dfa.minify(retain_names=retain_names)
        return new_dfa

    def complement(self) -> DFA:
        """Return the complement of this DFA."""

        return DFA(
            states = self.states,
            input_symbols = self.input_symbols,
            transitions = self.transitions,
            initial_state = self.initial_state,
            final_states = self.states - self.final_states,
            allow_partial = self.allow_partial
        )

    def _get_reachable_states_product(self, other: DFA) -> nx.DiGraph:
        """Get reachable states corresponding to product DFA between self and other"""

        assert self.input_symbols == other.input_symbols

        visited_set: Set[Tuple[DFAStateT, DFAStateT]] = set()
        queue: Deque[Tuple[DFAStateT, DFAStateT]] = Deque()

        product_initial_state = (self.initial_state, other.initial_state)
        queue.append(product_initial_state)
        visited_set.add(product_initial_state)

        while queue:
            q_a, q_b = queue.popleft()

            for chr in self.input_symbols:
                product_state = (self.transitions[q_a][chr], other.transitions[q_b][chr])

                if product_state not in visited_set:
                    visited_set.add(product_state)
                    queue.append(product_state)

        return visited_set

    def issubset(self, other: DFA) -> bool:
        """Return True if this DFA is a subset of another DFA."""

        for (state_a, state_b) in self._get_reachable_states_product(other):
            if state_a in self.final_states and state_b not in other.final_states:
                return False

        return True

    def issuperset(self, other: DFA) -> bool:
        """Return True if this DFA is a superset of another DFA."""
        return other.issubset(self)

    def isdisjoint(self, other: DFA) -> bool:
        """Return True if this DFA has no common elements with another DFA."""

        for (state_a, state_b) in self._get_reachable_states_product(other):
            if state_a in self.final_states and state_b in other.final_states:
                return False

        return True

    def isempty(self) -> bool:
        """Return True if this DFA is completely empty."""
        return len(self.get_reachable_states() & self.final_states) == 0

    def isfinite(self) -> bool:
        """
        Returns True if the DFA accepts a finite language, False otherwise.
        """
        G = self._get_digraph()

        accessible_nodes = nx.descendants(G, self.initial_state)
        accessible_nodes.add(self.initial_state)

        coaccessible_nodes: Set[DFAStateT] = self.final_states.union(*(
            nx.ancestors(G, state)
            for state in self.final_states
        ))

        important_nodes = accessible_nodes.intersection(coaccessible_nodes)

        try:
            nx.find_cycle(G.subgraph(important_nodes))
            return False
        except nx.exception.NetworkXNoCycle:
            return True

    @classmethod
    def from_nfa(cls: Type[DFA], target_nfa: nfa.NFA, retain_names: bool = False) -> DFA:
        """Initialize this DFA as one equivalent to the given NFA."""
        dfa_states: Set[DFAStateT] = set()
        dfa_symbols = target_nfa.input_symbols
        dfa_transitions: DFATransitionsT = dict()

        # Data structures for state renaming
        new_state_name_dict: Dict[FrozenSet[DFAStateT], int] = dict()
        state_name_counter = count(0)
        def get_name_renamed(states: FrozenSet[DFAStateT]) -> DFAStateT:
            nonlocal state_name_counter, new_state_name_dict
            return new_state_name_dict.setdefault(states, next(state_name_counter))

        def get_name_original(states: FrozenSet[DFAStateT]) -> DFAStateT:
            return states

        get_name = get_name_original if retain_names else get_name_renamed

        # equivalent DFA states states
        nfa_initial_states = frozenset(target_nfa._get_lambda_closure(target_nfa.initial_state))
        dfa_initial_state = get_name(nfa_initial_states)
        dfa_final_states: Set[DFAStateT] = set()

        state_queue: Deque[FrozenSet[DFAStateT]] = deque()
        state_queue.append(nfa_initial_states)
        while state_queue:
            current_states = state_queue.popleft()
            current_state_name: DFAStateT = get_name(current_states)
            if current_state_name in dfa_states:
                # We've been here before and nothing should have changed.
                continue

            # Add NFA states to DFA as it is constructed from NFA.
            dfa_states.add(current_state_name)
            dfa_transitions[current_state_name] = {}
            if (current_states & target_nfa.final_states):
                dfa_final_states.add(current_state_name)

            # Enqueue the next set of current states for the generated DFA.
            for input_symbol in target_nfa.input_symbols:
                next_current_states = target_nfa._get_next_current_states(
                    current_states, input_symbol)
                dfa_transitions[current_state_name][input_symbol] = get_name(next_current_states)
                state_queue.append(next_current_states)

        return cls(
            states=dfa_states, input_symbols=dfa_symbols,
            transitions=dfa_transitions, initial_state=dfa_initial_state,
            final_states=dfa_final_states)


    def get_diagram_dot_code(self) -> str:
        """
            Creates the graph associated with this DFA
        """
        G = pygraphviz.AGraph(directed=True, rankdir="LR")

        for state in sorted(self.states):
            G.add_node(state, shape="doublecircle" if state in self.final_states else "circle")

        # Draw arrow into starting state
        G.add_node("~", shape="point", width=0)
        G.add_edge("~", self.initial_state)

        for from_state, lookup in self.transitions.items():
            for input_symbol, to_state in lookup.items():
                if G.has_edge(from_state, to_state):
                    G.get_edge(from_state, to_state).attr["label"] += ',' + input_symbol
                else:
                    G.add_edge(from_state, to_state, label=input_symbol)
        G.layout(prog="dot")

        return G.string()


    def generate_html_description(self) -> str:
        '''
        Generates an HTML-based description of the input DFA. Assumes the alphabet is {'0', '1'}.
        @return string
            Description of the input DFA, including a transition table, its accepting states, and its start state.
        '''
        table_header = """
        <table border=\"1px solid black\" style=\"width:30%;text-align:center\">
            <tr>
                <th></th>
                <th>'0'</th>
                <th>'1'</th>
            </tr>
        """
        table_footer = '</table>'

        dfa_description_list = ['The transition table is as follows: ', table_header]
        for state in self.states:
            dfa_description_list.append('<tr>')
            for entry in [f'State {str(state)}', str(self.transitions[state]['0']), str(self.transitions[state]['1'])]:
                dfa_description_list.append(f'<td>{entry}</td>')
            dfa_description_list.append('</tr>')
        dfa_description_list.append(table_footer)

        accepting_state_string = ', '.join(str(state) for state in sorted(self.final_states))
        dfa_description_list.append(f'The set of accepting states is ${{{accepting_state_string}}}$.<br>')
        dfa_description_list.append(f'The start state is state ${str(self.initial_state)}$.')
        return ''.join(dfa_description_list)

    @classmethod
    def generate_random_dfa(cls: Type[DFA], lower: int, upper: int) -> DFA:
        '''
        Generates a random DFA with n states, where lower <= n <= upper, over the alphabet {0, 1}.
        @param lower
            minimum number of states
        @param upper
            maximum number of states
        @return DFA object
            Randomly generated DFA.
        '''
        # TODO: Use rejection sampling instead of creating a Hamiltonian path from the start state.

        n = randint(lower, upper)
        input_symbols = {'0', '1'}
        states = set(range(n))
        random_state_ordering = list(states)
        shuffle(random_state_ordering)

        initial_state = random_state_ordering[0]
        transitions: DFATransitionsT = {i: dict() for i in range(n)}

        # Fill in transitions so that start state can reach all other states.
        for i in range(n - 1):
            symbol = str(randint(0, 1))
            origin_state, destination_state = random_state_ordering[i], random_state_ordering[i + 1]
            transitions[origin_state][symbol] = destination_state

        # Fill in remaining transitions.
        for transition, symbol in product(transitions.values(), input_symbols):
            if symbol not in transition:
                transition[symbol] = randint(0, n - 1)

        # Generate accepting states.
        final_states = {
            state
            for state in states
            if randint(0, 1)
        }

        # Enforce at least one accepting state and at least one rejecting state.
        if len(final_states) == 0:
            final_states.add(randint(0, n - 1))

        if len(final_states) == n:
            final_states.remove(randint(0, n - 1))

        return cls(states=states, input_symbols=input_symbols, transitions=transitions,
                   initial_state=initial_state, final_states=final_states)

    @classmethod
    def from_json(cls: Type[DFA], jsonDFA: DFAJsonDict) -> DFA:
        states = su.list_as_set(jsonDFA['states'])
        input_symbols = su.list_as_set(jsonDFA['input_symbols'])
        transitions = jsonDFA['transitions']
        initial_state = jsonDFA['initial_state']
        final_states = su.list_as_set(jsonDFA['final_states'])
        return DFA(states=states, input_symbols=input_symbols, transitions=transitions,
                    initial_state=initial_state, final_states=final_states)


    def to_json(self) -> DFAJsonDict:
        state_map = {state: str(i) for i, state in enumerate(self.states)}

        json_states = sorted(state_map[state] for state in self.states)
        json_input_symbols = sorted(self.input_symbols)

        json_transitions = {
            state_map[start_state]: {
                char: state_map[end_state]
                for char, end_state in transition.items()
            }
            for start_state, transition in self.transitions.items()
        }

        json_initial_state = state_map[self.initial_state]
        json_final_states = sorted(state_map[state] for state in self.final_states)

        return {
            'states': json_states,
            'input_symbols': json_input_symbols,
            'transitions': json_transitions,
            'initial_state': json_initial_state,
            'final_states': json_final_states
        }

    def count_words_of_length(self, k: int) -> int:
        """
        Counts words of length `k` accepted by the DFA
        """
        self._populate_count_cache_up_to_len(k)
        return self._count_cache[self.initial_state][k]

    def _populate_count_cache_up_to_len(self, k: int) -> None:
        """
        Populate count cache up to length k
        """

        while len(self._count_cache[self.initial_state]) <= k:
            i = len(self._count_cache[self.initial_state])

            for state, state_path in self._count_cache.items():
                state_path.append(
                    sum(
                        self._count_cache[suffix_state][i - 1]
                        for suffix_state in self.transitions[state].values()
                    )
                )

    def get_input_path(
        self, input_str: str
    ) -> Tuple[List[Tuple[DFAStateT, DFAStateT, str]], bool]:
        """
        Calculate the path taken by input.

        Args:
            input_str (str): The input string to run on the DFA.

        Returns:
            tuple[list[tuple[DFAStateT, DFAStateT, DFASymbolT], bool]]: A list
            of all transitions taken in each step and a boolean indicating
            whether the DFA accepted the input.

        """

        state_history = list(self.read_input_stepwise(input_str, ignore_rejection=True))
        path = list(zip(state_history, state_history[1:], input_str))

        last_state = state_history[-1] if state_history else self.initial_state
        accepted = last_state in self.final_states

        return path, accepted

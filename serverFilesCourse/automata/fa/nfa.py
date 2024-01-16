#!/usr/bin/env python3
"""Classes and methods for working with nondeterministic finite automata."""

from __future__ import annotations
from typing import Dict, FrozenSet, Iterable, Set, Generator, Type, Tuple, Optional, TypeVar, TypedDict, List, Deque
from itertools import chain
from collections import deque

import copy
import pygraphviz
import random
import math
import string

import networkx as nx
import automata.base.exceptions as exceptions
import automata.fa.fa as fa
import automata.fa.dfa as dfa
import shared_utils as su

NFAStateT = fa.FAStateT

GraphT = Dict[NFAStateT, Set[NFAStateT]]
NFAPathT = Dict[str, Set[NFAStateT]]
NFATransitionsT = Dict[NFAStateT, NFAPathT]
InputPathListT = List[Tuple[NFAStateT, NFAStateT, str]]

class NFAJsonDict(TypedDict):
    "A class with type signatures for the nfa json dict"
    states: List[NFAStateT]
    input_symbols: List[str]
    transitions: Dict[NFAStateT, Dict[str, List[NFAStateT]]]
    initial_state: NFAStateT
    final_states: List[NFAStateT]

class NFA(fa.FA):
    """A nondeterministic finite automaton."""

    __slots__ = ['_lambda_closure_dict']

    transitions: NFATransitionsT
    _lambda_closure_dict: Dict[NFAStateT, Set[NFAStateT]]

    def __init__(self,
                 *,
                 states: Set[NFAStateT],
                 input_symbols: Set[str],
                 transitions: NFATransitionsT,
                 initial_state: NFAStateT,
                 final_states: Set[NFAStateT]) -> None:
        """Initialize a complete NFA."""

        object.__setattr__(self, "states", states.copy())
        object.__setattr__(self, "input_symbols", input_symbols.copy())
        object.__setattr__(self, "transitions", copy.deepcopy(transitions))
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "final_states", final_states.copy())
        self.validate()

        # Precompute lambda closures
        lambda_graph = nx.DiGraph()
        lambda_graph.add_nodes_from(self.states)
        lambda_graph.add_edges_from([
            (start_state, end_state)
            for start_state, transition in self.transitions.items()
            for char, end_states in transition.items()
            for end_state in end_states
            if char == ''
        ])

        object.__setattr__(self, "_lambda_closure_dict", {
            state: nx.descendants(lambda_graph, state) | {state}
            for state in self.states
        })


    def __add__(self, other: NFA) -> NFA:
        """Return the concatenation of this NFA and another NFA."""
        if isinstance(other, NFA):
            return self.concatenate(other)
        else:
            raise NotImplementedError


    def __or__(self, other: NFA) -> NFA:
        """Return the union of this NFA and another NFA."""
        if isinstance(other, NFA):
            return self.union(other)
        else:
            raise NotImplementedError

    def copy(self) -> NFA:
        """Create a deep copy of the automaton."""
        return NFA(
            states = self.states,
            input_symbols = self.input_symbols,
            transitions = self.transitions,
            initial_state = self.initial_state,
            final_states = self.final_states
        )

    @classmethod
    def from_dfa(cls: Type[NFA], dfa: dfa.DFA) -> NFA:
        """Initialize this NFA as one equivalent to the given DFA."""
        nfa_transitions: NFATransitionsT = {
            start_state: {
                input_symbol: {end_state}
                for input_symbol, end_state in paths.items()
            }
            for start_state, paths in dfa.transitions.items()
        }

        return cls(
            states=dfa.states, input_symbols=dfa.input_symbols,
            transitions=nfa_transitions, initial_state=dfa.initial_state,
            final_states=dfa.final_states)

    @classmethod
    def from_string_literal(cls: Type[NFA], literal: str, nfa_input_symbols: Set[str]) -> NFA:
        """Initialize this NFA as one accepting only the string literal."""

        nfa_initial_state = 0
        nfa_final_state = len(literal)

        nfa_transitions: NFATransitionsT = {
            i: {chr: {i+1}}
            for i, chr in enumerate(literal)
        }

        nfa_transitions[nfa_final_state] = dict()

        return cls(
            states=set(range(len(literal)+1)),
            input_symbols=nfa_input_symbols,
            transitions=nfa_transitions,
            initial_state=nfa_initial_state,
            final_states={nfa_final_state})


    @classmethod
    def generate_random_nfa(cls: Type[NFA],
                            states: int,
                            alphabet: str,
                            edge_density: float,
                            epsilon_density: float,
                            accepting: int) -> NFA:

        def int_to_lower(n: int) -> str:
            return chr(n + 97)


        # We shouldn't need random NFAs more than 26 states.
        state_max = 26

        if not (1 <= states < state_max):
            raise ValueError('Cannot request an NFA with more than 26 states')
        elif not (1 <= accepting <= states):
            raise ValueError(f'Cannot have {accepting} accept states in an NFA with {states} states')
        elif not (0 <= edge_density <= 1.0):
            raise ValueError(f'Edge density {edge_density} is not in the range [0.0, 1.0]')
        elif not (0 <= epsilon_density <= 1.0):
            raise ValueError(f'Edge density {epsilon_density} is not in the range [0.0, 1.0]')

        # Pick a number of edges between states and states*sqrt(states) (avoiding dense graphs)
        edges = int(math.ceil(((states ** (1.5) - states) * edge_density) + states))

        nfa = None
        while nfa is None:
            # Generate random graph until we find one with start state reachable to every other state
            G: Optional[nx.Graph] = None
            start = None
            while start is None:
                G = nx.gnm_random_graph(states, edges, directed=True)
                start_candidates = list(range(states))
                random.shuffle(start_candidates)
                for candidate in start_candidates:
                    if len(nx.shortest_path(G, source=candidate)) == states:
                        start = candidate
                        break

            # Assign all the nodes a state in the transitions dict.
            transitions: NFATransitionsT = {
                int_to_lower(v): dict()
                for v in G.nodes
            }

            # Assign symbols to all the edges in the graph, choosing epsilons occasionally.
            for u, v in G.edges:
                symbol = '' if random.random() < epsilon_density else random.choice(alphabet)
                transitions[int_to_lower(u)].setdefault(symbol, set()).add(int_to_lower(v))

            # Create NFA data structures
            nfa_states = set(string.ascii_lowercase[:states])
            input_symbols = set(alphabet)
            initial_state = int_to_lower(start)
            final_states = set(string.ascii_lowercase[(states - accepting):states])

            nfa = cls(states=nfa_states, input_symbols=input_symbols, transitions=transitions,
                       initial_state=initial_state, final_states=final_states)
            equiv_dfa = dfa.DFA.from_nfa(nfa, retain_names=False)

            # Check to see if NFA both accepts and rejects strings, if not start over
            if equiv_dfa.complement().isempty() or equiv_dfa.isempty():
                nfa = None

        return nfa

    def _validate_transition_invalid_symbols(self, start_state: NFAStateT, paths: NFAPathT) -> None:
        """Raise an error if transition symbols are invalid."""
        for input_symbol in paths.keys():
            if input_symbol not in self.input_symbols and input_symbol != '':
                raise exceptions.InvalidSymbolError(
                        f'state {start_state} has invalid transition symbol {input_symbol}'
                    )

    def _validate_transition_end_states(self, start_state: NFAStateT, paths: NFAPathT) -> None:
        """Raise an error if transition end states are invalid."""
        for end_states in paths.values():
            for end_state in end_states:
                if end_state not in self.states:
                    raise exceptions.InvalidStateError(
                        f'end state {end_state} for transition on {start_state} is not valid'
                    )

    def validate(self) -> None:
        """Raise exception if this NFA is not internally consistent."""
        self._validate_input_symbols()
        self._validate_transition_start_states()
        for start_state, paths in self.transitions.items():
            self._validate_transition_invalid_symbols(start_state, paths)
            self._validate_transition_end_states(start_state, paths)
        self._validate_initial_state()
        self._validate_initial_state_transitions()
        self._validate_final_states()

    def read_input_from_state(self, state: NFAStateT, input: str) -> Set[NFAStateT]:
        """
        Read in an input from a valid state of the NFA

        Return the resulting state after reading the input.
        """

        new_nfa = NFA(
            states=self.states,
            input_symbols=self.input_symbols,
            transitions=self.transitions,
            initial_state=state,
            final_states=self.states
        )

        try:
            return new_nfa.read_input(input)
        except exceptions.RejectionException:
            return set()

    def _get_lambda_closure(self, start_state: NFAStateT) -> Set[NFAStateT]:
        """
        Return the lambda closure for the given state.

        The lambda closure of a state q is the set containing q, along with
        every state that can be reached from q by following only lambda
        transitions.
        """
        return self._lambda_closure_dict[start_state]

    def _get_next_current_states(self, current_states: Iterable[NFAStateT], input_symbol: str) -> FrozenSet[NFAStateT]:
        """Return the next set of current states given the current set."""
        return frozenset().union(*(
            self._lambda_closure_dict[end_state]
            for current_state in current_states
            for end_state in self.transitions[current_state].get(input_symbol, set())
        ))

    def _check_for_input_rejection(self, current_states: Set[NFAStateT]) -> None:
        """Raise an error if the given config indicates rejected input."""
        if not (current_states & self.final_states):
            state_str = ', '.join(str(state) for state in current_states)
            raise exceptions.RejectionException(
                f"the NFA stopped on all non-final states ({state_str})")

    def read_input_stepwise(self, input_str: str) -> Generator[Set[NFAStateT], None, None]:
        """
        Check if the given string is accepted by this NFA.

        Yield the current configuration of the NFA at each step.
        """
        current_states = self._lambda_closure_dict[self.initial_state]

        yield current_states
        for input_symbol in input_str:
            current_states = set(self._get_next_current_states(
                current_states, input_symbol))
            yield current_states

        self._check_for_input_rejection(current_states)

    @staticmethod
    def _load_new_transition_dict(state_map_dict: Dict[NFAStateT, NFAStateT],
                                 old_transition_dict: NFATransitionsT,
                                 new_transition_dict: NFATransitionsT) -> None:
        """
        Load the new_transition_dict with the old transitions corresponding to
        the given state_map_dict.
        """

        for state_a, transitions in old_transition_dict.items():
            for symbol, states in transitions.items():
                new_transition_dict[state_map_dict[state_a]][symbol] = {
                    state_map_dict[state_b] for state_b in states
                }


    @staticmethod
    def _add_new_state(state_set: Set[NFAStateT]) -> NFAStateT:
        "Adds new state to the state set and returns it"
        new_state = 0
        while new_state in state_set:
            new_state += 1

        state_set.add(new_state)

        return new_state

    @staticmethod
    def _get_state_maps(state_set_a: Set[NFAStateT],
                        state_set_b: Set[NFAStateT]) -> Tuple[Dict[NFAStateT, int], Dict[NFAStateT, int]]:
        """
        Generate state map dicts from given sets. Useful when the state set has
        to be a union of the state sets of component FAs.
        """

        state_map_a: Dict[NFAStateT, int] = {
            state: i
            for i, state in enumerate(state_set_a)
        }

        state_map_b: Dict[NFAStateT, int] = {
            state: i
            for i, state in enumerate(state_set_b, start=len(state_map_a))
        }

        return (state_map_a, state_map_b)

    def union(self, other: NFA) -> NFA:
        """
        Given two NFAs, M1 and M2, which accept the languages
        L1 and L2 respectively, returns an NFA which accepts
        the language L1 union L2.
        """

        # Get state maps
        state_map_a, state_map_b = NFA._get_state_maps(self.states, other.states)

        new_states = set(state_map_a.values()) | set(state_map_b.values())

        # Get new initial state
        new_initial_state = NFA._add_new_state(new_states)

        new_transitions: NFATransitionsT = {
            state: dict()
            for state in new_states
        }

        # Transitions of self
        NFA._load_new_transition_dict(state_map_a, self.transitions, new_transitions)
        # Transitions of other
        NFA._load_new_transition_dict(state_map_b, other.transitions, new_transitions)

        # Add epsilon transitions from new start state to old ones
        new_transitions[new_initial_state][''] = {
            state_map_a[self.initial_state], state_map_b[other.initial_state]
        }

        new_final_states = set(chain(
            (state_map_a[state] for state in self.final_states),
            (state_map_b[state] for state in other.final_states)
        ))

        return NFA(
            states=new_states,
            input_symbols=self.input_symbols | other.input_symbols,
            transitions=new_transitions,
            initial_state=new_initial_state,
            final_states=new_final_states
        )



    def concatenate(self, other: NFA) -> NFA:
        """
        Given two NFAs, M1 and M2, which accept the languages
        L1 and L2 respectively, returns an NFA which accepts
        the languages L1 concatenated with L2.
        """

        # Get state maps
        state_map_a, state_map_b = NFA._get_state_maps(self.states, other.states)

        new_states = set(state_map_a.values()) | set(state_map_b.values())

        new_transitions: NFATransitionsT = {
            state: dict()
            for state in new_states
        }

        # Transitions of self
        NFA._load_new_transition_dict(state_map_a, self.transitions, new_transitions)
        # Transitions of other
        NFA._load_new_transition_dict(state_map_b, other.transitions, new_transitions)


        # Transitions from self to other
        for state in self.final_states:
            new_transitions[state_map_a[state]].setdefault('', set()).add(state_map_b[other.initial_state])

        # Final states of other
        new_final_states = {state_map_b[state] for state in other.final_states}

        return NFA(
            states=new_states,
            input_symbols=self.input_symbols | other.input_symbols,
            transitions=new_transitions,
            initial_state=state_map_a[self.initial_state],
            final_states=new_final_states
        )


    def kleene_star(self) -> NFA:
        """
        Given an NFA which accepts the language L returns
        an NFA which accepts L repeated 0 or more times.
        """
        new_states = set(self.states)

        # Add new initial state
        new_initial_state = NFA._add_new_state(new_states)

        # Transitions are the same with a few additions.
        new_transitions = copy.deepcopy(self.transitions)
        # We add epsilon transition from new initial state
        # to old initial state.
        new_transitions[new_initial_state] = {
            '': {self.initial_state}
        }

        # For each final state in original NFA we add epsilon
        # transition to the old initial state
        for state in self.final_states:
            new_transitions[state].setdefault('', set()).add(self.initial_state)

        return NFA(
            states=new_states,
            input_symbols=self.input_symbols,
            transitions=new_transitions,
            initial_state=new_initial_state,
            final_states=self.final_states | {new_initial_state}
        )

    def get_diagram_dot_code(self, *, states_to_highlight: Optional[Set[NFAStateT]] = None,
            transitions_to_highlight: Optional[Set[Tuple[NFAStateT, str, NFAStateT]]] = None) -> str:
        """
        Creates the graph associated with this NFA
        """

        G = pygraphviz.AGraph(directed=True, rankdir="LR")

        ElemT = TypeVar("ElemT")

        def get_color(elem: ElemT, elem_set: Optional[Set[ElemT]]) -> str:
            return "red" if (elem_set and elem in elem_set) else "black"

        for state in sorted(self.states):
            state_color = get_color(state, states_to_highlight)

            # TODO figure out how to  change label to switch to math font
            G.add_node(state,
                shape="doublecircle" if state in self.final_states else "circle",
                color=state_color,
                fontcolor=state_color)

        # Draw arrow into starting state
        G.add_node("~", shape="point", width=0)
        G.add_edge("~", self.initial_state)

        for from_state, lookup in self.transitions.items():
            for input_symbol, to_state_dict in lookup.items():
                display_input_symbol = su.replace_empty(input_symbol)
                for to_state in to_state_dict:
                    if G.has_edge(from_state, to_state):
                        G.get_edge(from_state, to_state).attr["label"] += ',' + display_input_symbol
                    else:
                        transition_tuple = (from_state, input_symbol, to_state)
                        edge_color = get_color(transition_tuple, transitions_to_highlight)

                        G.add_edge(from_state, to_state,
                            label=display_input_symbol,
                            color=edge_color,
                            fontcolor=edge_color)
        G.layout(prog="dot")

        return G.string()

    def reverse(self) -> NFA:
        """
        Given an NFA which accepts the language L this function
        returns an NFA which accepts the reverse of L.
        """
        new_states = set(self.states)

        new_initial_state = NFA._add_new_state(new_states)

        # Transitions are the same except reversed
        new_transitions: NFATransitionsT = dict()
        for state in new_states:
            new_transitions[state] = dict()
        for state_a, transitions in self.transitions.items():
            for symbol, states in transitions.items():
                for state_b in states:
                    new_transitions[state_b].setdefault(symbol, set()).add(state_a)

        # And we additionally have epsilon transitions from
        # new initial state to each old final state.
        new_transitions[new_initial_state][''] = self.final_states

        new_final_states = {self.initial_state}

        return NFA(
            states=new_states,
            input_symbols=self.input_symbols,
            transitions=new_transitions,
            initial_state=new_initial_state,
            final_states=new_final_states
        )

    def get_reachable_states(self) -> Set[NFAStateT]:
        """Compute the states which are reachable from the initial state."""
        G = nx.DiGraph([
            (start_state, end_state)
            for start_state, transition in self.transitions.items()
            for end_states in transition.values()
            for end_state in end_states
        ])

        G.add_nodes_from(self.states)

        return nx.descendants(G, self.initial_state) | {self.initial_state}

    @classmethod
    def from_json(cls: Type[NFA], json_nfa: NFAJsonDict) -> NFA:
        states = su.list_as_set(json_nfa['states'])
        input_symbols = su.list_as_set(json_nfa['input_symbols'])

        # Check for no duplicate states
        json_transitions = json_nfa['transitions']
        transitions: NFATransitionsT = dict()

        for start_state, transition in json_transitions.items():
            transitions[start_state] = {
                char: su.list_as_set(end_states)
                for char, end_states in transition.items()
            }

        initial_state = json_nfa['initial_state']
        final_states = su.list_as_set(json_nfa['final_states'])
        return NFA(states=states, input_symbols=input_symbols, transitions=transitions,
                    initial_state=initial_state, final_states=final_states)


    def to_json(self) -> NFAJsonDict:
        state_map = {state: str(i) for i, state in enumerate(self.states)}

        json_states = sorted(state_map[state] for state in self.states)
        json_input_symbols = sorted(self.input_symbols)

        json_transitions = {
            state_map[start_state]: {
                char: sorted(state_map[end_state] for end_state in end_states)
                for char, end_states in transition.items()
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

    def get_input_path(
        self, input_str: str
    ) -> Tuple[List[Tuple[NFAStateT, NFAStateT, str]], bool]:
        """
        Get best input path. A path is better if (with priority):

        1. It is an accepting path (ends in a final state)
        2. It reads more of the input (if the input is not accepted we
            select the path such that we can stay on the nfa the longest)
        3. It has the fewest jumps (uses less lambda symbols)

        Returns a tuple of:
        1. the path taken
        2. whether the path was accepting
        """

        visited = set()
        work_queue: Deque[Tuple[InputPathListT, NFAStateT, str]] = deque(
            [([], self.initial_state, input_str)]
        )

        last_non_accepting_input: InputPathListT = []
        least_input_remaining = input_str

        while work_queue:
            visited_states, curr_state, remaining_input = work_queue.popleft()

            # First final state we hit is the best according to desired criteria
            if curr_state in self.final_states and not remaining_input:
                return visited_states, True

            # Otherwise, update longest non-accepting input
            if len(remaining_input) < len(least_input_remaining):
                least_input_remaining = remaining_input
                last_non_accepting_input = visited_states

            # First, get next states that result from reading from input
            if remaining_input:
                next_symbol = remaining_input[0]
                rest = remaining_input[1:] if remaining_input else ""

                next_states_from_symbol = self.transitions[curr_state].get(
                    next_symbol, set()
                )

                for next_state in next_states_from_symbol:
                    if (next_state, rest) not in visited:
                        next_visited_states = visited_states.copy()
                        next_visited_states.append(
                            (curr_state, next_state, next_symbol)
                        )
                        visited.add((next_state, rest))
                        work_queue.append((next_visited_states, next_state, rest))

            # Next, get next states resulting from lambda transition
            next_states_from_lambda = self.transitions[curr_state].get("", set())

            for next_state in next_states_from_lambda:
                if (next_state, remaining_input) not in visited:
                    next_visited_states = visited_states.copy()
                    next_visited_states.append((curr_state, next_state, ""))
                    visited.add((next_state, remaining_input))
                    work_queue.append(
                        (next_visited_states, next_state, remaining_input)
                    )

        return last_non_accepting_input, False

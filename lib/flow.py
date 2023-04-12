# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import itertools
from heapq import heapify, heappop, heappush
from typing import Iterable, List, Set, Tuple, Union


class _Transition:
    def __init__(self, dependencies) -> None:
        # TODO check should be tuple
        self._dependencies = dependencies

    @property
    def dependencies(self):
        return self._dependencies

    def get_dependencies_from_session(self, session):
        return tuple(session[idx] for idx in self._dependencies)

    def __call__(self, session, inputs):
        raise NotImplementedError


class _InputExtraction(_Transition):
    def __init__(self, name) -> None:
        super().__init__(())
        self._name = name

    def __call__(self, _, inputs):
        return inputs[self._name]


class _ConstantExtraction(_Transition):
    def __init__(self, value) -> None:
        super().__init__(())
        self._value = value

    def __call__(self, _s, _i):
        return self._value


class _TupleOutputExtraction(_Transition):
    def __init__(self, output_index, tuple_index) -> None:
        super().__init__((output_index,))
        self._tuple_index = tuple_index

    def __call__(self, session, _):
        # TODO check index bounds
        return self.get_dependencies_from_session(session)[0][self._tuple_index]


class _FunctionCall(_Transition):
    def __init__(
        self,
        function,
        *args,
        **kwargs,
    ) -> None:
        # TODO check arguments
        self._function = function
        ordered_keys = tuple(kwargs.keys())
        dependencies = tuple(args) + tuple(kwargs[key] for key in ordered_keys)

        self._n_args = len(args)
        self._key_to_index = {
            name: self._n_args + i for i, name in enumerate(ordered_keys)
        }
        self._signature = inspect.signature(function)

        # test bind
        self._signature.bind(*args, **kwargs)

        super().__init__(dependencies)

    def args(self, dependencies):
        return dependencies[: self._n_args]

    def kwargs(self, dependencies):
        return {name: dependencies[idx] for name, idx in self._key_to_index.items()}

    def __call__(self, session, _):
        dependency_values = self.get_dependencies_from_session(session)

        args = self.args(dependency_values)
        kwargs = self.kwargs(dependency_values)

        arguments = self._signature.bind(*args, **kwargs)
        arguments.apply_defaults()

        return self._function(*arguments.args, **arguments.kwargs)


class Flow:
    class Constant:  # noqa: B903
        def __init__(self, value) -> None:
            self.value = value

    def __init__(self, *inputs: Tuple[str]) -> None:
        # TODO check redundant names
        # TODO check no "outputs" names
        # TODO should not be empty
        self._inputs = inputs
        self._name_to_index = {}
        self._index_to_name = {}
        self._transitions = []
        self._flow_signature = inspect.Signature(
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
            )
            for name in self._inputs
        )

        for name in self._inputs:
            self._add_transition(_InputExtraction(name), name)

    @property
    def inputs(self):
        return self._inputs

    @property
    def names(self):
        return tuple(self._index_to_name.values())

    def index_of(self, name):
        # TODO check if name exist
        if isinstance(name, str):
            return self._name_to_index[name]
        elif isinstance(name, Flow.Constant):
            return self._add_transition(_ConstantExtraction(name.value))
        raise RuntimeError(f"cannot handle name of type {type(name)}")

    def _add_transition(self, transition, name=None):
        # TODO check if name doesn't already exist
        index = len(self._transitions)
        self._transitions.append(transition)
        if name:
            self._name_to_index[name] = index
            self._index_to_name[index] = name
        return index

    def define_transition(
        self,
        names: Union[str, Tuple[str]],
        function,
        *args,
        **kwargs,
    ):
        # TODO check names
        args = tuple(self.index_of(name) for name in args)
        kwargs = {param: self.index_of(name) for param, name in kwargs.items()}

        transition = _FunctionCall(function, *args, **kwargs)

        if isinstance(names, str):
            index = self._add_transition(transition, name=names)
        else:
            index = self._add_transition(transition, name=None)
            for i, name in enumerate(names):
                self._add_transition(_TupleOutputExtraction(index, i), name=name)

    def get_tape(self, outputs):
        if isinstance(outputs, str):
            outputs = (outputs,)

        tape = []
        max_dependants = {}
        output_indexes = set(  # noqa: C401
            self._name_to_index[name] for name in outputs
        )

        head_indexes = [
            (-self._name_to_index[name], -self._name_to_index[name]) for name in outputs
        ]
        heapify(head_indexes)

        last_index = None
        while len(head_indexes) > 0:
            index, max_dependant = heappop(head_indexes)
            if index == last_index:
                continue
            last_index = index

            index = -index
            if max_dependant is not None:
                max_dependant = -max_dependant
                if index not in output_indexes:
                    max_dependants.setdefault(max_dependant, []).append(index)

            transition = self._transitions[index]
            for idx in transition.dependencies:
                heappush(head_indexes, (-idx, -index))

            tape.append(index)

        for i, index in enumerate(tape):
            tape[i] = (index, max_dependants.get(index, ()))

        return tuple(tape[::-1])

    def flow_from_tape(self, tape, output_indexes, inputs):
        session = [None] * len(self._transitions)
        for index, to_clean in tape:
            session[index] = self._transitions[index](session, inputs)
            for i in to_clean:
                session[i] = None

        if isinstance(output_indexes, int):
            return session[output_indexes]
        return tuple(session[index] for index in output_indexes)

    def names_to_indexes(self, names):
        if isinstance(names, str):
            return self._name_to_index[names]
        return tuple(self._name_to_index[name] for name in names)

    def inputs_as_dict(self, *args, **kwargs):
        return self._flow_signature.bind(*args, **kwargs).arguments

    def flow(self, outputs, *inputs_args, **inputs_kwargs):
        inputs = self.inputs_as_dict(*inputs_args, **inputs_kwargs)
        tape = self.get_tape(outputs)
        output_indexes = self.names_to_indexes(outputs)
        return self.flow_from_tape(tape, output_indexes, inputs)

    __call__ = flow

    def with_outputs(self, outputs):
        return FixedOutputFlow(self, outputs)

    def tape_as_pseudocode(self, tape):
        instructions = []
        for index, to_clean in tape:
            name = self._index_to_name.get(index, "@")
            transition = self._transitions[index]
            if isinstance(transition, _FunctionCall):
                dep = tuple(
                    self._index_to_name.get(index, "@")
                    for index in transition.dependencies
                )
                args = transition.args(dep)
                kwargs = {k: v for k, v in transition.kwargs(dep).items()}
                all_args = itertools.chain(args, kwargs)
                all_args = ",".join(all_args)
                func_name = getattr(
                    transition._function, "__name__", repr(transition._function)
                )
                instructions.append(f"{name} = {func_name}({all_args})")
            elif isinstance(transition, _InputExtraction):
                instructions.append(f"${transition._name}")
            elif isinstance(transition, _TupleOutputExtraction):
                instructions.append(f"{name} = @[{transition._tuple_index}]")

            for i in to_clean:
                name = self._index_to_name.get(i, "@")
                instructions.append(f"delete {name}")

        return "\n".join(instructions)


class FixedOutputFlow:
    def __init__(self, flow, outputs: Union[str, Tuple[str]]) -> None:
        self._flow = flow
        self._outputs = outputs

        self._tape = self._flow.get_tape(outputs)
        self._output_indexes = self._flow.names_to_indexes(self._outputs)

    @property
    def outputs(self):
        return self._outputs

    @property
    def tape(self):
        return self._tape

    @property
    def flow(self):
        return self._flow

    def __call__(self, *args, **kwargs):
        inputs = self._flow.inputs_as_dict(*args, **kwargs)
        return self._flow.flow_from_tape(self._tape, self._output_indexes, inputs)

    def with_outputs(self, outputs):
        return self.flow.with_outputs(outputs)


class AutoForward:
    def __init__(self, flow: Flow, default_outputs: Union[str, Tuple[str]]) -> None:
        self._default_outputs = default_outputs
        self._flow = flow
        self._forward_flow = None

    @property
    def default_outputs(self):
        return self._default_outputs

    @property
    def flow(self):
        return self._flow

    def forward_flow(self, outputs: Union[str, Tuple[str]], *args, **kwargs):
        return self._flow(outputs, *args, **kwargs)

    def forward(self, *args, **kwargs):
        if self._forward_flow is None:
            self._forward_flow = self._flow.with_outputs(self._default_outputs)
        return self._forward_flow(*args, **kwargs)


class ConditionalReturn:
    """Structure that helps a function to determine what output(s) to return and when."""

    def __init__(
        self,
        required_variables: Union[Iterable[str], str],
        valid_variables: Union[Set[str], None],
        from_locals: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        required_variables : Union[Iterable[str], str]
            List of outputs required for return.
        valid_variables : Union[Set[str], None]
            Total list of valid outputs to require.
        from_locals : bool, optional
            Automatically gather variables from stack frames, by default False
        """
        self._single_return = isinstance(required_variables, str)
        required_variables = ConditionalReturn._as_iterable(required_variables)

        # TODO(Pierre) change to exceptions
        assert len(valid_variables) > 0
        assert len(required_variables) > 0
        assert len(valid_variables) == len(set(valid_variables))

        self._valid_variables = valid_variables
        self._required_variables = required_variables
        self._required_variables_left = set(required_variables)

        self._values = {var: None for var in required_variables}
        self._from_locals = from_locals

    @staticmethod
    def split(
        required_variables: Union[Iterable[str], str],
        valid_variables: Set[str],
    ) -> Tuple[List[str], List[str]]:
        """Split into required variables found in valid variable, and those which are not.

        Parameters
        ----------
        required_variables : Union[Iterable[str], str]
            Set of variable names to split.
        valid_variables : Set[str]
            Set of variable name that are considered valid in the current flow.

        Returns
        -------
        Tuple[List[str], List[str]]
            Both set of valid variable names and invalid ones.
        """
        required_variables = ConditionalReturn._as_iterable(required_variables)
        mine = [var for var in required_variables if var in valid_variables]
        other = [var for var in required_variables if var not in valid_variables]
        return mine, other

    @staticmethod
    def _as_iterable(el: Union[Iterable[str], str]):
        if isinstance(el, str):
            return (el,)
        return el

    def should_return(self) -> bool:
        """Determine if all required outputs are ready to be returned.

        Returns
        -------
        bool
            Ready or not to return.
        """
        return len(self._required_variables_left) == 0

    def _get_stack_frame_locals(self, depth=1):
        if not self._from_locals:
            return {}

        calling_fn_frame = inspect.currentframe()

        try:
            for _ in range(depth):
                if calling_fn_frame.f_back is not None:
                    calling_fn_frame = calling_fn_frame.f_back
                else:
                    raise RuntimeError(f"couldn't find frame at depth {depth}")
            frame_locals = calling_fn_frame.f_locals
        finally:
            del calling_fn_frame  # to avoid reference loop

        return frame_locals

    def _gather(self, calling_fn_locals, **local_mapping):
        for var in tuple(self._required_variables_left):
            if var in local_mapping:
                self._values[var] = local_mapping[var]
                self._required_variables_left.remove(var)
            elif var in calling_fn_locals:
                self._values[var] = calling_fn_locals[var]
                self._required_variables_left.remove(var)

    def gather(self, **local_mapping):
        """Gather provided outputs or find them in the caller's stack frame's locals."""
        self._gather(
            self._get_stack_frame_locals(depth=2),
            **local_mapping,
        )

    def gathered(self, **local_mapping) -> bool:
        """Call `gather` and returns `should_return`."""
        self._gather(
            self._get_stack_frame_locals(depth=2),
            **local_mapping,
        )
        return self.should_return()

    def return_value(self, **local_mapping):
        """Returns gathered outputs."""
        if not self.should_return():
            self._gather(
                self._get_stack_frame_locals(depth=2),
                **local_mapping,
            )

        assert self.should_return()

        values = tuple(self._values[var] for var in self._required_variables)

        if self._single_return:
            return values[0]
        return values

    def requires_either_one_of(self, *names):
        """Return if any variable name is required by the conditional return."""
        assert len(names) > 0
        for name in names:
            if name in self._required_variables:
                return True
        return False

    def subcall(self, fn, **names):
        n = len(names)
        names_to_index = {name: i for i, name in enumerate(names.keys())}
        outputs = tuple(
            name
            for name, dependants in names.items()
            if self.requires_either_one_of(*dependants)
        )

        def wrapped_fn(*args, **kwargs):
            results = fn(*args, outputs=outputs, **kwargs)

            normalized_result = [None] * n
            for i, name in enumerate(outputs):
                normalized_result[names_to_index[name]] = results[i]

            return tuple(normalized_result)

        return wrapped_fn

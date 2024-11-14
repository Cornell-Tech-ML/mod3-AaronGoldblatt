from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Set, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    f_inputs = list(vals)
    f_inputs[arg] += epsilon
    f_plus_epsilon = f(*f_inputs)
    # -2 because need to undo the prior addition of epsilon to the arg and then do the subtraction of epsilon
    f_inputs[arg] -= 2 * epsilon
    f_minus_epsilon = f(*f_inputs)
    return (f_plus_epsilon - f_minus_epsilon) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable by adding `x` to the current derivative.

        Args:
        ----
            x: The derivative to be accumulated.

        Returns:
        -------
            None

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier of the variable.

        Args:
        ----
            None: The function does not take any arguments.

        Returns:
        -------
            int: The unique identifier.

        """
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node in the computation graph.

        Args:
        ----
            None: The function does not take any arguments.

        Returns:
        -------
            bool: True if the variable is a leaf node, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is a constant, i.e. it has no `history`.

        Args:
        ----
            None: The function does not take any arguments.

        Returns:
        -------
            bool: True if the variable is a constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables in the computation graph. The parent variables are the variables that were used to create this variable in this variable's `history`.

        Args:
        ----
            None: The function does not take any arguments.

        Returns:
        -------
            Iterable["Variable"]: An iterable of parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients.

        Args:
        ----
            d_output: The gradient of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing
            parent variables and their corresponding gradients.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited: Set[int] = set()
    topological_order: Deque[Variable] = deque()

    def visit(v: Variable) -> None:
        """Recursively visit all the parents of the current variable and add them to the topological sort. Helper function for topological_sort.

        Args:
        ----
            v: The current variable being visited.

        Returns:
        -------
            None. The function modifies the `visited` set in place.

        """
        if not (v.unique_id in visited or v.is_constant()):
            visited.add(v.unique_id)
            for parent in v.parents:
                visit(parent)
            topological_order.appendleft(v)

    visit(variable)
    return topological_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    topological_order: Iterable[Variable] = topological_sort(variable)
    derivative_dict: Dict[int, Any] = {}
    derivative_dict[variable.unique_id] = deriv
    for v in topological_order:
        deriv = derivative_dict[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(deriv)
        else:
            for parent, parent_d in v.chain_rule(deriv):
                if not parent.is_constant():
                    derivative_dict.setdefault(parent.unique_id, 0)
                    derivative_dict[parent.unique_id] += parent_d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors.

        Args:
        ----
            None: The function does not take any arguments.

        Returns:
        -------
            Tuple[Any, ...]: The saved tensors.

        """
        return self.saved_values

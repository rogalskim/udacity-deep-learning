class ParamRange:
    def __init__(self, minimum: float, maximum: float, step: float) -> None:
        assert maximum >= minimum
        assert step > 0
        self.min = minimum
        self.max = maximum
        self.step = step
        self.current_value = None

    def __next__(self) -> float:
        if self.current_value is None:
            self.current_value = self.min
            return self.current_value

        next_value = self.current_value + self.step
        if next_value <= self.max:
            self.current_value = next_value
            return self.current_value
        else:
            raise StopIteration

    def __iter__(self) -> "ParamRange":
        self.current_value = None
        return self


class HyperparameterGenerator:
    def __init__(self,
                 iteration_range: ParamRange,
                 hidden_node_range: ParamRange,
                 learning_rate_range: ParamRange) -> None:
        self.iteration_range = iteration_range
        self.hidden_node_range = hidden_node_range
        self.learning_rate_range = learning_rate_range

    def get_parameters(self) -> dict:
        for iteration_count in self.iteration_range:
            for hidden_node_count in self.hidden_node_range:
                for learning_rate in self.learning_rate_range:
                    yield {"iterations": int(iteration_count),
                           "hidden nodes": int(hidden_node_count),
                           "learning rate": learning_rate}

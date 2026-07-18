def _variable_name(i):
    '''Counts on base 26 and converts to letters'''
    digits = []
    n = i
    while True:
        n, d = divmod(n, 26)
        digits.append(d)
        if n == 0:
            break
    return "".join(chr(ord("a") + d) for d in reversed(digits))


class Vocabulary:
    def __init__(
        self,
        control_verbs,
        num_variables,
        value_range,
        delimiter,
        specials=None,
        has_delimiter=True,
    ):
        self.control_verbs = list(control_verbs)
        self.num_variables = num_variables
        self.value_range = tuple(value_range)
        self.delimiter = delimiter
        self.specials = list(specials or [])
        self.has_delimiter = has_delimiter

        variables = [_variable_name(i) for i in range(self.num_variables)]
        self.variables = variables

        min_value, max_value = self.value_range
        values = [str(v) for v in range(min_value, max_value + 1)]

        self.words = self.specials + self.control_verbs + variables + values
        if self.has_delimiter:
            self.words.append(self.delimiter)

    def state(self):
        return {
            "control_verbs": self.control_verbs,
            "num_variables": self.num_variables,
            "value_range": list(self.value_range),
            "delimiter": self.delimiter,
            "specials": self.specials,
            "has_delimiter": self.has_delimiter,
        }

    def matches(self, other_state):
        return self.state() == other_state

    @classmethod
    def from_state(cls, state):
        return cls(**state)

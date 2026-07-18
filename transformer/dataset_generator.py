import argparse
import csv
import json
import random

from transformer.vocab import Vocabulary


class DatasetGenerator:
    def __init__(
        self,
        vocab,
        target_reassignments_range,
        target_readings_range,
        num_distractors_range,
        distractor_reassignments_range,
        distractor_readings_range,
        max_readings_between_sets,
        seed,
    ):
        self.vocab = vocab
        self.target_reassignments_range = tuple(target_reassignments_range)
        self.target_readings_range=tuple(target_readings_range)
        self.num_distractors_range = tuple(num_distractors_range)
        self.distractor_reassignments_range = tuple(distractor_reassignments_range)
        self.distractor_readings_range = tuple(distractor_readings_range)
        self.max_readings_between_sets = max_readings_between_sets
        self.rng = random.Random(seed)

    def _sample_shape(self):
        target_reassignments = self.rng.randint(*self.target_reassignments_range)
        target_readings = self.rng.randint(*self.target_readings_range)
        num_distractors = self.rng.randint(*self.num_distractors_range)
        distractor_reassignments = [
            self.rng.randint(*self.distractor_reassignments_range)
            for _ in range(num_distractors)
        ]
        distractor_readings = [
            self.rng.randint(*self.distractor_readings_range)
            for _ in range(num_distractors)
        ]
        return target_reassignments, target_readings, num_distractors, distractor_reassignments, distractor_readings

    def _sample_variables(self, num_distractors):
        pool = self.rng.sample(self.vocab.variables, num_distractors + 1)
        target, distractors = pool[0], pool[1:]
        return target, distractors
    
    def _build_ops(self, variable_counts):
        remaining = [[var, sets, reads] for var, sets, reads in variable_counts]
        ops = []
        consecutive_reads = 0
        while any(sets or reads for _, sets, reads in remaining):
            candidates = [i for i, (_, sets, reads) in enumerate(remaining) if sets or reads]
            if consecutive_reads >= self.max_readings_between_sets:
                forced = [i for i in candidates if remaining[i][1] > 0]
                if forced:
                    candidates = forced
                else:
                    break

            choice = self.rng.choice(candidates)
            _, sets, reads = remaining[choice]
            options = (["SET"] if sets > 0 else []) + (["GET"] if reads > 0 else [])
            verb = self.rng.choice(options)

            if verb == "SET":
                remaining[choice][1] -= 1
                consecutive_reads = 0
            else:
                remaining[choice][2] -= 1
                consecutive_reads += 1

            ops.append((verb, remaining[choice][0]))
        return ops

    def _format_example(self, ops, target):
        state = {}
        parts = []
        min_value, max_value = self.vocab.value_range
        last_index = len(ops) - 1
        for i, (verb, variable) in enumerate(ops):
            if verb == "SET":
                value = self.rng.randint(min_value, max_value)
                state[variable] = value
                parts.append(f"SET {variable} {value}")
            else:
                if variable not in state:
                    state[variable] = self.rng.randint(min_value, max_value)
                if i == last_index:
                    parts.append(f"GET {variable}")
                else:
                    parts.append(f"GET {variable} {state[variable]}")

        output = state[target]
        text = f"{self.vocab.delimiter} ".join(parts)
        return text, output

    def generate_example(self):
        (
            target_reassignments,
            target_readings,
            num_distractors,
            distractor_reassignments,
            distractor_readings,
        ) = self._sample_shape()
        target, distractors = self._sample_variables(num_distractors)

        variable_counts = [(target, target_reassignments, target_readings)] + list(
            zip(distractors, distractor_reassignments, distractor_readings)
        )

        ops = self._build_ops(variable_counts)
        ops.append(("GET", target))

        return self._format_example(ops, target)

    @classmethod
    def from_config(cls, config, vocab):
        return cls(
            vocab=vocab,
            target_reassignments_range=config["target_reassignments_range"],
            target_readings_range=config["target_readings_range"],
            num_distractors_range=config["num_distractors_range"],
            distractor_reassignments_range=config["distractor_reassignments_range"],
            distractor_readings_range=config["distractor_readings_range"],
            max_readings_between_sets=config["max_readings_between_sets"],
            seed=config["seed"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("-n", "--num-examples", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = Vocabulary.from_state(json.load(f))
    with open(args.config) as f:
        config = json.load(f)

    generator = DatasetGenerator.from_config(config, vocab)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "output"])
        for _ in range(args.num_examples):
            text, output = generator.generate_example()
            writer.writerow([text, output])
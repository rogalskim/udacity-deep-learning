from pathlib import Path
import typing

import pandas as pd


class LogParser:
    def __init__(self) -> None:
        self.data_column_names = ["iterations", "hidden nodes", "learning rate", "training loss", "validation loss"]
        self.__init_data_dict()

    def __init_data_dict(self) -> None:
        self.data_dict = {name: [] for name in self.data_column_names}

    def __parse_file(self, file: typing.TextIO) -> None:
        entry = ""
        for line in file:
            if line.isspace():
                self.__parse_entry(entry)
                entry = ""
                continue
            entry += line

    def __parse_entry(self, entry: str) -> None:
        lines = entry.splitlines(False)
        assert len(lines) >= 4, f"Invalid line count in entry: {entry}"
        hyperparams = self.__extract_hyperparams(lines[1])
        training_loss = self.__extract_value(lines[2])
        validation_loss = self.__extract_value(lines[3])
        values = (*hyperparams, training_loss, validation_loss)
        for column_name, value in zip(self.data_column_names, values):
            self.data_dict[column_name].append(value)

    @staticmethod
    def __extract_hyperparams(line: str) -> (float, float, float):
        segments = line.split(" | ")
        iterations = LogParser.__extract_value(segments[0])
        learn_rate = LogParser.__extract_value(segments[2])
        hidden_nodes = LogParser.__extract_value(segments[1])
        return iterations, learn_rate, hidden_nodes

    @staticmethod
    def __extract_value(line_segment: str) -> float:
        return float(line_segment.split(" ")[-1])

    def __make_data_frame(self) -> pd.DataFrame:
        data = pd.DataFrame({name: pd.Series(self.data_dict[name]) for name in self.data_column_names})
        data.iterations = data.iterations.astype(int)
        data["hidden nodes"] = data["hidden nodes"].astype(int)
        return data

    def parse(self, log_file_path: Path) -> pd.DataFrame:
        assert log_file_path.exists() and log_file_path.is_file()
        self.__init_data_dict()
        log_file = log_file_path.open('r')
        self.__parse_file(log_file)
        log_file.close()
        return self.__make_data_frame()


if __name__ == "__main__":
    log_path = Path("hyperparam_log.txt")
    print(LogParser().parse(log_path))



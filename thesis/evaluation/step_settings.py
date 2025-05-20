import string
from dataclasses import dataclass
from typing import Literal, Dict, Any, List, Tuple

from thesis.evaluation.utils import pick_steps_h_grid, pick_steps_q_ratio


@dataclass
class StepSettings:
    method: Literal["q_ratio", "h_grid"]
    method_params: Dict[str, Any]

    def __post_init__(self):
        # define expected keys and their types per method
        expected_schemas = {
            "q_ratio": {
                "q_ratio": float,
                "allow_duplicate_steps": bool,
            },
            "h_grid": {
                "h_grid": list,
                "q_strictly_descending": bool,
            }
        }

        if self.method not in expected_schemas:
            raise ValueError(f"Unsupported method: {self.method}")

        schema = expected_schemas[self.method]

        # check for missing keys
        missing = schema.keys() - self.method_params.keys()
        if missing:
            raise ValueError(f"Missing keys for method '{self.method}': {missing}")

        # check for type mismatches
        for key, expected_type in schema.items():
            value = self.method_params[key]

            # special case: check that 'h_grid' is a list of ints
            if key == "h_grid":
                if not isinstance(value, list) or not all(isinstance(i, int) for i in value):
                    raise TypeError(
                        f"'h_grid' must be a list of integers, got: {value}"
                    )
            elif not isinstance(value, expected_type):
                raise TypeError(
                    f"Expected '{key}' to be {expected_type.__name__} for method '{self.method}', "
                    f"but got {type(value).__name__}"
                )

    def get_list_param_steps(self, data_dir_path: string) -> List[Tuple[int, List[int]]]:
        if self.method == "q_ratio":
            return pick_steps_q_ratio(
                data_dir_path=data_dir_path,
                q_ratio=self.method_params["q_ratio"],
                allow_duplicate_steps=self.method_params["allow_duplicate_steps"])
        elif self.method == "h_grid":
            return pick_steps_h_grid(
                data_dir_path=data_dir_path,
                h_grid=self.method_params["h_grid"],
                q_strictly_descending=self.method_params["q_strictly_descending"])
        else:
            raise ValueError(f"Unsupported method: {self.method}")


    def to_dirname(self) -> str:
        if self.method == "q_ratio":
            q_ratio = self.method_params["q_ratio"]
            allow_dup = self.method_params["allow_duplicate_steps"]
            dup_str = "with_duplicates" if allow_dup else "no_duplicates"
            return f"q_ratio__{q_ratio}__{dup_str}"

        elif self.method == "h_grid":
            h_grid = self.method_params["h_grid"]
            desc = self.method_params["q_strictly_descending"]

            desc_str = "desc" if desc else "no_desc"
            grid_str = "-".join(map(str, h_grid))

            return f"h_grid__{grid_str}__{desc_str}"

        else:
            raise ValueError(f"Unknown method: {self.method}")

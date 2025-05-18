from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple

from thesis.evaluation.utils import pick_steps_h_grid, pick_steps_q_half


@dataclass
class QscEvaluationParameters:
    method: Literal["q_half", "h_grid"]  # Specifies which method to use
    q_strictly_descending: bool = True
    include_inbetween_steps: bool = False
    h_grid: Optional[List[int]] = None  # Only needed if method is "h_grid"
    data_dir_path: Optional[str] = None

    parameter_associated_steps: List[Tuple[int, List[int]]] = field(init=False)

    def load_parameter_associated_steps(self):
        if self.method == "q_half":
            self.parameter_associated_steps = pick_steps_q_half(
                data_dir_path=self.data_dir_path,
                q_strictly_descending=self.q_strictly_descending,
                include_inbetween_steps=self.include_inbetween_steps
            )
        elif self.method == "h_grid":
            if self.h_grid is None:
                raise ValueError("h_grid must be provided when method is 'h_grid'")
            self.parameter_associated_steps = pick_steps_h_grid(
                data_dir_path=self.data_dir_path,
                h_grid=self.h_grid,
                q_strictly_descending=self.q_strictly_descending,
                include_inbetween_steps=self.include_inbetween_steps
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def set_data_dir_path(self, data_dir_path: str):
        self.data_dir_path = data_dir_path
        self.load_parameter_associated_steps()

    def to_dirname(self) -> str:
        # Extract just the parameter values
        param_values = [str(param) for param, _ in self.parameter_associated_steps]
        steps_str = "-".join(param_values)

        inbetween_str = "with_inbetween" if self.include_inbetween_steps else "without_inbetween"
        desc_str = "with_desc" if self.q_strictly_descending else "without_desc"

        return f"{self.method}__{steps_str}__{inbetween_str}__{desc_str}"
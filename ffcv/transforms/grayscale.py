import numpy as np
from numpy.random import rand
from typing import Callable, Optional, Tuple
from dataclasses import replace
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
import numbers
import numba as nb

class RandomGrayscale(Operation):
    """Apply grayscale transformation with probability grayscale_prob.
    Operates on raw arrays (not tensors).
    Parameters
    ----------
    grayscale_prob : float, The probability with which to apply grayscale transformation.
    """

    def __init__(self, grayscale_prob):
        super().__init__()
        self.grayscale_prob = grayscale_prob
        assert self.grayscale_prob >= 0 and self.grayscale_prob <= 1

    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        grayscale_prob = self.grayscale_prob
            
        def grayscale(img, dst):
            should_grayscale = rand(img.shape[0]) < grayscale_prob
            for i in my_range(img.shape[0]):
                if should_grayscale[i]:
                    dst[i, :, :, 0] = np.clip(0.2989 * img[i, :, :, 0] + 0.5870 * img[i, :, :, 1] + 0.1140 * img[i, :, :, 2], 0, 255)
                    dst[i, :, :, 1] = dst[i, :, :, 0]
                    dst[i, :, :, 2] = dst[i, :, :, 0]
                else:
                    dst[i] = img[i]
            return dst

        grayscale.is_parallel = True
        return grayscale

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(shape=previous_state.shape, dtype=previous_state.dtype))

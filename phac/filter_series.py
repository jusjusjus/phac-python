
from typing import List, Tuple
from collections import namedtuple

import numpy as np

from .frequency_band import FrequencyBand

_FilterSeries: Tuple[float, float, float] = namedtuple("FilterSeries", "min max width")

class FilterSeries(_FilterSeries):

    _resolution = 2

    @property
    def num(self) -> int:
        return 1+self._resolution*(self.max-self.min)/self.width

    @property
    def df(self) -> float:
        return self.width/2

    def __iter__(self):
        for f0 in np.arange(self.min, self.max, self.df/self._resolution):
            yield FrequencyBand(f0-self.df, f0+self.df)

    @classmethod
    def from_equidistant_list(cls, x: List[float]):
        return cls(x[0], x[-1], 2*(x[1]-x[0]))

from collections import namedtuple
from typing import Tuple

_FrequencyBand: Tuple[float, float] = namedtuple(  # type: ignore
    'FrequencyBand', 'left right')


class FrequencyBand(_FrequencyBand):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        cls.validate(instance)
        return instance

    def validate(self) -> None:
        try:
            assert self.left > 0.0, \
                "Left band edge has to be larger than zero %s" % (self,)
            assert self.right > 0.0, \
                "Right band nedge has to be larger than zero %s" % (self,)
            assert self.left < self.right, \
                "Right band edge has to be larger than the left %s" % (self,)
        except AssertionError as err:
            raise ValueError(str(err))

    @property
    def width(self) -> float:
        return self.right-self.left

    @property
    def center(self) -> float:
        return 0.5 * (self.left+self.right)

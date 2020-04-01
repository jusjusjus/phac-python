
import numpy as np

from .util import trapezoid, _hilbert


class SegmentedSignal:
        
    def __init__(self, nsegment: int, noverlap: int, n: int=None,
            arr: np.ndarray=None, dtype=np.float64):
        self.arr = np.zeros(n, dtype=dtype) if arr is None else arr
        self.nsegment = nsegment
        self.noverlap = noverlap
    
    @property
    def size(self) -> int:
        return self.arr.size
            
    @property
    def segment_minus_overlap(self) -> int:
        return self.nsegment-self.noverlap

    @property
    def num_segments(self) -> int:
        return int(np.ceil(self.size/self.segment_minus_overlap))

    def segment(self, n: int) -> np.ndarray:
        m = n*self.segment_minus_overlap
        return self.arr[m:m+self.nsegment]

    def add_to_segment(self, n: int, arr: np.ndarray):
        m = n*self.segment_minus_overlap
        self.arr[m:m+self.nsegment] += arr


def hilbert(arr, nsegment: int=8192, noverlap: int=1024) -> np.ndarray:
    # If size is smaller then segmentation size ..
    if arr.size < nsegment: return _hilbert(arr)

    arr = SegmentedSignal(nsegment, noverlap, arr=arr)
    ans = SegmentedSignal(nsegment, noverlap, n=arr.size, dtype=np.complex)
    trapz = trapezoid(nsegment, noverlap)

    # first segment is not cut in the beginning
    segment = _hilbert(arr.segment(0))
    segment[noverlap:] *= trapz[noverlap:]
    ans.add_to_segment(0, segment)
    # rest of the segments are treated  as regular (This is not 100% correct
    # for the last segment where special cases should be treated.)
    for i in range(1, arr.num_segments):
        transformed = _hilbert(arr.segment(i))
        ans.add_to_segment(i, trapz[:transformed.size]*transformed)   

    return ans.arr

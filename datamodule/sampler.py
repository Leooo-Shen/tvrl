import random

import matplotlib.pyplot as plt
import numpy as np


class PowerFunction:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        return self.a * np.power(x, self.b) + self.c


class CDF:
    def __init__(self, name):
        self.func = self._sampling_function(name)

    def _sampling_function(self, name):
        # increase
        if name == "linear_increase":
            func = PowerFunction(1, 1, 0.5)
        elif name == "quadratic_increase":
            func = PowerFunction(1, 2, 0.5)
        # decrease
        elif name == "linear_decrease":
            func = PowerFunction(-0.5, 1, 1)
        elif name == "quadratic_decrease":
            func = PowerFunction(-0.5, 2, 1)
        elif name == "root_decrease":
            func = PowerFunction(-0.5, 0.5, 1)
        # constant
        elif name == "constant":
            func = PowerFunction(0, 0, 0.5)
        return func

    def _normalize(self, prob):
        min_value = min(prob)
        max_value = max(prob)

        # Normalize the values to the range [0, 1]
        if max_value != min_value:
            prob_zero_one = [(value - min_value) / (max_value - min_value) for value in prob]
        else:
            prob_zero_one = prob
        # Adjust the normalized values to ensure the sum is 1
        sum_prob = sum(prob_zero_one)
        normalized_prob = [value / sum_prob for value in prob_zero_one]
        return normalized_prob

    def __call__(self, x):
        return self._normalize(self.func(x))

    @staticmethod
    def visualize():
        # plot all the functions
        names = [
            "linear_increase",
            "quadratic_increase",
            "linear_decrease",
            "quadratic_decrease",
            "root_decrease",
            "constant",
        ]
        x_axis = np.linspace(0, 50, 50)
        for name in names:
            func = CDF(name)
            y = func(x_axis)
            plt.plot(x_axis, y, label=name)
            plt.legend()


class ClipSamplingWithCDF:
    """
    Sample 2 clips from a video, following a given CDF.
    The sampling is bi-directional: clip2 can be either left or right to clip1.
    Total temporal length = clip_frames * stride

    Each clip has shape (c, t, h, w) or (t, h, w)
    Return a list of ndarray clips and a list of ndarray indices.
    """

    def __init__(self, clip_frames, cdf_type, stride):
        self.clip_frames = clip_frames
        self.cdf = CDF(cdf_type)
        self.stride = stride

        print(f"[*] Sampling 2 clips with {clip_frames} frames {stride} strides and {cdf_type} function")

    def __call__(self, video):
        total_frames = video.shape[1]
        if len(video.shape) == 3:
            total_frames = video.shape[0]
        start_range = total_frames - self.clip_frames * self.stride

        first_clip_start = random.randint(0, start_range)
        first_clip_idx = np.arange(
            first_clip_start,
            first_clip_start + self.clip_frames * self.stride,
            self.stride,
        )

        direction = np.random.choice([-1, 1])
        if direction == 1:
            # sample from the right
            interval_range = np.arange(total_frames - max(first_clip_idx) + 1)
        else:
            # sample from the left
            interval_range = np.arange(first_clip_start + 1)
        p = CDF("linear_decrease")(interval_range)
        interval = np.random.choice(len(interval_range), p=p)
        second_clip_start = first_clip_start + direction * interval
        second_clip_idx = np.arange(
            second_clip_start,
            second_clip_start + self.clip_frames * self.stride,
            self.stride,
        )
        if len(first_clip.shape) == 3:
            first_clip = video[first_clip_idx]
            second_clip = video[second_clip_idx]
        elif len(first_clip.shape) == 4:
            first_clip = video[:, first_clip_idx, ...]
            second_clip = video[:, second_clip_idx, ...]
        else:
            raise ValueError("Invalid video shape")
        return [first_clip, second_clip], [first_clip_idx, second_clip_idx]


class ClipSamplingRandom:
    """
    Sample n clips from a video randomly.
    Total temporal length = clip_frames * stride

    Each clip has shape (c, t, h, w) or (t, h, w)
    Return a list of ndarray clips and a list of ndarray indices.
    """

    def __init__(self, clip_frames, n_clips, stride, make_clips_identical=False):
        self.clip_frames = clip_frames
        self.n_clips = n_clips
        self.stride = stride
        self.make_clips_identical = make_clips_identical
        print(f"[*] Random sampling {n_clips} clips with {clip_frames} frames and stride {stride}")

    def __call__(self, video):
        total_frames = video.shape[1]
        if len(video.shape) == 3:
            total_frames = video.shape[0]

        # return the whole video if it is too short
        if total_frames <= self.clip_frames:
            # return [video] * self.n_clips, [np.arange(total_frames)] * self.n_clips
            return [video], [np.arange(total_frames)]

        start_range = total_frames - self.clip_frames * self.stride
        clips = []
        clip_indices = []

        if self.make_clips_identical:
            clip_start = random.randint(0, start_range)
            clip_idx = np.arange(clip_start, clip_start + self.clip_frames * self.stride, self.stride)
            for _ in range(self.n_clips):
                if len(video.shape) == 3:
                    c = video[clip_idx]
                elif len(video.shape) == 4:
                    c = video[:, clip_idx, ...]
                else:
                    raise ValueError("Invalid video shape")
                clips.append(c)
                clip_indices.append(clip_idx)
        else:
            for _ in range(self.n_clips):
                clip_start = random.randint(0, start_range)
                clip_idx = np.arange(clip_start, clip_start + self.clip_frames * self.stride, self.stride)
                if len(video.shape) == 3:
                    c = video[clip_idx]
                elif len(video.shape) == 4:
                    c = video[:, clip_idx, ...]
                else:
                    raise ValueError("Invalid video shape")
                clips.append(c)
                clip_indices.append(clip_idx)
        return clips, clip_indices


class ClipSamplingDense:
    """
    Sample n clips from a video densely with sliding window.
    Each clip has shape (1, t, h, w) or (t, h, w)

    """

    def __init__(self, window_size, window_step_size, data_sample_stride):
        if window_size < 1:
            window_size = 1
        if window_step_size < 1:
            window_step_size = 1
        self.window_size = window_size
        self.window_step_size = window_step_size
        self.data_sample_stride = data_sample_stride

        print(
            f"[*] Dense sampling with sliding window: size {window_size}, step_size {window_step_size} and data_sample_stride {data_sample_stride}"
        )

    def sliding_window(self, total_frames, window_size, window_step_size, data_sample_stride):
        # return a list of indices from the sliding window results, [[0,1,2], [1,2,3], ...
        data_range = list(range(total_frames))
        results = []
        for i in range(0, len(data_range) - window_size + 1, window_step_size):
            window = [data_range[j] for j in range(i, i + window_size, data_sample_stride)]
            results.append(window)
        return results

    def __call__(self, video):
        total_frames = video.shape[1]
        if len(video.shape) == 3:
            total_frames = video.shape[0]

        # return the whole video if it is too short
        if total_frames < self.window_size:
            return [video], [np.arange(total_frames)]

        clip_indices = self.sliding_window(
            total_frames, self.window_size, self.window_step_size, self.data_sample_stride
        )
        clips = []
        for clip_idx in clip_indices:
            if len(video.shape) == 3:
                c = video[clip_idx]
            elif len(video.shape) == 4:
                c = video[:, clip_idx, ...]
            else:
                raise ValueError("Invalid video shape")
            clips.append(c)

        return clips, clip_indices


class PathSamplingRandom:
    """
    For OCT data with changing length.
    Sample n clips from a list of paths randomly.
    Total temporal length = clip_frames * stride
    """

    def __init__(self, clip_frames, n_clips, stride, make_clips_identical=False):
        self.clip_frames = clip_frames
        self.n_clips = n_clips
        self.stride = stride

        self.make_clips_identical = make_clips_identical
        print(f"[*] Random sampling {n_clips} clips with {clip_frames} frames and stride {stride}")

    def __call__(self, paths):
        assert isinstance(paths, list)
        total_frames = len(paths)
        if total_frames <= self.clip_frames:
            return [paths] * self.n_clips, [np.arange(total_frames)] * self.n_clips

        clip_paths = []
        clip_indices = []
        start_range = total_frames - self.clip_frames * self.stride

        if self.make_clips_identical:
            clip_start = random.randint(0, start_range)
            clip_idx = np.arange(clip_start, clip_start + self.clip_frames * self.stride, self.stride)
            for _ in range(self.n_clips):
                p = [paths[i] for i in clip_idx]
                clip_paths.append(p)
                clip_indices.append(clip_idx)
        else:
            for _ in range(self.n_clips):
                clip_start = random.randint(0, start_range)
                clip_idx = np.arange(clip_start, clip_start + self.clip_frames * self.stride, self.stride)
                p = [paths[i] for i in clip_idx]
                clip_paths.append(p)
                clip_indices.append(clip_idx)
        return clip_paths, clip_indices

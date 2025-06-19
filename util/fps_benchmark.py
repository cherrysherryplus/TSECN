import torch
import time
from typing import Callable


class FPSBenchmark():
    def __init__(
        self,
        model: torch.nn.Module,
        input_size: tuple,
        input_constructor: Callable = None,
        preprocessor: Callable = torch.nn.Identity(), 
        device: str = "cpu",
        warmup_num: int = 5,
        log_interval: int = 10,
        iterations: int = 100,
        repeat_num: int = 1,
    ) -> None:
        """FPS benchmark.

        Ref:
            MMDetection: https://mmdetection.readthedocs.io/en/stable/useful_tools.html#fps-benchmark.

        Args:
            model (torch.nn.Module): model to be tested.
            input_size (tuple): model acceptable input size, e.g. `BCHW`, make sure `batch_size` is 1.
            input_constructor: construct inputs if more than one input is needed
            preprocessor: preprocess inputs with Denoiser or nn.Identity
            device (str): device for test. Default to "cpu".
            warmup_num (int, optional): the first several iterations may be very slow so skip them. Defaults to 5.
            iterations (int, optional): numer of iterations in a single test. Defaults to 100.
            repeat_num (int, optional): number of repeat tests. Defaults to 1.
        """
        # Parameters for `load_model`
        self.model = model
        self.input_size = input_size
        self.input_constructor = input_constructor
        self.preprocessor = preprocessor
        self.device = device

        # Parameters for `measure_inference_speed`
        self.warmup_num = warmup_num
        self.log_interval = log_interval
        self.iterations = iterations

        # Parameters for `repeat_measure_inference_speed`
        self.repeat_num = repeat_num

    def load_model(self):
        model = self.model.to(self.device)
        model.eval()
        return model
    
    def load_preprocessor(self):
        preprocessor = self.preprocessor.to(self.device)
        preprocessor.eval()
        return preprocessor

    def measure_inference_speed(self):
        model = self.load_model()
        preprocessor = self.load_preprocessor()
        pure_inf_time = 0
        fps = 0
        for i in range(self.iterations):
            if self.input_constructor is None:
                input_data = torch.randn(self.input_size, device=self.device).unsqueeze_(0)
            else:
                input_data = self.input_constructor(self.input_size)
            if "cuda" in self.device:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    if isinstance(input_data, dict):
                        input_data.update(input=preprocessor(input_data['input']))
                        _ = model(**input_data)
                    else:
                        input_data = preprocessor(input_data)
                        _ = model(input_data)
                torch.cuda.synchronize()
            elif "cpu" in self.device:
                start_time = time.perf_counter()
                with torch.no_grad():
                    if isinstance(input_data, dict):
                        _ = model(**input_data)
                    else:
                        _ = model(input_data)
            else:
                NotImplementedError(
                    f"{self.device} hasn't been implemented yet."
                )
            elapsed = time.perf_counter() - start_time

            if i >= self.warmup_num:
                pure_inf_time += elapsed
                if (i + 1) % self.log_interval == 0:
                    fps = (i + 1 - self.warmup_num) / pure_inf_time
                    print(
                        f'Done image [{i + 1:0>3}/{self.iterations}], '
                        f'FPS: {fps:.2f} img/s, '
                        f'Times per image: {1000 / fps:.2f} ms/img',
                        flush=True,
                    )
                else:
                    pass
            else:
                pass
        fps = (self.iterations - self.warmup_num) / pure_inf_time
        print(
            f'Overall FPS: {fps:.2f} img/s, '
            f'Times per image: {1000 / fps:.2f} ms/img',
            flush=True,
        )
        return fps

    def repeat_measure_inference_speed(self):
        assert self.repeat_num >= 1
        fps_list = []
        for _ in range(self.repeat_num):
            fps_list.append(self.measure_inference_speed())
        if self.repeat_num > 1:
            fps_list_ = [round(fps, 2) for fps in fps_list]
            times_pre_image_list_ = [round(1000 / fps, 2) for fps in fps_list]
            mean_fps_ = sum(fps_list_) / len(fps_list_)
            mean_times_pre_image_ = sum(times_pre_image_list_) / len(
                times_pre_image_list_)
            print(
                f'Overall FPS: {fps_list_}[{mean_fps_:.2f}] img/s, '
                f'Times per image: '
                f'{times_pre_image_list_}[{mean_times_pre_image_:.2f}] ms/img',
                flush=True,
            )
            return fps_list
        else:
            return fps_list[0]


if __name__ == '__main__':
    FPSBenchmark(
        model=torch.nn.Conv2d(3, 64, 3, 1, 1),
        input_size=(3, 256, 256),
        device="cuda:0",
        repeat_num=2
    ).repeat_measure_inference_speed()

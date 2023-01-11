import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# prof = profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18"))
# prof.start()

model = models.resnet34()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, \
    # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2), \
        profile_memory=True, on_trace_ready=tensorboard_trace_handler("./log/resnet34")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()
        print(prof.step())

# prof.stop()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
# prof.export_chrome_trace("trace.json")

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from model import Generator


net = Generator()
net.load_state_dict(torch.load('./face_paint_512_v2_0.pt', map_location="cpu"))

scripted_model = torch.jit.script(net)

opt_model = optimize_for_mobile(scripted_model)

opt_model._save_for_lite_interpreter("./face_paint_512_v2_0.ptl")
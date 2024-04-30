import gradio
import torch
import modules


def _normalize(u, v):
    uu = torch.mean(u ** 2, dim=list(range(1, u.ndim)), keepdim=True) + 1.0e-05
    vv = torch.mean(v ** 2, dim=list(range(1, v.ndim)), keepdim=True)
    return ((torch.relu(1 - vv) / uu) ** 0.5) * u


class CFGNormalizeForForge(modules.scripts.Script):
    sorting_priority = 114

    def title(self):
        return "CFG Normalize"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with modules.ui_components.InputAccordion(False, label=self.title()) as enabled:
            cfg_divider = gradio.Slider(label='CFG Scale divider', minimum=1.0, maximum=32.0, step=0.1, value=16.0, visible=False)
        return enabled, cfg_divider

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, cfg_divider = script_args

        def sampler_cfg_function(args):
            sigma = args["sigma"]
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]

            cfg_scale = (cond_scale - 1.0) / cfg_divider
            cfg = _normalize(cond - uncond, cond / sigma) * sigma
            return cond + cfg_scale * cfg

        if enabled:
            model = p.sd_model.forge_objects.unet.clone()
            model.set_model_sampler_cfg_function(sampler_cfg_function)
            p.sd_model.forge_objects.unet = model
            p.extra_generation_params.update(dict(cfgnormalize=enabled))

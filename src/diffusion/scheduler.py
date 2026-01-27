import torch 
import os
import sys

# tab-ddpm 패키지 경로 보정
try:
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
except ImportError:
    this_dir = os.path.dirname(__file__)
    tab_ddpm_root = os.path.abspath(os.path.join(this_dir, "..", "..", "tab-ddpm"))
    if tab_ddpm_root not in sys.path:
        sys.path.insert(0, tab_ddpm_root)
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

#  1. forward noising; gaussian_q_sample
# TODO 2. reverse mean/variance; gaussian_p_mean_variance
#  3. reverse sampling; gaussian_p_sample
# 4. training loss ; _gaussian_loss


class TabDDPMGaussianScheduler:
    def __init__(self, num_classes, num_numerical_features, denoise_fn, num_timesteps=1000, 
                 gaussian_loss_type="mse", gaussian_parametrization="eps", scheduler="cosine", device="cuda"):
        self.ddpm = GaussianMultinomialDiffusion(
            num_classes=num_classes,
            num_numerical_features=num_numerical_features,
            denoise_fn=denoise_fn,
            num_timesteps=num_timesteps,
            device=device,
            gaussian_loss_type=gaussian_loss_type,
            gaussian_parametrization=gaussian_parametrization,
            scheduler=scheduler
        )

    def sample_time(self, b, device, method='uniform'):
        """타임스텝 샘플링"""
        return self.ddpm.sample_time(b, device, method)
    
    def gaussian_q_sample(self, x0, t, noise=None):
        """Forward noising: x0 -> xt"""
        return self.ddpm.gaussian_q_sample(x0, t, noise)

    def gaussian_loss(self, model_out, x0, xt, t, noise):
        """Gaussian loss 계산"""
        return self.ddpm._gaussian_loss(model_out, x0, xt, t, noise)

    def gaussian_p_sample(self, model_out, xt, t):
        """Reverse sampling: xt -> x_{t-1}"""
        sample = self.ddpm.gaussian_p_sample(model_out, xt, t)["sample"]
        # float64 -> float32 변환 (dtype 불일치 방지)
        return sample.float() if sample.dtype == torch.float64 else sample
    
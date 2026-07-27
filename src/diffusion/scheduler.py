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

    def predict_xstart(self, model_out, xt, t):
        """eps 예측(model_out)으로부터 깨끗한 좌표 x̂₀ 복원 """
        return self.ddpm._predict_xstart_from_eps(x_t=xt, t=t, eps=model_out)

    def gaussian_p_sample(self, model_out, xt, t, temperature=0.7, project_fn=None):
        """Reverse sampling: xt -> x_{t-1}

        project_fn: (x0_hat, t) -> x0_hat_projected. 주어지면 posterior mean을
        계산하기 전에 예측된 x̂₀를 feasible set(예: 도로점)으로 project한 뒤
        renoise한다 (predict-project-renoise, 도로 스냅 비율 감소 목적).
        """
        out = self.ddpm.gaussian_p_mean_variance(
            model_out,
            xt,
            t,
        )

        mean = out["mean"]
        if project_fn is not None:
            x0_projected = project_fn(out["pred_xstart"], t)
            mean, _, _ = self.ddpm.gaussian_q_posterior_mean_variance(
                x_start=x0_projected, x_t=xt, t=t
            )

        noise = torch.randn_like(xt)
        

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        )

        sample = mean + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise * temperature
        # float64 -> float32 변환 (dtype 불일치 방지)
        return sample.float() if sample.dtype == torch.float64 else sample
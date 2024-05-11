disabled = 'Disabled'
enabled = 'Enabled'
subtle_variation = 'Vary (Subtle)'
strong_variation = 'Vary (Strong)'
upscale_15 = 'Upscale (1.5x)'
upscale_2 = 'Upscale (2x)'
upscale_fast = 'Upscale (Fast 2x)'

uov_list = [
    disabled, subtle_variation, strong_variation, upscale_15, upscale_2, upscale_fast
]

KSAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm"]

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

sampler_list = SAMPLER_NAMES
scheduler_list = SCHEDULER_NAMES
cn_ip = "Image Prompt"
cn_depth = "Depth"
cn_canny = "PyraCanny"
cn_cpds = "CPDS"
cn_pose = 'Pose'
cn_reColor = 'ReColor'
cn_Sketch = 'sketch'
# cn_revision = 'revision'
# cn_tileBlur = 'TileBlur'
# cn_tileBlurAnime = 'TileBlurAnime'

ip_list = [cn_ip, cn_canny, cn_cpds, cn_depth, cn_pose, cn_reColor, cn_Sketch,
           # cn_revision, cn_tileBlur,cn_tileBlurAnime
           ]
default_ip = cn_ip

default_parameters = {
    cn_ip: (0.5, 0.6),
    cn_canny: (0.5, 1.0),
    cn_cpds: (0.5, 1.0),
    cn_depth: (0.5, 1.0),
    cn_pose: (0.5, 1.0),
    cn_reColor: (0.5, 1.0),
    cn_Sketch: (0.5, 1.0),
    # cn_revision: (0.5, 1.0),
    # cn_tileBlur: (0.5, 1.0),
    # cn_tileBlurAnime: (0.5, 1.0),
}  # stop, weight

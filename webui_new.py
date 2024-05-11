from python_hijack import *

import gradio as gr
import random
import os
import time
import shared
import modules.path
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.advanced_parameters as advanced_parameters
import args_manager

from modules.sdxl_styles import legal_style_names, aspect_ratios
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript

state = 0

def generate_clicked(*args):
    execution_start_time = time.perf_counter()

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Initializing ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False)

    worker.buffer.append(list(args))
    finished = False

    while not finished:
        time.sleep(0.01)
        if len(worker.outputs) > 0:
            flag, product = worker.outputs.pop(0)
            if flag == 'preview':
                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


reload_javascript()

shared.gradio_root = gr.Blocks(
    title=f'Fooocus {fooocus_version.version} ' + (
        '' if args_manager.args.preset is None else args_manager.args.preset),
    css=modules.html.css).queue()

with shared.gradio_root:
    with gr.Row():
        with gr.Column():
            progress_window = grh.Image(label='Preview', show_label=True, height=640, visible=False)
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', height=745, visible=True,
                                 elem_classes='resizable_area')
            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=0.85):
                    prompt = gr.Textbox(show_label=False, placeholder="在此输入提示词.",
                                        value=modules.path.default_positive_prompt,
                                        container=False, autofocus=True, elem_classes='type_row', lines=1024)
                with gr.Column(scale=0.15, min_width=0):
                    generate_button = gr.Button(label="Generate", value="图片生成", elem_classes='type_row',
                                                elem_id='generate_button', visible=True)
                    skip_button = gr.Button(label="Skip", value="跳过当前", elem_classes='type_row_half', visible=False)
                    stop_button = gr.Button(label="Stop", value="停止生成", elem_classes='type_row_half',
                                            elem_id='stop_button', visible=False)


                    def stop_clicked():
                        import fcbh.model_management as model_management
                        shared.last_stop = 'stop'
                        model_management.interrupt_current_processing()
                        return [gr.update(interactive=False)] * 2


                    def skip_clicked():
                        import fcbh.model_management as model_management
                        shared.last_stop = 'skip'
                        model_management.interrupt_current_processing()
                        return


                    stop_button.click(stop_clicked, outputs=[skip_button, stop_button], queue=False,
                                      _js='cancelGenerateForever')
                    skip_button.click(skip_clicked, queue=False)
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='图片输入', value=False, container=False,
                                                   elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='高级', value=False, container=False, elem_classes='min_check')
            with gr.Row(visible=False) as image_input_panel:
                with gr.Tabs():
                    with gr.TabItem(label='图片提示模式') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for _ in range(4):
                                with gr.Column():
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy', show_label=False,
                                                         height=300)
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=False) as ad_col:
                                        with gr.Row():
                                            default_end, default_weight = flags.default_parameters[flags.default_ip]

                                            ip_stop = gr.Slider(label='提示阶段', minimum=0.0, maximum=1.0, step=0.001,
                                                                value=default_end)
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='权重', minimum=0.0, maximum=2.0, step=0.001,
                                                                  value=default_weight)
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        # ip_list_new = ["默认采样", "Canny采样", "综合采样", "深度图采样",
                                        #                "姿态采样", "重新上色", "Sketch采样"]

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list, value=flags.default_ip,
                                                           container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type],
                                                       outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        ip_advanced = gr.Checkbox(label='高级', value=False, container=False)


                        # gr.HTML(
                        #     '* \"Image Prompt\" is powered by Fooocus Image Mixture Engine (v1.0.1). <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Document</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)


                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights, queue=False)

                    with gr.TabItem(label='超分辨率与变化模式') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='上传图片', source='upload',
                                                            type='numpy')
                            with gr.Column():
                                uov_list_new = ["不启用", "变化（细微）", "变化（强烈）", "超分辨率（1.5x）", "超分辨率（2x）",
                                                "超分辨率（快速2x）"]
                                uov_method = gr.Radio(label='选择变化(Vary)或超分辨率(Upscale):',
                                                      choices=flags.uov_list,
                                                      value=flags.disabled)
                                # gr.HTML(
                                #     '<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Document</a>')

                    with gr.TabItem(label='局部重绘与扩图') as inpaint_tab:
                        inpaint_input_image = grh.Image(label='上传图片', source='upload', type='numpy',
                                                        tool='sketch', height=500, brush_color="#FFFFFF")
                        inpaint_input_mask = grh.Image(
                            label='重绘区域图片（可为空，可直接在上一个框中上传图片并直接选择重绘区域）', source='upload',
                            type='numpy',
                            tool='sketch', height=500, brush_color="#FFFFFF")

                        gr.HTML(
                            '扩图方向 (可多选):')
                        outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[],
                                                               label='Outpaint', show_label=False, container=False)
                        # gr.HTML(
                        #     '* \"Inpaint or Outpaint\" is powered by the sampler \"DPMPP Fooocus Seamless 2M SDE Karras Inpaint Sampler\" (beta)')

            switch_js = "(x) => {if(x){setTimeout(() => window.scrollTo({ top: 850, behavior: 'smooth' }), 50);}else{setTimeout(() => window.scrollTo({ top: 0, behavior: 'smooth' }), 50);} return x}"
            down_js = "() => {setTimeout(() => window.scrollTo({ top: 850, behavior: 'smooth' }), 50);}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, _js=switch_js)
            ip_advanced.change(lambda: None, queue=False, _js=down_js)

            current_tab = gr.Textbox(value='ip', visible=False)

            default_image = None


            def update_default_image(x):
                global default_image
                if isinstance(x, dict):
                    default_image = x['image']
                else:
                    default_image = x
                return


            def clear_default_image():
                global default_image
                default_image = None
                return


            uov_input_image.upload(update_default_image, inputs=uov_input_image, queue=False)
            inpaint_input_image.upload(update_default_image, inputs=inpaint_input_image, queue=False)

            uov_input_image.clear(clear_default_image, queue=False)
            inpaint_input_image.clear(clear_default_image, queue=False)

            uov_tab.select(lambda: ['uov', default_image], outputs=[current_tab, uov_input_image], queue=False,
                           _js=down_js)
            inpaint_tab.select(lambda: ['inpaint', default_image], outputs=[current_tab, inpaint_input_image],
                               queue=False, _js=down_js)
            ip_tab.select(lambda: 'ip', outputs=[current_tab], queue=False, _js=down_js)

        with gr.Column(scale=0.5, visible=False) as right_col:
            with gr.Tab(label='基础设置'):
                performance_selection = gr.Radio(label='生成模式', choices=['速度优先', '性能优先'], value='速度优先')
                aspect_ratios_selection = gr.Radio(label='生成图片分辨率', choices=list(aspect_ratios.keys()),
                                                   value=modules.path.default_aspect_ratio, info='width × height')
                image_number = gr.Slider(label='生成图片数量', minimum=1, maximum=4, step=1, value=1)
                negative_prompt = gr.Textbox(label='负向提示词', show_label=True, placeholder="在此输入负向提示词.",
                                             info='描述你不想看到的东西.', lines=2,
                                             value=modules.path.default_negative_prompt)
                seed_random = gr.Checkbox(label='随机生成', value=True)
                image_seed = gr.Number(label='随机种子', value=0, precision=0, visible=False)


                def random_checked(r):
                    return gr.update(visible=not r)


                def refresh_seed(r, s):
                    if r:
                        return random.randint(1, 1024 * 1024 * 1024)
                    else:
                        return s


                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed], queue=False)

                gr.HTML(
                    f'<a href="/file={get_current_html_path()}" target="_blank">\U0001F4DA 历史记录 </a>')

            with gr.Tab(label='模型'):
                with gr.Row():
                    base_model = gr.Dropdown(label='基础模型 (SDXL only)', choices=modules.path.model_filenames,
                                             value=modules.path.default_base_model_name, show_label=True)
                    refiner_model = gr.Dropdown(label='优化模型 (SDXL or SD 1.5)',
                                                choices=['None'] + modules.path.model_filenames,
                                                value=modules.path.default_refiner_model_name, show_label=True)
                with gr.Accordion(label='微调模型', open=True):
                    lora_ctrls = []
                    for i in range(5):
                        with gr.Row():
                            lora_model = gr.Dropdown(label=f'微调模型 {i + 1}',
                                                     choices=['None'] + modules.path.lora_filenames,
                                                     value='None')
                            lora_weight = gr.Slider(label='权重', minimum=0, maximum=2, step=0.01,
                                                    value=modules.path.default_lora_weight)
                            lora_ctrls += [lora_model, lora_weight]
                with gr.Row():
                    model_refresh = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files',
                                              variant='secondary', elem_classes='refresh_button')

            with gr.Tab(label='插件'):
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=legal_style_names,
                                                    value=modules.path.default_styles,
                                                    label='Image Style')

            with gr.Tab(label='高级设置'):
                sharpness = gr.Slider(label='锐化', minimum=0.0, maximum=30.0, step=0.001, value=2.0,
                                      info='值越高，表示图像和纹理越清晰.')
                guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                           value=modules.path.default_cfg_scale,
                                           info='值越高，表示图像风格更干净、更生动、更艺术.')

                # gr.HTML(
                #     '<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Document</a>')
                dev_mode = gr.Checkbox(label='开发者模式', value=False, container=False)

                with gr.Column(visible=False) as dev_tools:
                    with gr.Tab(label='Developer Debug Tools'):
                        adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=1.5,
                                                        info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                        adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=0.8,
                                                        info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                        adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                   step=0.001, value=0.3,
                                                   info='When to end the guidance from positive/negative ADM. ')

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method', value='joint',
                                                          choices=['joint', 'separate', 'vae'])

                        adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=7.0,
                                                 info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                      '(effective when real CFG > mimicked CFG).')
                        sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                   value=modules.path.default_sampler,
                                                   info='Only effective in non-inpaint mode.')
                        scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                     value=modules.path.default_scheduler,
                                                     info='Scheduler of Sampler.')

                        overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                   minimum=-1, maximum=200, step=1, value=-1,
                                                   info='Set as -1 to disable. For developer debugging.')
                        overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                     minimum=-1, maximum=200, step=1, value=-1,
                                                     info='Set as -1 to disable. For developer debugging.')
                        overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                    minimum=-1, maximum=2048, step=1, value=-1,
                                                    info='Set as -1 to disable. For developer debugging. '
                                                         'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                     minimum=-1, maximum=2048, step=1, value=-1,
                                                     info='Set as -1 to disable. For developer debugging. '
                                                          'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        overwrite_vary_strength = gr.Slider(label='Forced Overwrite of Denoising Strength of "Vary"',
                                                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                            info='Set as negative number to disable. For developer debugging.')
                        overwrite_upscale_strength = gr.Slider(
                            label='Forced Overwrite of Denoising Strength of "Upscale"',
                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                            info='Set as negative number to disable. For developer debugging.')

                        inpaint_engine = gr.Dropdown(label='Inpaint Engine', value='v1', choices=['v1', 'v2.5'],
                                                     info='Version of Fooocus inpaint model')

                    with gr.Tab(label='Control Debug'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessor of ControlNets', value=False)

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='Mixing Image Prompt and Vary/Upscale',
                                                                           value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                      value=False)

                        controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

                        with gr.Tab(label='PyramidCanny'):
                            canny_low_threshold = gr.Slider(label='PyramidCanny Low Threshold', minimum=1, maximum=255,
                                                            step=1, value=64)
                            canny_high_threshold = gr.Slider(label='PyramidCanny High Threshold', minimum=1,
                                                             maximum=255,
                                                             step=1, value=128)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                adps = [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name,
                        scheduler_name, overwrite_step, overwrite_switch, overwrite_width, overwrite_height,
                        overwrite_vary_strength, overwrite_upscale_strength,
                        mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint,
                        debugging_cn_preprocessor, controlnet_softness, canny_low_threshold, canny_high_threshold,
                        inpaint_engine, refiner_swap_method]
                adps += freeu_ctrls


                def dev_mode_checked(r):
                    return gr.update(visible=r)


                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools], queue=False)


                def model_refresh_clicked():
                    modules.path.update_all_model_names()
                    results = []
                    results += [gr.update(choices=modules.path.model_filenames),
                                gr.update(choices=['None'] + modules.path.model_filenames)]
                    for i in range(5):
                        results += [gr.update(choices=['None'] + modules.path.lora_filenames), gr.update()]
                    return results


                model_refresh.click(model_refresh_clicked, [], [base_model, refiner_model] + lora_ctrls, queue=False)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, right_col, queue=False)

        ctrls = [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, image_seed, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image, inpaint_input_mask]
        ctrls += ip_ctrls

        generate_button.click(lambda: (
            gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True),
            gr.update(visible=False),
            []), outputs=[stop_button, skip_button, generate_button, gallery]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed],
                  outputs=image_seed) \
            .then(advanced_parameters.set_all_advanced_parameters, inputs=adps) \
            .then(fn=generate_clicked, inputs=ctrls,
                  outputs=[progress_html, progress_window, gallery]) \
            .then(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
                  outputs=[generate_button, stop_button, skip_button]) \
            .then(fn=None, _js='playNotification')

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

shared.gradio_root.launch(
    inbrowser=args_manager.args.auto_launch,
    server_name="0.0.0.0",
    server_port=args_manager.args.port,
    share=args_manager.args.share
)

"""Image edits with words: Qwen-Image-Edit-2511 (Apache 2.0) on Intel's xpu (or CUDA), several edits per model load.

edit([(image, prompt, seed, out_path), ...]) runs in two phases so the text encoder (Qwen2.5-VL 7B, 16.6 GB) and the
image model (the GGUF Q8 build, 22.4 GB) are never on the GPU together: every prompt is encoded first, the encoder is
freed, then the image model loads once and makes every image. Neither is ever copied whole into RAM (WSL has about
12 GB free); run it in a memory cap anyway (studio jobs do: systemd-run -p MemoryMax=12G -p MemorySwapMax=0).
Peak GPU memory about 27 GB at 1024x1024; about 2 min to load, 4.3 min per image on an Arc Pro B70.

Needs diffusers >= 0.36 and gguf, and the weights:
    hf download Qwen/Qwen-Image-Edit-2511 --exclude "transformer/*.safetensors"
    hf download unsloth/Qwen-Image-Edit-2511-GGUF qwen-image-edit-2511-Q8_0.gguf
"""
import gc, time

import torch

REPO, GGUF_REPO = "Qwen/Qwen-Image-Edit-2511", "unsloth/Qwen-Image-Edit-2511-GGUF"
NEGATIVE = ("blurry, low quality, deformed, extra limbs, missing limbs, merged limbs, cropped, cut off, "
            "different outfit, different character, changed design")
DT = torch.bfloat16


def device():
    return "cuda" if torch.cuda.is_available() else "xpu"


def _patch(dev):
    if dev == "xpu":  # the B70's driver can't report free memory, which transformers asks for when loading onto it
        total = torch.xpu.get_device_properties(0).total_memory
        torch.xpu.mem_get_info = lambda d=None: (total - torch.xpu.memory_reserved(), total)
    import diffusers.models.model_loading_utils as mlu

    def gguf_to_device(path):  # diffusers' loader copies the whole file into RAM first (22 GB): read it tensor by tensor
        import gguf
        from diffusers.quantizers.gguf.utils import GGUFParameter
        out = {}
        for t in gguf.GGUFReader(path).tensors:
            w = torch.from_numpy(t.data).to(dev)  # from the memory map: only page cache, which the kernel can drop
            plain = t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
            out[t.name] = w if plain else GGUFParameter(w, quant_type=t.tensor_type)
        return out

    mlu.load_gguf_checkpoint = gguf_to_device


def edit(tasks, steps=40, cfg=4.0, gguf="Q8_0", negative=NEGATIVE, log=print):
    """tasks: (PIL image, prompt, seed, out path). Writes each image; returns the paths."""
    from diffusers import GGUFQuantizationConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import CONDITION_IMAGE_SIZE, calculate_dimensions
    from huggingface_hub import hf_hub_download
    from transformers import Qwen2_5_VLForConditionalGeneration

    dev = device()
    _patch(dev)
    t0 = time.time()
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=DT, device_map=dev)
    pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, transformer=None, text_encoder=te, torch_dtype=DT)
    pipe.vae.to(dev)
    encoded = []
    with torch.no_grad():
        for img, prompt, _, _ in tasks:
            img = img.convert("RGB")
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, img.width / img.height)
            cond = [pipe.image_processor.resize(img, ch, cw)]
            encoded.append((img, pipe.encode_prompt(prompt=prompt, image=cond, device=dev),
                            pipe.encode_prompt(prompt=negative, image=cond, device=dev)))
    pipe.text_encoder = None
    del te
    gc.collect()
    getattr(torch, dev).empty_cache()
    log(f"encoded {len(tasks)} prompt(s) in {time.time() - t0:.0f}s")

    pipe.transformer = QwenImageTransformer2DModel.from_single_file(
        hf_hub_download(GGUF_REPO, f"qwen-image-edit-2511-{gguf}.gguf"),
        quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT, config=REPO,
        subfolder="transformer", device=dev)
    log(f"image model loaded at {time.time() - t0:.0f}s")
    done = []
    for (img, (pe, pm), (ne, nm)), (_, _, seed, path) in zip(encoded, tasks):
        t1 = time.time()
        with torch.no_grad():
            out = pipe(image=[img], prompt_embeds=pe, prompt_embeds_mask=pm, negative_prompt_embeds=ne,
                       negative_prompt_embeds_mask=nm, true_cfg_scale=cfg, num_inference_steps=steps,
                       generator=torch.Generator(dev).manual_seed(seed)).images[0]
        out.save(path)
        done.append(path)
        log(f"wrote {path} {out.size} in {time.time() - t1:.0f}s")
    return done

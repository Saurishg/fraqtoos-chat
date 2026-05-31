#!/usr/bin/env python3
"""
Wan2.2-Animate runner for the FraqtoOS chat box.
  image + driving video  ->  preprocess (pose/face/ref)  ->  ComfyUI(8189) GGUF gen  ->  mp4

Run with the ROCm venv:  /home/work/ComfyUI/venv-rocm/bin/python wan_animate_run.py ...
Usage: wan_animate_run.py --image I --video V --output O [--prompt P] [--frames N] [--steps S]
Prints progress lines to stderr; writes the final mp4 to --output.
"""
import argparse, json, os, shutil, subprocess, sys, time, uuid, urllib.request

WAN_DIR    = "/home/work/Wan2.2"
PY         = "/home/work/ComfyUI/venv-rocm/bin/python"
PROC_CKPT  = "/home/work/ComfyUI/models/_wan22_animate/process_checkpoint"
COMFY      = "http://127.0.0.1:8189"
COMFY_IN   = "/home/work/ComfyUI/input"
COMFY_OUT  = "/home/work/ComfyUI/output"
ENV = {**os.environ, "HSA_OVERRIDE_GFX_VERSION": "10.3.0", "HIP_VISIBLE_DEVICES": "0"}


def log(m): print(f"[wan-animate] {m}", file=sys.stderr, flush=True)


def preprocess(image, video, out_dir, w, h, fps):
    log("preprocessing (pose / face / mask)…")
    cmd = [PY, f"{WAN_DIR}/wan/modules/animate/preprocess/preprocess_data.py",
           "--ckpt_path", PROC_CKPT, "--video_path", video, "--refer_path", image,
           "--save_path", out_dir, "--resolution_area", str(w), str(h),
           "--fps", str(fps), "--retarget_flag"]
    r = subprocess.run(cmd, cwd=WAN_DIR, env=ENV, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError("preprocess failed: " + (r.stderr or "")[-800:])
    for f in ("src_pose.mp4", "src_face.mp4", "src_ref.png"):
        if not os.path.isfile(os.path.join(out_dir, f)):
            raise RuntimeError(f"preprocess produced no {f}")


def frame_count(path):
    try:
        import cv2
        c = cv2.VideoCapture(path); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
        return n
    except Exception:
        return 0


def build_workflow(pose, face, ref, w, h, length, prompt, steps):
    neg = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 画面, 最差质量, 低质量, 畸形的, 多余的手指"
    return {
        "1":  {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-Animate-14B-Q8_0.gguf"}},
        "2":  {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "3":  {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4":  {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}},
        "5":  {"class_type": "VAELoader", "inputs": {"vae_name": "Wan2_1_VAE_bf16.safetensors"}},
        "6":  {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "7":  {"class_type": "LoadImage", "inputs": {"image": ref}},
        "8":  {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["6", 0], "image": ["7", 0], "crop": "center"}},
        "9":  {"class_type": "LoadVideo", "inputs": {"file": pose}},
        "10": {"class_type": "GetVideoComponents", "inputs": {"video": ["9", 0]}},
        "11": {"class_type": "LoadVideo", "inputs": {"file": face}},
        "12": {"class_type": "GetVideoComponents", "inputs": {"video": ["11", 0]}},
        "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0],
                "lora_name": "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", "strength_model": 1.0}},
        "13": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["20", 0], "shift": 8.0}},
        "14": {"class_type": "WanAnimateToVideo", "inputs": {
                "positive": ["3", 0], "negative": ["4", 0], "vae": ["5", 0],
                "width": w, "height": h, "length": length, "batch_size": 1,
                "continue_motion_max_frames": 5, "video_frame_offset": 0,
                "clip_vision_output": ["8", 0], "reference_image": ["7", 0],
                "face_video": ["12", 0], "pose_video": ["10", 0]}},
        "15": {"class_type": "KSampler", "inputs": {
                "model": ["13", 0], "seed": int(time.time()) % 2**31, "steps": steps,
                "cfg": 1.0, "sampler_name": "uni_pc", "scheduler": "simple",
                "positive": ["14", 0], "negative": ["14", 1], "latent_image": ["14", 2], "denoise": 1.0}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["5", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": 16.0}},
        "18": {"class_type": "SaveVideo", "inputs": {"video": ["17", 0],
                "filename_prefix": "chat_wananimate", "format": "mp4", "codec": "h264"}},
    }


def comfy_generate(workflow):
    data = json.dumps({"prompt": workflow}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        COMFY + "/prompt", data=data, headers={"Content-Type": "application/json"}), timeout=30)
    pid = json.loads(r.read())["prompt_id"]
    log(f"generating on 6800 XT (prompt {pid[:8]})…")
    for _ in range(700):  # up to ~35 min
        time.sleep(3)
        try:
            h = json.loads(urllib.request.urlopen(COMFY + f"/history/{pid}", timeout=10).read())
        except Exception:
            continue
        if pid not in h:
            continue
        st = h[pid].get("status", {})
        if st.get("status_str") == "error":
            details = []
            for m in st.get("messages", []):
                ev, info = m[0], (m[1] if len(m) > 1 else {})
                if ev == "execution_error":
                    details.append(f"{info.get('node_type')}: {info.get('exception_message')}")
                elif ev == "execution_interrupted":
                    details.append("interrupted (stopped by user)")
            raise RuntimeError("ComfyUI generation error — " + ("; ".join(details) or f"status={st.get('status_str')} (no node detail — likely OOM or interrupt)"))
        for o in h[pid].get("outputs", {}).values():
            for key in ("images", "gifs", "video"):
                if key in o and o[key]:
                    return o[key][0]["filename"]
    # Timed out — stop the orphaned ComfyUI job so it doesn't keep holding the GPU.
    try: urllib.request.urlopen(urllib.request.Request(COMFY + "/interrupt", data=b"", method="POST"), timeout=5)
    except Exception: pass
    raise RuntimeError("generation timed out (interrupted). This usually means too many "
                       "frames for the 6800 XT — try fewer frames (≤33) or it will exceed ~35 min.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--prompt", default="a person performing the motion, high quality")
    ap.add_argument("--frames", type=int, default=33); ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--width", type=int, default=832); ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    work = f"/tmp/wananimate_{uuid.uuid4().hex[:8]}"
    os.makedirs(work, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    try:
        preprocess(a.image, a.video, work, a.width, a.height, 16)
        avail = frame_count(os.path.join(work, "src_pose.mp4")) or a.frames
        length = max(5, min(a.frames, avail))
        length = ((length - 1) // 4) * 4 + 1  # Wan likes 4n+1
        log(f"pose frames available={avail}, generating length={length}")
        pose, face, ref = f"wa_{tag}_pose.mp4", f"wa_{tag}_face.mp4", f"wa_{tag}_ref.png"
        shutil.copy(os.path.join(work, "src_pose.mp4"), os.path.join(COMFY_IN, pose))
        shutil.copy(os.path.join(work, "src_face.mp4"), os.path.join(COMFY_IN, face))
        shutil.copy(os.path.join(work, "src_ref.png"),  os.path.join(COMFY_IN, ref))
        wf = build_workflow(pose, face, ref, a.width, a.height, length, a.prompt, a.steps)
        out_name = comfy_generate(wf)
        src = os.path.join(COMFY_OUT, out_name)
        shutil.copy(src, a.output)
        log(f"done -> {a.output}")
        # cleanup staged inputs
        for f in (pose, face, ref):
            try: os.remove(os.path.join(COMFY_IN, f))
            except OSError: pass
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

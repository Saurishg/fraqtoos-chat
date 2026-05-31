#!/usr/bin/env python3
"""
Wan2.1-VACE-14B (Q8 GGUF) runner for the FraqtoOS chat box — motion/structure control.
  control video (+ optional reference image) + prompt  ->  ComfyUI(8189) GGUF  ->  mp4

Run with the ROCm venv: /home/work/ComfyUI/venv-rocm/bin/python vace_run.py ...
Usage: vace_run.py --video CONTROL --output O [--image REF] [--prompt P]
                   [--frames N] [--steps S] [--strength F] [--width W] [--height H]
"""
import argparse, json, os, shutil, sys, time, uuid, urllib.request

COMFY="http://127.0.0.1:8189"; COMFY_IN="/home/work/ComfyUI/input"; COMFY_OUT="/home/work/ComfyUI/output"

def log(m): print(f"[vace] {m}", file=sys.stderr, flush=True)

def frame_count(path):
    try:
        import cv2; c=cv2.VideoCapture(path); n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release(); return n
    except Exception: return 0

def build(control, ref, w, h, length, prompt, steps, strength):
    neg="色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 画面, 最差质量, 低质量, 畸形的, 多余的手指"
    g={
     "1":{"class_type":"UnetLoaderGGUF","inputs":{"unet_name":"Wan2.1_14B_VACE-Q8_0.gguf"}},
     "2":{"class_type":"CLIPLoader","inputs":{"clip_name":"umt5_xxl_fp8_e4m3fn_scaled.safetensors","type":"wan"}},
     "3":{"class_type":"CLIPTextEncode","inputs":{"text":prompt,"clip":["2",0]}},
     "4":{"class_type":"CLIPTextEncode","inputs":{"text":neg,"clip":["2",0]}},
     "5":{"class_type":"VAELoader","inputs":{"vae_name":"Wan2_1_VAE_bf16.safetensors"}},
     "6":{"class_type":"LoadVideo","inputs":{"file":control}},
     "7":{"class_type":"GetVideoComponents","inputs":{"video":["6",0]}},
     "8":{"class_type":"ImageScale","inputs":{"image":["7",0],"upscale_method":"lanczos","width":w,"height":h,"crop":"center"}},
     "20":{"class_type":"LoraLoaderModelOnly","inputs":{"model":["1",0],
            "lora_name":"lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors","strength_model":1.0}},
     "10":{"class_type":"ModelSamplingSD3","inputs":{"model":["20",0],"shift":8.0}},
     "12":{"class_type":"KSampler","inputs":{"model":["10",0],"seed":int(time.time())%2**31,"steps":steps,
            "cfg":1.0,"sampler_name":"uni_pc","scheduler":"simple","positive":["11",0],"negative":["11",1],
            "latent_image":["11",2],"denoise":1.0}},
     "13":{"class_type":"VAEDecode","inputs":{"samples":["12",0],"vae":["5",0]}},
     "14":{"class_type":"CreateVideo","inputs":{"images":["13",0],"fps":16.0}},
     "15":{"class_type":"SaveVideo","inputs":{"video":["14",0],"filename_prefix":"chat_vace","format":"mp4","codec":"h264"}},
    }
    vace={"positive":["3",0],"negative":["4",0],"vae":["5",0],"width":w,"height":h,"length":length,
          "batch_size":1,"strength":strength,"control_video":["8",0]}
    if ref:
        g["9"]={"class_type":"LoadImage","inputs":{"image":ref}}
        g["16"]={"class_type":"ImageScale","inputs":{"image":["9",0],"upscale_method":"lanczos","width":w,"height":h,"crop":"center"}}
        vace["reference_image"]=["16",0]
    g["11"]={"class_type":"WanVaceToVideo","inputs":vace}
    return g

def generate(wf):
    data=json.dumps({"prompt":wf}).encode()
    pid=json.loads(urllib.request.urlopen(urllib.request.Request(COMFY+"/prompt",data=data,
        headers={"Content-Type":"application/json"}),timeout=30).read())["prompt_id"]
    log(f"generating on 6800 XT (prompt {pid[:8]})…")
    for _ in range(700):
        time.sleep(3)
        try: h=json.loads(urllib.request.urlopen(COMFY+f"/history/{pid}",timeout=10).read())
        except Exception: continue
        if pid not in h: continue
        st=h[pid].get("status",{})
        if st.get("status_str")=="error":
            details=[]
            for m in st.get("messages",[]):
                ev,info=m[0],(m[1] if len(m)>1 else {})
                if ev=="execution_error": details.append(f"{info.get('node_type')}: {info.get('exception_message')}")
                elif ev=="execution_interrupted": details.append("interrupted (stopped by user)")
            raise RuntimeError("ComfyUI error — "+("; ".join(details) or f"status={st.get('status_str')} (no node detail — likely OOM or interrupt)"))
        for o in h[pid].get("outputs",{}).values():
            for k in ("images","gifs","video"):
                if k in o and o[k]: return o[k][0]["filename"]
    try: urllib.request.urlopen(urllib.request.Request(COMFY+"/interrupt",data=b"",method="POST"),timeout=5)
    except Exception: pass
    raise RuntimeError("generation timed out (interrupted) — too many frames for the 6800 XT; try fewer (≤33).")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--image",default=""); ap.add_argument("--prompt",default="high quality, detailed, smooth motion")
    ap.add_argument("--frames",type=int,default=33); ap.add_argument("--steps",type=int,default=8)
    ap.add_argument("--strength",type=float,default=1.0)
    ap.add_argument("--width",type=int,default=832); ap.add_argument("--height",type=int,default=480)
    a=ap.parse_args(); tag=uuid.uuid4().hex[:8]
    ctrl=f"vace_{tag}_ctrl.mp4"; shutil.copy(a.video,os.path.join(COMFY_IN,ctrl))
    ref=""
    if a.image:
        ext=os.path.splitext(a.image)[1] or ".png"; ref=f"vace_{tag}_ref{ext}"
        shutil.copy(a.image,os.path.join(COMFY_IN,ref))
    try:
        avail=frame_count(os.path.join(COMFY_IN,ctrl)) or a.frames
        length=max(5,min(a.frames,avail)); length=((length-1)//4)*4+1
        log(f"control frames={avail}, length={length}")
        out=generate(build(ctrl,ref,a.width,a.height,length,a.prompt,a.steps,a.strength))
        shutil.copy(os.path.join(COMFY_OUT,out),a.output); log(f"done -> {a.output}")
    finally:
        for f in (ctrl,ref):
            if f:
                try: os.remove(os.path.join(COMFY_IN,f))
                except OSError: pass

if __name__=="__main__": main()

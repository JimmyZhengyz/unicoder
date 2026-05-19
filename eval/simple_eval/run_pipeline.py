import os
import sys
import json
import re
import argparse
import subprocess
import traceback
from tqdm import tqdm
from pathlib import Path
from models import QwenVLModel, OpenAIModel, GeminiModel, AnthropicModel, QwenAPIModel

# For PDF to PNG conversion, may not be needed
try:
    import fitz
except ImportError:
    fitz = None

# For different python versions
original_system = os.system
def patched_system(command):
    if command.strip().startswith("python3 "):
        command = command.replace("python3 ", f"{sys.executable} ", 1)
    elif command.strip().startswith("python "):
        command = command.replace("python ", f"{sys.executable} ", 1)
    return original_system(command)

os.system = patched_system

UNISVG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../unisvg.github.io"))
CHARTMIMIC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ChartMimic"))

def load_json_prompt(path):
    if not os.path.exists(path):
        print(f"Warning: Prompt file not found at {path}")
        return ""
    with open(path, 'r') as f:
        data = json.load(f)
        return data.get("instruction", "")

def parse_score(text):
    match = re.search(r'Score:\s*(\d+(\.\d+)?)', text, re.IGNORECASE)
    if match: return float(match.group(1))
    return 0.0

def pdf_to_png(pdf_path, png_path):
    if not fitz:
        return False
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=100) # Standard DPI
        pix.save(png_path)
        return True
    except Exception as e:
        print(f"PDF to PNG Error: {e}")
        return False

class ChartMimicHighLevelEvaluator:
    def __init__(self, judge_model):
        self.judge = judge_model
        self.prompt_path = os.path.join(CHARTMIMIC_ROOT, "chart2code/prompts/GPT4EvaluationAgent/gpt-4-vision-preview.json")
        self.prompt = load_json_prompt(self.prompt_path)
        
    def evaluate(self, gt_image_path, gen_image_path):
        if not self.judge or not self.prompt: return 0.0
        try:
            resp = self.judge.generate(self.prompt, image_path=[gt_image_path, gen_image_path])
            return parse_score(resp)
        except Exception as e:
            print(f"High-Level Eval Error: {e}")
            return 0.0

def load_model(args, model_type=None, model_path=None):
    m_type = model_type or args.model
    m_path = model_path or args.model_path
    
    print(f"\n[Model] Loading {m_type} ({m_path})...")
    
    if m_type == "qwen":
        return QwenVLModel(m_path)
    elif m_type == "qwen_api":
        return QwenAPIModel(model_name=m_path)
    elif m_type == "gpt":
        return OpenAIModel(model_name=m_path)
    elif m_type == "gemini":
        return GeminiModel(model_name=m_path)
    elif m_type == "claude":
        return AnthropicModel(model_name=m_path)
    else:
        raise ValueError(f"Unknown model type: {m_type}")

# For UniSVG
def run_unisvg(args, model):
    print("\n" + "="*50)
    print("Running UniSVG Pipeline (ISVGEN)")
    print("="*50)

    data_file = args.data_file
    image_dir = args.image_dir
    output_dir = os.path.join(args.output_dir, "unisvg")
    os.makedirs(output_dir, exist_ok=True)
    
    inference_file = os.path.join(output_dir, "inference.json")
    gen_params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 8192}
    
    if args.skip_inference and os.path.exists(inference_file):
        print(f"Skipping inference, using existing: {inference_file}")
    else:
        print(f"[UniSVG] Phase 1: Inference (Params: {gen_params})")
        with open(data_file, 'r') as f: full_data = json.load(f)
        
        data = [d for d in full_data if d.get('type') == 'ISVGEN']
        if args.num: data = data[:args.num]
        
        results = []
        prompt = "Please generate SVG code that recreates the image shown. Output only valid SVG code."
        
        for item in tqdm(data, desc="UniSVG Inference"):
            img_path = os.path.join(image_dir, item['image_path'])
            if not os.path.exists(img_path): continue
            
            try:
                resp = model.generate(prompt, img_path, **gen_params)
            except Exception: resp = ""
            
            results.append({
                "q_text": item.get("q_text", prompt),
                "image_path": item["image_path"],
                "model_answer": resp,
                "type": "ISVGEN"
            })
            
        with open(inference_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    print("[UniSVG] Phase 2: Evaluation (Running Official Script)")
    official_script = os.path.join(UNISVG_ROOT, "evaluation.py")
    
    # Predefined metrics as we only need ISVGEN and the rest are not used
    all_task_types = {
        'ISVGEN': {'ssim': 0, 'psnr': 0, 'lpips': 1, 'clip_score': 0},
        'TSVGEN': {'ssim': 0, 'psnr': 0, 'lpips': 1, 'clip_score': 0},
        'CSVGUN_color': {'accuracy': 0},
        'CSVGUN_category': {'accuracy': 0},
        'ISVGUN_color': {'accuracy': 0},
        'ISVGUN_category': {'accuracy': 0},
        'CSVGUN_size': {'accuracy': 0},
        'CSVGUN_shape': {'accuracy': 0},
        'CSVGUN_transform': {'accuracy': 0},
        'CSVGUN_usage': {'bertscore': 0, 'sbert_score': 0},
        'ISVGUN_usage': {'bertscore': 0, 'sbert_score': 0},
        'CSVGUN_rect': {'bertscore': 0, 'sbert_score': 0},
        'CSVGUN_circle': {'bertscore': 0, 'sbert_score': 0},
        'CSVGUN_description': {'bertscore': 0, 'sbert_score': 0},
        'ISVGUN_description': {'bertscore': 0, 'sbert_score': 0}
    }

    with open(official_script, 'r') as f: lines = f.readlines()
    new_lines = []
    
    injected_init = False
    
    for line in lines:
        if line.strip().startswith("input_folder ="):
            new_lines.append(f"input_folder = '{os.path.abspath(image_dir)}'\n")
        elif line.strip().startswith("igen_output_folder ="):
            new_lines.append(f"igen_output_folder = '{os.path.abspath(os.path.join(output_dir, 'svg_output'))}'\n")
        elif line.strip().startswith("tgen_output_folder ="):
            new_lines.append(f"tgen_output_folder = '{os.path.abspath(os.path.join(output_dir, 'tsvgen_output'))}'\n")
        elif line.strip().startswith("test_dataset_path ="):
            new_lines.append(f"test_dataset_path = '{os.path.abspath(data_file)}'\n")
        elif line.strip().startswith("model_answer_path ="):
            new_lines.append(f"model_answer_path = '{os.path.abspath(inference_file)}'\n")
        elif line.strip().startswith("output_json_path ="):
            new_lines.append(f"output_json_path = '{os.path.abspath(os.path.join(output_dir, 'scores.json'))}'\n")
            
        elif 'print("\\nFinal Results:")' in line and not injected_init:
            new_lines.append(line)
            new_lines.append("\n# === Injected by run_pipeline.py to fix KeyErrors ===\n")
            new_lines.append("all_task_types = " + json.dumps(all_task_types) + "\n")
            new_lines.append("for task, default_metrics in all_task_types.items():\n")
            new_lines.append("    if task not in type_metrics:\n")
            new_lines.append("        type_metrics[task] = default_metrics.copy()\n")
            new_lines.append("        type_samples[task] = 0\n")
            new_lines.append("    if task not in final_results.get('type_metrics', {}):\n")
            new_lines.append("        if 'type_metrics' not in final_results: final_results['type_metrics'] = {}\n")
            new_lines.append("        final_results['type_metrics'][task] = default_metrics.copy()\n")
            new_lines.append("# ====================================================\n")
            injected_init = True
            
        elif line.strip().startswith("isvgen_score = calculate_isvgen_tsvgen_score"):
             new_lines.append(line)
        elif line.strip().startswith("tsvgen_score = calculate_isvgen_tsvgen_score"):
             new_lines.append(line)
        elif "if 'ISVGEN' in final_results['type_metrics']:" in line: pass # Remove old patch
        elif "if 'TSVGEN' in final_results['type_metrics']:" in line: pass # Remove old patch
        elif "isvgen_score = 0" in line: pass
        elif "tsvgen_score = 0" in line: pass
        
        else:
            new_lines.append(line)
            
    temp_script = os.path.join(output_dir, "temp_eval.py")
    with open(temp_script, 'w') as f: f.writelines(new_lines)
    
    try:
        subprocess.run([sys.executable, temp_script], check=True)
        print(f"\nUniSVG Done! Results: {os.path.join(output_dir, 'scores.json')}")
    except subprocess.CalledProcessError:
        print("UniSVG Evaluation Failed.")
    finally:
        if os.path.exists(temp_script): os.remove(temp_script)


# For ChartMimic
def run_chartmimic(args, model, judge_model=None):
    print("\n" + "="*50)
    print("Running ChartMimic Pipeline (Direct)")
    print("="*50)

    dataset_dir = os.path.join(CHARTMIMIC_ROOT, "dataset", "direct_1800")
    output_dir = os.path.abspath(os.path.join(args.output_dir, "chartmimic"))
    code_dir = os.path.join(output_dir, "code")
    img_dir = os.path.join(output_dir, "img")
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    gen_prompt_path = os.path.join(CHARTMIMIC_ROOT, "chart2code/prompts/DirectAgent/gpt-4-vision-preview.json")
    gen_prompt = load_json_prompt(gen_prompt_path)
    if not gen_prompt:
        print("Error: Could not load generation prompt.")
        return

    high_level_evaluator = ChartMimicHighLevelEvaluator(judge_model) if judge_model else None
    gen_params = {"temperature": 0.1, "top_p": 1, "max_tokens": 4096}

    print(f"[ChartMimic] Phase 1: Inference (Params: {gen_params})")
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".png")])
    if args.num: files = files[:args.num]

    results = []

    for fname in tqdm(files, desc="ChartMimic Inference"):
        base_name = fname.replace(".png", "")
        img_path = os.path.join(dataset_dir, fname)
        gt_code_path = os.path.join(dataset_dir, base_name + ".py")
        
        try:
            resp = model.generate(gen_prompt, img_path, **gen_params)
        except Exception: resp = ""
            
        code_match = re.search(r'```python\n(.*?)```', resp, re.DOTALL)
        code = code_match.group(1) if code_match else resp
        
        gen_code_path = os.path.join(code_dir, f"{base_name}.py")
        gen_pdf_path = os.path.join(img_dir, f"{base_name}.pdf")
        gen_img_path = os.path.join(img_dir, f"{base_name}.png")
        
        code = re.sub(r"plt\.savefig\(.*\n*", "", code, flags=re.S)
        code = re.sub(r"plt\.show\(.*\n*", "", code, flags=re.S)
        
        code = code.strip().replace("\n", "\n    ")
        wrapped_code = f"try:\n    {code}\nexcept Exception as e:\n    pass\nplt.savefig('{gen_pdf_path}')"
            
        with open(gen_code_path, "w") as f: f.write(wrapped_code)
        
        exec_success = False
        try:
            proc = subprocess.run([sys.executable, gen_code_path], 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            
            if os.path.exists(gen_pdf_path) and fitz:
                if pdf_to_png(gen_pdf_path, gen_img_path):
                    exec_success = True
            elif os.path.exists(gen_pdf_path) and not fitz:
                exec_success = True
                
        except Exception as e:
            pass
            
        results.append({
            "id": base_name,
            "gen_code": gen_code_path,
            "gt_code": gt_code_path,
            "gen_img": gen_img_path,
            "gt_img": img_path,
            "exec_success": exec_success
        })

    print("[ChartMimic] Phase 2: Evaluation")
    
    original_cwd = os.getcwd()
    
    try:
        os.chdir(CHARTMIMIC_ROOT)
        print(f"Changed CWD to {CHARTMIMIC_ROOT} for evaluation")

        if CHARTMIMIC_ROOT not in sys.path:
            sys.path.append(CHARTMIMIC_ROOT)
        
        chart2code_path = os.path.join(CHARTMIMIC_ROOT, "chart2code")
        if chart2code_path not in sys.path:
            sys.path.append(chart2code_path)
            
        try:
            from utils.evaluator.text_evaluator import TextEvaluator
            from utils.evaluator.color_evaluator import ColorEvaluator
            from utils.evaluator.chart_type_evaluator import ChartTypeEvaluator
            from utils.evaluator.layout_evaluator import LayoutEvaluator
            
            os.environ["PROJECT_PATH"] = CHARTMIMIC_ROOT
            
            text_eval = TextEvaluator(use_position=False, use_axs=False)
            color_eval = ColorEvaluator()
            type_eval = ChartTypeEvaluator()
            layout_eval = LayoutEvaluator()
            
            evaluators_loaded = True
            print("ChartMimic Evaluators Loaded Successfully.")
        except ImportError as e:
            print(f"Error: ChartMimic evaluators not found. Path: {chart2code_path}")
            print(f"Details: {e}")
            evaluators_loaded = False
        
        scores = {"text": [], "color": [], "type": [], "layout": [], "high_level": []}
        
        for item in tqdm(results, desc="Eval"):
            if not item["exec_success"]: 
                if high_level_evaluator: scores["high_level"].append(0.0)
                continue
            
            gen_code_abs = os.path.abspath(item["gen_code"])
            gt_code_abs = os.path.abspath(item["gt_code"])
            gen_img_abs = os.path.abspath(item["gen_img"])
            gt_img_abs = os.path.abspath(item["gt_img"])

            if evaluators_loaded:
                try:
                    text_eval(gen_code_abs, gt_code_abs)
                    scores["text"].append(text_eval.metrics.get("f1", 0))
                    
                    color_eval(gen_code_abs, gt_code_abs)
                    scores["color"].append(color_eval.metrics.get("f1", 0))
                    
                    type_eval(gen_code_abs, gt_code_abs)
                    scores["type"].append(type_eval.metrics.get("f1", 0))
                    
                    layout_eval(gen_code_abs, gt_code_abs)
                    scores["layout"].append(layout_eval.metrics.get("f1", 0))
                except Exception as e:
                    print(f"Eval Error ({item['id']}): {e}")
            
            if high_level_evaluator:
                if os.path.exists(gen_img_abs):
                    hl_score = high_level_evaluator.evaluate(gt_img_abs, gen_img_abs)
                    scores["high_level"].append(hl_score)
                else:
                    scores["high_level"].append(0.0)

        print("\n" + "-"*30)
        exec_rate = sum(1 for r in results if r['exec_success'])/len(results) if len(results) > 0 else 0
        print(f"Execution Rate: {exec_rate:.2%}")
        
        for k, v in scores.items():
            if v:
                avg = sum(v)/len(v)
                print(f"{k.capitalize()} Score: {avg:.4f}")
            else:
                print(f"{k.capitalize()} Score: N/A")
        print("-"*30)
        
    finally:
        # Restore original CWD (The official script in ChartMimic needs to be run in the ChartMimic root directory)
        os.chdir(original_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["unisvg", "chartmimic", "all"])
    parser.add_argument("--model", type=str, required=True, help="Generator: qwen, qwen_api, gpt, gemini, claude")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--judge_model", type=str, help="Judge: qwen, gpt, etc. (Optional)")
    parser.add_argument("--judge_path", type=str, help="Judge Path/Name (Optional)")
    parser.add_argument("--data_file", type=str, help="UniSVG data file")
    parser.add_argument("--image_dir", type=str, help="UniSVG image dir")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--skip_inference", action="store_true")
    
    args = parser.parse_args()
    
    model = load_model(args) if not args.skip_inference else None
    
    judge = None
    if args.judge_model and args.judge_path:
        print("Loading Judge Model...")
        judge = load_model(args, model_type=args.judge_model, model_path=args.judge_path)

    if args.task == "unisvg" or args.task == "all":
        if not args.data_file or not args.image_dir:
            print("Error: UniSVG requires --data_file and --image_dir")
        else:
            run_unisvg(args, model)
            
    if args.task == "chartmimic" or args.task == "all":
        run_chartmimic(args, model, judge_model=judge)
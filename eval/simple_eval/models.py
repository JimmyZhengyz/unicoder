import os
import base64
import time
from typing import List, Optional, Union
from PIL import Image
import torch
import uuid
import shutil

class BaseModel:
    def generate(self, prompt: str, image_path: Union[str, List[str]] = None, **kwargs) -> str:
        raise NotImplementedError

class QwenVLModel(BaseModel):
    def __init__(self, model_path: str):
        print(f"Loading Qwen Model from {model_path}...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.process_vision_info = process_vision_info

    def generate(self, prompt: str, image_path: Union[str, List[str]] = None, **kwargs) -> str:
        messages = []
        content = []
        
        if image_path:
            if isinstance(image_path, str):
                content.append({"type": "image", "image": image_path})
            elif isinstance(image_path, list):
                for img in image_path:
                    content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.95),
        }

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

class OpenAIModel(BaseModel):
    def __init__(self, model_name="gpt-4o", api_key=None, base_url=None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL")
        )
        self.model_name = model_name

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(self, prompt: str, image_path: Union[str, List[str]] = None, **kwargs) -> str:
        content = []
        
        if image_path:
            paths = [image_path] if isinstance(image_path, str) else image_path
            for path in paths:
                base64_image = self._encode_image(path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
                
        content.append({"type": "text", "text": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.1),
                top_p=kwargs.get("top_p", 0.95)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return ""

class QwenAPIModel(BaseModel):
    def __init__(self, model_name="qwen-vl-max", api_key=None):
        try:
            import dashscope
        except ImportError:
            raise ImportError("Please install dashscope: pip install dashscope")
            
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for qwen_api models.")
        self.model_name = model_name
        dashscope.api_key = self.api_key

    def generate(self, prompt: str, image_path: Union[str, List[str]] = None, **kwargs) -> str:
        from dashscope import MultiModalConversation

        temp_files = []
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                content = []
                if image_path:
                    paths = [image_path] if isinstance(image_path, str) else image_path
                    for path in paths:
                        if os.path.exists(path):
                            ext = os.path.splitext(path)[1]
                            unique_name = f"temp_{uuid.uuid4().hex}{ext}"
                            temp_path = os.path.join(os.path.dirname(os.path.abspath(path)), unique_name)
                            shutil.copy(path, temp_path)
                            temp_files.append(temp_path)
                            path_to_use = temp_path
                        else:
                            path_to_use = path
                            
                        content.append({"image": f"file://{os.path.abspath(path_to_use)}"})
                
                content.append({"text": prompt})
                messages = [{"role": "user", "content": content}]

                gen_kwargs = {
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.1),
                    "top_p": kwargs.get("top_p", 0.95),
                }

                response = MultiModalConversation.call(
                    model=self.model_name,
                    messages=messages,
                    result_format='message',
                    **gen_kwargs
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content[0]["text"]
                else:
                    print(f"DashScope Error: {response.code} - {response.message}")
                    if attempt < max_retries - 1:
                        print(f"Retrying... ({attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    return ""
            except Exception as e:
                print(f"Qwen API Error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying... ({attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                return ""
            finally:
                for tf in temp_files:
                    if os.path.exists(tf):
                        try:
                            os.remove(tf)
                        except:
                            pass
                temp_files = []
        return ""

class GeminiModel(BaseModel):
    def __init__(self, model_name="gemini-2.0-flash", api_key=None):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, image_path: Union[str, List[str]] = None, **kwargs) -> str:
        content = []
        if image_path:
            paths = [image_path] if isinstance(image_path, str) else image_path
            for path in paths:
                content.append(Image.open(path))
        
        content.append(prompt)

        gen_config = {
            "max_output_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.95),
        }

        try:
            response = self.model.generate_content(
                content,
                generation_config=gen_config
            )
            return response.text
        except Exception as e:
            print(f"Gemini Error: {e}")
            return ""

class AnthropicModel(BaseModel):
    def __init__(self, model_name="claude-3-opus-20240229", api_key=None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def generate(self, prompt: str, image_path: Union[str, List[str]] = None, **kwargs) -> str:
        content = []
        if image_path:
            paths = [image_path] if isinstance(image_path, str) else image_path
            for path in paths:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": data}
                })
        
        content.append({"type": "text", "text": prompt})

        try:
            msg = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.1),
                top_p=kwargs.get("top_p", 0.95)
            )
            return msg.content[0].text
        except Exception as e:
            print(f"Claude Error: {e}")
            return ""

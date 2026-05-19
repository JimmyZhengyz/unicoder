from typing import List, Tuple, Any, Dict
from dotenv import load_dotenv
load_dotenv()

import os
import sys
# sys.path.insert(0, os.environ["PROJECT_PATH"])

import matplotlib.pyplot as plt
import eval_configs.global_config as gloabl_config

import re

class ChartTypeEvaluator:

    def __init__(self):
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0
        }
    
    def __call__(self, generation_code_file, golden_code_file):
        generation_chart_types = self._get_chart_types(generation_code_file)
        golden_chart_types = self._get_chart_types(golden_code_file)

        self.golden_code_file = golden_code_file

        self._calculate_metrics(generation_chart_types, golden_chart_types)

        redunant_file = os.environ["PROJECT_PATH"] + "/" + os.path.basename(golden_code_file).replace(".py", ".pdf")
        if os.path.exists(redunant_file) == True:
            os.remove(redunant_file)

        # print(self.metrics)
    
    def _get_chart_types(self, code_file):
        with open(code_file, "r") as f:
            lines = f.readlines()
        code = "".join(lines)

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", "_log_chart_types.txt")
        suffix = self._get_suffix(output_file)
        code = prefix + code + suffix

        code_log_chart_types_file = code_file.replace(".py", "_log_chart_types.py")
        with open(code_log_chart_types_file, "w") as f:
            f.write(code)
        
        os.system(f"python {code_log_chart_types_file}")

        if os.path.exists(output_file) == True:
            with open(output_file, "r") as f:
                chart_types = f.read()
                try:
                    chart_types = eval(chart_types)
                except:
                    chart_types = {}
            os.remove(output_file)
        else:
            chart_types = {}
        
        if os.path.exists(code_log_chart_types_file):
            os.remove(code_log_chart_types_file)
        
        # pdf_file = re.findall(r"plt\.savefig\('(.*)'\)", code)
        # if len(pdf_file) != 0:
        #     pdf_file = pdf_file[0].split(",")[0][:-1]
        #     print(pdf_file)
        #     if os.path.basename(pdf_file) == pdf_file:
        #         os.remove(pdf_file)
        
        return chart_types

    def _calculate_metrics(self, generation_chart_types: Dict[str, int], golden_chart_types: Dict[str, int]):
        """
        Calculate precision, recall, and f1 score of the chart types.

        Args:
            - generation_chart_types: Dict[str, int]
                - key: chart type
                - value: number of times the chart type is called
            - golden_chart_types: Dict[str, int]
                - key: chart type
                - value: number of times the chart type is called
        """
        if len(generation_chart_types) == 0:
            return

        n_correct = 0
        total = sum(generation_chart_types.values())

        for chart_type, count in generation_chart_types.items():
            if chart_type in golden_chart_types:
                n_correct += min(count, golden_chart_types[chart_type])

        self.metrics["precision"] = n_correct / total
        try:
            self.metrics["recall"] = n_correct / sum(golden_chart_types.values())
        except:
            print("<<<<<<<<<<<<<<<<<<<<golden_code_file", self.golden_code_file)
        if self.metrics["precision"] + self.metrics["recall"] == 0:
            self.metrics["f1"] = 0
        else:
            self.metrics["f1"] = 2 * self.metrics["precision"] * self.metrics["recall"] / (self.metrics["precision"] + self.metrics["recall"])
        return

    def _get_prefix(self):
        with open(os.environ["PROJECT_PATH"]+"/chart2code/utils/evaluator/chart_type_evaluator_prefix.py", "r") as f:
            prefix = f.read()
        return prefix

    def _get_suffix(self, output_file):  
        return f"""
# print(called_functions)
with open('{output_file}', 'w') as f:
    f.write(str(called_functions))
"""

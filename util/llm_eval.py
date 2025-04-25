import os
from .device import get_device
from openai import OpenAI
from dotenv import load_dotenv
from .generation import generate_nucleus
import json

BASE_DIR = os.path.join(os.path.dirname(__file__), "../llm_eval")
LLM_MODEL = 'gpt-4o'

SYSTEM_PROMPT = "You are a writing evaluator designed to assess student story completions. You will be provided children's stories written for a 3-4 year old audience. Your role is to provide constructive, fair, and detailed evaluations based on specific rubric criteria."

USER_PROMPT = """
In the following exercise, the student is given a pre-written beginning of a story. The student needs to complete this story. The exercise tests the student´s language abilities and creativity.

Here is the pre-written beginning:

<PROVIDED BEGINNING>
[STORY_BEGIN]
</PROVIDED BEGINNING>

Here is the students response:

<STUDENT RESPONSE>
[STORY_END]
</STUDENT RESPONSE>

First, provide a concise qualitative assessment about the student's writing. Then, give the writing a grade out of 10. These assessments should be done for each of the following rubric items:

1. Grammar:
* Is the writing grammatically correct?
* Evaluate syntax, punctuation, and sentence structure.
2. Consistency:
* Is the student's writing consistent with the provided beginning of the story?
* How well does the student complete the final sentence of the prescribed beginning?
3. Plot:
* Does the plot of the student's writing make sense (regardless of the provided beginning)?
4. Creativity: 
* How creative is the student's writing?

Format your response as follows:

<GRAMMAR>
[Qualitative assessment of grammar]
</GRAMMAR>
<GRAMMAR_GRADE>
[Grade out of 10]
</GRAMMAR_GRADE>

<CONSISTENCY>
[Qualitative assessment of consistency]
</CONSISTENCY>
<CONSISTENCY_GRADE>
[Grade out of 10]
</CONSISTENCY_GRADE>

<PLOT>
[Qualitative assessment of plot]
</PLOT>
<PLOT_GRADE>
[Grade out of 10]
</PLOT_GRADE>

<CREATIVITY>
[Qualitative assessment of creativity]
</CREATIVITY>
<CREATIVITY_GRADE>
[Grade out of 10]
</CREATIVITY_GRADE>

Provide your assessment below:
"""

class LLMEvaluator:
    def __init__(self, model, tokenizer, splits):
        self.model = model
        self.tokenizer = tokenizer
        self.test_loader = splits["test"]
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()
        
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        assert self.api_key is not None, "OPENAI_API_KEY is not set. Please set it in your environment variables."
        self.client = OpenAI(api_key=self.api_key)
        
        self.info_path = os.path.join(BASE_DIR, f"{model.config.name}/info.json")
        self.input_path = os.path.join(BASE_DIR, f"{model.config.name}/input.jsonl")
        self.output_path = os.path.join(BASE_DIR, f"{model.config.name}/output.jsonl")
        self.results_path = os.path.join(BASE_DIR, f"{model.config.name}/results.json")
    
    def generate_input(self, max_generations=200):
        
        input_items = []
        
        num_generations = 0
        num_skipped = 0
        
        for batch in self.test_loader:
            if num_generations >= max_generations:
                break
            
            sequence = batch[0].tolist()
            input_size = min(self.model.config.max_seq_len // 2, len(sequence) // 2)
            
            if input_size < 10:
                num_skipped += 1
                continue
            
            input = sequence[:input_size]
            
            generation = generate_nucleus(self.model, self.tokenizer, input, device=self.device)
            
            assert not self.tokenizer.eos_token_id in generation, "EOS token found in generation."
            
            decoded_input = self.tokenizer.decode(input)
            decoded_generation = self.tokenizer.decode(generation)
            
            prompt = USER_PROMPT.replace('[STORY_BEGIN]', decoded_input).replace('[STORY_END]', decoded_generation)
            
            id = f"{self.model.name}_{num_generations}"
            
            input_items.append({
                    "custom_id": f"{id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 1000
                    }
                })
            
            num_generations += 1
        print(f"Generated {num_generations} input items (skipped {num_skipped})")

        with open(self.input_path, "w+") as f:
            for item in input_items:
                f.write(json.dumps(item) + "\n")
        print(f"Input file written to {self.input_path}")
        
    def create_batch(self):
        
        if os.path.exists(self.info_path):
            print(f"Batch info file {self.info_path} exists.")
            return
        
        if not os.path.exists(self.input_path):
            self.generate_input()
        
        input_file = self.client.files.create(file=open(self.input_path, "rb"), purpose="batch")
        
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window='24h',
            metadata={
            'description': f'{self.model.name} evaluation'
            }
        )
        
        with open(self.info_path, "w+") as f:
            f.write(json.dumps({"batch_id": batch.id}))
        
        print(f"Created batch with ID: {batch.id}")
    
    def save_batch_output(self):
        
        if os.path.exists(self.output_path):
            print(f"Output file {self.output_path} exists.")
            return True
        
        with open(self.input_path, "r") as f:
            batch = json.load(f)
            batch_id = batch["batch_id"]
            if batch_id is None:
                raise ValueError("Input batch id was not found.")
        
        batch = self.client.batches.retrieve(batch["batch_id"])
        if batch.status == "completed":
            print(f"Batch {batch_id} completed.")
            output_id = batch.output_file_id
            output_text = self.client.files.retrieve(output_id).text
            with open(self.output_path, "w+") as f:
                f.write(output_text)
            print(f"Output written to {self.output_path}")
            return True
        elif batch.status == "failed":
            raise ValueError(f"Batch {batch_id} failed.")
        else:
            print(f"Batch {batch_id} is still processing.")
            return False
    
    def parse_batch_output(self):
        
        if os.path.exists(self.results_path):
            with open(self.results_path, "r") as f:
                return json.load(f)
        
        return None
        
    def run_llm_eval(self):
        
        self.create_batch()
        
        if not self.save_batch_output():
            return
        
        scores = self.parse_batch_output()
        print(f"Scores: {scores}")
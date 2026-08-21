INSTRUCTION = """You are a function caller. You are given a user query and the definition of a single API function. 
You must generate a function call using the exact name and parameters of the provided API function. You are not allowed to use any other function besides the one given. 
Return only the function call, using single quotes for strings and separating parameters with commas.

You are not permitted to deviate from the given API function in any way. You must use the exact function name and parameter types specified, even if you think another function would be more appropriate for the user's request.

Example:
-----------
API: 
def add_reminder(text: str, date: str, time: str):
  <api description>
Query: 
"Add a reminder to buy groceries tomorrow at 2 PM"
Response: 
add_reminder('Buy groceries', 'tomorrow', '2 PM')
-----------

Here is the API definition of the function you must use to fulfill the user's request:
"""


# INSTRUCTION="""Given a list of functions with their documentation, call the correct function
# with the correct parameters in the form function_name(parameter 1, parameter 2).
# Do not add any other text apart from the function call.
# Example: Can you add a note saying 'Rembember the milk'? Response: add_note('Remember the milk').
# Here is the documentation of all the functions.
# """

TOOL_LLAMA_GROQ_PROMPT = """
You are a function calling AI model. You are provided with a function signature within <tools></tools> XML tags. Your task is to call the given function to assist with the user query. Don't make assumptions about what values to plug into functions. Only return the function call with a standard format FUNCTION_NAME(ARGS).

Here are the available tool:
"""

pseudo_name_mapping = {
    "llama3-8B" : "meta-llama/Meta-Llama-3-8B-Instruct",
    'codestral-22B' : "mistralai/Codestral-22B-v0.1",
    'gemma2-2B' : "google/gemma-2-2b-it",
    'groqLlama3Tool-8B' : "Groq/Llama-3-Groq-8B-Tool-Use",
    'groqLlama3Tool-8B' : "Groq/Llama-3-Groq-8B-Tool-Use",
    "gemma3" : "google/gemma-3-1b-it",
    "qwen3" : "Qwen/Qwen3-4B",
    "llama3.2" : "meta-llama/Llama-3.2-3B-Instruct"
}

pseudo_name_instr_mapping = {
    "llama3-8B" : INSTRUCTION,
    'codestral-22B' : INSTRUCTION,
    'gemma2-2B' : INSTRUCTION,
    'groqLlama3Tool-8B' : TOOL_LLAMA_GROQ_PROMPT,
    "gemma3" : INSTRUCTION,
    "qwen3" : INSTRUCTION,
    "llama3.2" : INSTRUCTION
}


PROMPT_TEMPLATES = {
    'llama3-8B' : {
        'prompt_template' : (
            '<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}\n{api_def}'
            '<|eot_id|><|start_header_id|>user<|end_header_id|>{query}:<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>'
        ),
        'answer_template' : '<|start_header_id|>assistant<|end_header_id|>'
    },
    'codestral-22B' : {
        'prompt_template' : (
            '<s>[INST] {instruction}\n{api_def}\n\nQuery: {query}'
            '[/INST] {answer}</s>'
        ),
        'answer_template' : '[/INST]'
    },
    'gemma2-2B' : {
      'prompt_template' : (
          '<bos><start_of_turn>user\n{instruction}\n{api_def}\n\n{query}<end_of_turn>\n'
          '<start_of_turn>model\n{answer}<end_of_turn>\n'
      ),
      'answer_template' : '<start_of_turn>model\n'
    },
    'groqLlama3Tool-8B' : {
         'prompt_template' : (
            '<|start_header_id|>system<|end_header_id|>{instruction}\n<tools>{api_def}</tools>'
            '<|eot_id|><|start_header_id|>user<|end_header_id|>{query}'
            '<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>'
        ),
        'answer_template' : '<|start_header_id|>assistant<|end_header_id|>'
    },
    'qwen3' : {
        'prompt_template' : (
            '<|im_start|>system\n{instruction}\n{api_def}'
            '<|im_end|>\n<|im_start|>user\n{query}'
            '<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>'
        ),
        'answer_template' : '<|im_start|>assistant'
    },
    'llama3.2'  : {
        'prompt_template' : (
            '<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}\n{api_def}'
            '<|eot_id|><|start_header_id|>user<|end_header_id|>{query}:<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>'
        ),
        'answer_template' : '<|start_header_id|>assistant<|end_header_id|>'
    },
    'gemma3' : {
        'prompt_template' : (
            '<bos><start_of_turn>user\n{instruction}\n{api_def}\n\n{query}'
            '<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>'
        ),
        'answer_template' : '<start_of_turn>model'
    }
}
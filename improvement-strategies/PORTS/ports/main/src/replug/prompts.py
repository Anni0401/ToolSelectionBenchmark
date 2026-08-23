INSTRUCTION = '{}'
QUERY = '{}'
RESPONSE = '{}'
RETRIEVED_TEXT = '{}'

PROMPT_TEMPLATES = {
    'llama3' : {
        'prompt_template' : (
            f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{INSTRUCTION} {RETRIEVED_TEXT}'
            f'<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuery: {QUERY} Response:<|eot_id|>'
            f'<|start_header_id|>assistant<|end_header_id|>\n\n{RESPONSE}<|eot_id|>'
        ),
        'answer_template' : '<|start_header_id|>assistant<|end_header_id|>\n\n'
    },
    'phi3' : {
        'prompt_template' : (
            f'<|system|>\n{INSTRUCTION} {RETRIEVED_TEXT}<|end|>\n'
            f'<|user|>\nQuery: {QUERY} Response:<|end|>\n'
            f'<|assistant|>\n{RESPONSE}<|end|>'
        ),
        'answer_template' : '<|assistant|>\n'
    },
    'codestral' : {
        'prompt_template' : (
            f'<s>[INST] {INSTRUCTION} {RETRIEVED_TEXT}\n\n'
            f'Query: {QUERY} Response:[/INST] '
            f'{RESPONSE}</s>'
        ),
        'answer_template' : '[/INST]'
    },
    'llama3groq' : {
         'prompt_template' : (
            f'<|start_header_id|>system<|end_header_id|>{INSTRUCTION}\n<tools>{RETRIEVED_TEXT}</tools>'
            f'<|eot_id|><|start_header_id|>user<|end_header_id|>{QUERY}'
            f'<|eot_id|><|start_header_id|>assistant<|end_header_id|>{RESPONSE}<|eot_id|>'
        ),
        'answer_template' : '<|start_header_id|>assistant<|end_header_id|>'
    }

}

INSTRUCTIONS = {
    'function_calling' : (
        "Given a list of functions with their documentation, call the correct function "
        "with the correct parameters in the form function_name(parameter 1, parameter 2). "
        "Do not add any other text apart from the function call.\n"
        "Example: Can you add a note saying 'Rembember the milk'? Response: add_note('Remember the milk'). "
        "Here is the documentation of all the functions."
    ),
    'function_calling_groq' : (
        "You are a function calling AI model. You are provided with a function signature within <tools></tools> XML tags. Your task is to call the given function to assist with the user query. Don't make assumptions about what values to plug into functions. Only return the function call with a standard format FUNCTION_NAME(ARGS).\n\nHere are the available tool:"
    )
    
}
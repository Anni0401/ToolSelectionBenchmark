<div align="center"><img src="assets/fishing_rod_tool.png" width="10%"> </div>
<h1 align="center"><img src="assets/ports_icon.png" alt="port icon" width="25" height="auto"> PORTS</h1>
<h2 align="center">Preference-Optimized Retrievers for Tool Selection with Large Language Models  </h2>
<h3 align="center">Datasets Specifics</h3>

# ToolE Dataset Card

## Overview
The ToolE dataset is a benchmark to evaluate LLMs' tool awareness and their tool selection abilities. It was introduced in the paper [MetaTool](https://arxiv.org/abs/2310.03128). This dataset contains the training examples from the original benchmark which are related to single tools only.

## Data Splits
This dataset is organized into two main partitions:
- `train_test_data` contains the queries and the respective API selections. This partition contains the splits `train` and `test`, which contain examples related to different APIs. 
- `APIs`: collect the set of APIs used across the whole dataset

## Data Features
Each instance in the `train_test_data` partition contains the following features:
- **`query`**: A textual description of the user's request involving one API.
  - **`Example`**: "Can I find academic research papers on this topic?"
- **`response`**: The correct tool to call.
  - **`Example`**: "ResearchFinder"
- **`api_name`**: The ground truth tool for the query (since this datasest has responses made of tool selection only, the api_name is the same as the response for every example).

Each instance in the `APIs` partition contains the following features:
- **`api_name`**: The name of the API.
  - **`Example`**: "ResearchFinder"
- **`api_description`**: Description of the API.
  - **`Example`**: "ResearchFinder: Tool for searching academic papers."
 
# Octopus Dataset Card

## Overview
This dataset was parsed from the Hugginface page of [Octopus-v2](https://huggingface.co/NexaAIDev/Octopus-v2), a model introduced in the paper [Octopus v2: On-device language model for super agent](https://arxiv.org/abs/2404.01744).

## Data Splits
This dataset is organized into two main partitions:
- `train_test_data` contains the queries and the respective API selections. This partition contains the splits `train` and `test`, which contain examples related to different APIs. 
- `APIs`: collect the set of APIs used across the whole dataset.

## Data Features
Each instance in the `train_test_data` partition contains the following features:
- **`query`**: A textual description of the user's request involving one API.
  - **`Example`**: "How can I take a selfie with the front camera?"
- **`response`**: The correct function call to make.
  - **`Example`**: "take_a_photo('front')"
- **`api_name`**: The ground truth tool for the query.
  - **`Example`**: "take_a_photo"

Each instance in the `APIs` partition contains the following features:
- **`api_name`**: The name of the API.
  - **`Example`**: "take_a_photo"
- **`api_description`**: Description of the API.
  - **`Example`**: 
  ```
  def take_a_photo(camera):
    """
    Captures a photo using the specified camera and resolution settings.

    Parameters:
    - camera (str): Specifies the camera to use. Can be 'front' or 'back'. The default is 'back'.

    Returns:
    - str: The string contains the file path of the captured photo if successful, or an error message if not. Example: '/storage/emulated/0/Pictures/MyApp/IMG_20240310_123456.jpg'
    """
  ```




## Organization
This dataset is organized into two main partitions:
- `data`: contains the queries and the respective functions (it has only the test split)
- `function_corpora`: collect the set of function used across the whole dataset divided by scope, namely classes of function (here splits refer to the class names)

## How to use
The dataset's `data` partion has the following format:
```
features = Features({
    "id": Value("string"),                              # unique string identifier
    "scope": Value("string"),                           # type of query
    "question": Value("string"),                        # user question text
    "functions": Sequence({                             # list of functions at the model's disposal
        "name" : Value("string"),                       # - function name
        "description" : Value("string"),                # - general description of the function
        "parameters" : Sequence({                       # - list of input parameters for the function
            "type" : Value("string"),                   #   - name of the input argument
            "properties": Sequence({                    #   - property of the argument           
                "type": Value("string"),                #     - type of the property    
                "additionalProperties": Sequence({      #     - infos on the sub-paramters (if any)
                    "type": Value("string"),
                    "description": Value("string")
                })
            }), 
            "required" : Sequence(Value("string"))      #   - whether the parameters are optional or required
        })
    }),
    "ground_truth": Sequence(Value("string")),          # list of ground truth function (function to use for answering the question) 
    "augmented_descriptions": Sequence(Value("string")) # description of the functions augmented with all the information about its input paramters 
})
```
However, to meet HuggingFace's requirements regarding the usage of structured data such as lists or dictionaries, 
we had to parse the `functions` fields into a string. One can simply parse the data back to JSON to access each of its parameters. 
Here's a code example.
```Python
from datasets import load_dataset
import json

bfcl_ds = load_dataset("ToolRetriever/BFCL", "data", split="test")
first_instance = bfcl_ds[0]

function_data = json.loads(first_instance["functions"])
print(json.dumps(function_data,indent=4))
""" OUTPUT
{
    "name": "requests.get",
    "description": "Sends a GET request to the specified URL.",
    "parameters": {
        "type": "dict",
        "properties": {
            "url": {
                "type": "string",
                "description": "Convert any GPS Lat/Lon location into its timezone",
                "default": "https://timezone-by-location.p.rapidapi.com/timezone"
            },
            "headers": {
                "properties": {
                    "X-RapidAPI-Key": {
                        "type": "string",
                        "description": "The API key for authenticating requests to RapidAPI."
                    },
                    "X-RapidAPI-Host": {
                        "type": "string",
                        "description": "The host domain for the RapidAPI service being accessed."
                    }
                },
                "type": "dict",
                "required": [
                    "X-RapidAPI-Key",
                    "X-RapidAPI-Host"
                ]
            },
            "timeout": {
                "type": "integer",
                "description": "How many seconds to wait for the server to send data before giving up."
            },
            "params": {
                "properties": {
                    "lat": {
                        "type": "float",
                        "description": "Latitude of the position for which the timezone is being requested."
                    },
                    "lon": {
                        "type": "float",
                        "description": "Longitude of the position for which the timezone is being requested."
                    },
                    "c": {
                        "type": "integer",
                        "description": "Optional. Return compact JSON. Useful for reducing the size of the response data."
                    },
                    "s": {
                        "type": "integer",
                        "description": "Optional. Additional parameter, specifics not provided."
                    }
                },
                "type": "dict",
                "required": [
                    "lat",
                    "lon"
                ]
            },
            "allow_redirects": {
                "type": "boolean",
                "description": "A Boolean to enable/disable redirection.",
                "default": true
            },
            "auth": {
                "type": "tuple",
                "description": "A tuple to enable a certain HTTP authentication.",
                "default": "None",
                "items": {
                    "type": "string"
                }
            },
            "cert": {
                "type": "string",
                "description": "A String or Tuple specifying a cert file or key.",
                "default": "None"
            },
            "cookies": {
                "type": "dict",
                "additionalProperties": {
                    "type": "string"
                },
                "description": "Dictionary of cookies to send with the request."
            },
            "proxies": {
                "type": "dict",
                "additionalProperties": {
                    "type": "string"
                },
                "description": "Dictionary of the protocol to the proxy url."
            },
            "stream": {
                "type": "boolean",
                "description": "A Boolean indication if the response should be immediately downloaded (False) or streamed (True).",
                "default": false
            },
            "verify": {
                "type": "string",
                "description": "A Boolean or a String indication to verify the servers TLS certificate or not.",
                "default": true
            }
        },
        "required": [
            "url"
        ]
    }
}""""
```



# APIBank Dataset Card

## Overview:

The **APIBank**🏦 dataset is a collection of user-assistant conversations that interleaves messages with API requests interacting with external tools. 
Introduced in ["API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs"](https://arxiv.org/abs/2304.08244), APIBank collects api-augmented utterances
of three types:
- `level_1`: instances that test the ability to call APIs based on the given query when the APIs are known;
- `level_2`:  instances proving the ability to retrieve and call a single API when the APIs are unknown;
- `level_3`:  instances that test the ability to continuously plan, retrieve, and call multiple APIs when the APIs are unknown.


To test the effectiveness of retrieval systems and gauge their ability to correctly select useful API definitions based on the input context, we created two main dataset
partitions, each further divided according to the 3 levels of complexity mentioned above:
- `data`: storing utterances, APIs, and expected results (splits `level_1_parsed`, `level_2_parsed`, `level_3_parsed`);
- `corpora`: collecting the information of all APIs used in each data subste (splits `corpora_level_1`, `corpora_level_2`, `corpora_level_3`).


```Python
# load level_1 data
load_dataset("ToolRetriever/APIBank", "level_1_parsed")

# load API functions used in the train set of the level_1 queries
load_dataset("ToolRetriever/APIBank", "corpora_level_1", split="train")
```


### Post-processing
Data have been processed to remove API calls to the `ToolSercher` tool. This is because the function itself is replaced by the actions of the retrieval system. Therefore,
we deleted lines from the user-assistant message history mentioning this tool or having it as the next expected output of the intermediate conversation.

## Data Formats

### Retrieval Queries
The data samples for training and testing the retrieval capabilities given a conversational context are parsed into the following fields:
- `sample_id`: data unique identifier
- `api_id`: id of the API within the same conversation (if available in the original data)
- `instruction`: input prompt defining the instructions for solving the generative task
- `utterance`: user-assistant message history
- `utterance_retrieval`: user-assistant message history deprived from API-Requests messages to use as input query for the retrieval model
- `apis_params`: APIs given as input along with the input query (depending on the dataset complexity level)
- `output`: expected output API call
- `output_params`: parameters of the expected output API call

```
----------
sample_id
7
----------
api_id
0
----------
instruction
Generate an API request in the format of [ApiName(key1='value1', key2='value2', ...)] based on the previous dialogue context.The current time is 2039-03-09 18:56:09 Wednesday.Input: User: User's utterenceAI: AI's responseExpected output:API-Request: [ApiName(key1='value1', key2='value2', ...)]API descriptions:
----------
utterance
User: I need to know the estimated arrival time of an ambulance from 同仁医院 to 维多利亚酒店, the location of the incident is at the intersection of 西安路 and 复兴路.
AI: Can I first confirm the location of the nearest hospital to you? Please tell me your current location.
User: I am on 四川北路, near 和平公园.
API-Request: [get_nearby_hospital(location='四川北路, near 和平公园', distance='2000')]->[{"name": "\u540c\u4ec1\u533b\u9662", "location": "\u957f\u5b81\u533a\u9ec4\u6865\u8def170\u53f7"}, {"name": "\u534e\u5c71\u533b\u9662", "location": "\u5f90\u6c47\u533a\u534e\u5c71\u8def12\u53f7"}]
AI: Here are the nearby hospitals: 同仁医院 located at 长宁区黄桥路170号, and 华山医院 located at 徐汇区华山路12号. Which one would you like to choose for the ambulance?
User: I choose 同仁医院.
Generate API Request: 
----------
utterance_retrieval
User: I need to know the estimated arrival time of an ambulance from 同仁医院 to 维多利亚酒店, the location of the incident is at the intersection of 西安路 and 复兴路.
AI: Can I first confirm the location of the nearest hospital to you? Please tell me your current location.
User: I am on 四川北路, near 和平公园.
AI: Here are the nearby hospitals: 同仁医院 located at 长宁区黄桥路170号, and 华山医院 located at 徐汇区华山路12号. Which one would you like to choose for the ambulance?
User: I choose 同仁医院.
Generate API Request: 
----------
apis_params
[{"apiCode": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}, "response": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "The best match tool(s)."}}}]
----------
output
get_ambulance_eta
----------
output_params
incident_location='the intersection of 西安路 and 复兴路', hospital_location='同仁医院'
```

### API Corpora
The API corpora collects all the API definitions and parameters used in each specific data split. Instances are organized as follows:
- `api_name`: name of the tool
- `description`: a brief description of the API core functioning and purpose
- `parameters`: types and description of each input parameter
- `response`: types and description of each output parameter
- `augmented_description`: description of the function augmented with all the input and output parameters' details

```
----------------------------------------------------------------------------------------------------
api_name
RecordHealthData
----------------------------------------------------------------------------------------------------
parameters
{"user_id": {"type": "str", "description": "The ID of user."}, "time": {"type": "str", "description": "The time of health data. Format: %Y-%m-%d %H:%M:%S"}, "health_data": {"type": "list", "description": "The health data, with the format like [{'name': 'blood_pressure', 'value': '120/80'}, {'name': 'heart_rate', 'value': '80'}]"}}
----------------------------------------------------------------------------------------------------
description
This API records the health data of a user.
----------------------------------------------------------------------------------------------------
augmented_description

Description:
This API records the health data of a user.

Arguments:
---------
- user_id : str (optional)
  Description: The ID of user.
  Format: Not specified
- time : str (optional)
  Description: The time of health data. Format: %Y-%m-%d %H:%M:%S
  Format: Not specified
- health_data : list (optional)
  Description: The health data, with the format like [{'name': 'blood_pressure', 'value': '120/80'}, {'name': 'heart_rate', 'value': '80'}]
  Format: Not specified

Output:
---------
- status : str (optional)
  Description: The status of recording.
  Format: Not specified


----------------------------------------------------------------------------------------------------
response
{"status": {"type": "str", "description": "The status of recording."}}
```



# ToolBench Dataset Card

## Overview
The **ToolBench**⚙️ dataset is a comprehensive collection designed to assist in the evaluation and development of tool-augmented language models. Introduced in the paper [TOOLLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs](https://arxiv.org/pdf/2307.16789), this dataset includes various queries and associated API metadata, providing a robust resource for training and testing models on tool-use scenarios.

The ToolBench dataset has been parsed and divided into four distinct splits, each serving a unique purpose in understanding and utilizing the data. 

## Data Splits

### 1. `instructions` 🔍
This split is the main component of the ToolBench dataset, encompassing detailed instructions for three partitions (G1, G2, and G3). Each partition includes:
- `query`: The user's query or instruction.
- `relevant_apis`: APIs relevant to the given query.
- `apis_metadata`: Metadata information for each relevant API.

#### Group Details
- **G1**: Focuses on foundational queries and APIs, providing a broad base for general tool usage.
- **G2**: Contains more specialized queries, targeting APIs with specific functionalities.
- **G3**: Includes advanced queries that require complex interactions with APIs, often involving multiple tools.

#### Data Features
Each instance in the `instructions` split contains the following fields:

- **`query`**: A textual description of the user's request involving one or more APIs.
  - **Example**: "I'm working on a logistics project for my company and need to check the health of the SQUAKE API. Can you verify the API health by calling the 'Checkhealth' API endpoint? Additionally, I would like to retrieve the list of projects using the 'Projects' API endpoint."
  
- **`relevant_apis`**: A dictionary indicating the tools and corresponding APIs that are relevant to the query.
  - **Example**: 
    ```python
    {
      'tool_name': ['SQUAKE', 'SQUAKE'],
      'api_name': ['Checkhealth', 'Projects']
    }
    ```

- **`apis_metadata`**: A dictionary providing metadata for each relevant API, including the tool name, API name, description, and parameters.
  - **Example**: 
    ```python
    # First API
    {
        'tool_name': 'SQUAKE',
        'api_name': 'Checkhealth',
        'description': 'Check the health status of the SQUAKE authentication system.',
        'required_parameters': [
            {'name': 'origin', 'type': 'string', 'description': 'Origin city', 'default': 'New York'},
            {'name': 'destination', 'type': 'string', 'description': 'Destination city', 'default': 'London'}
        ]
    }
    ```


### 2. `g1_corpus` 📖
This split includes a comprehensive corpus of API metadata, providing detailed information about various APIs that can be used to respond to user queries.

#### Features
- `ID`: Unique identifier for each entry.
- `DOC_ID`: Document identifier.
- `category_name`: Category of the API.
- `tool_name`: Name of the tool.
- `api_name`: Name of the API.
- `api_description`: Description of the API.
- `required_parameters`: Parameters required by the API.
- `optional_parameters`: Optional parameters for the API.
- `method`: Method of the API.
- `template_response`: Template response provided by the API.


### 3. `g1_queries` ❓
This split consists of user queries that can be used to match with the relevant APIs from the corpus.

#### Features
- `query_id`: Identifier for each query.
- `text`: Text of the query.


### 4. `g1_raw` 🧱
This split includes raw data, combining user queries with relevant API details. It contains detailed fields for multiple APIs associated with each query.

#### Features
- `query`: User's query.
- `query_id`: Identifier for the query.
- `api_0_category_name` to `api_10_api_description`: Detailed information for up to 11 APIs related to the query.

## Loading the Dataset
Here is an example of how to load a specific split of our parsed version of the ToolBench dataset:

```python
from datasets import load_dataset

# Load the instructions split
dataset = load_dataset("ToolRetriever/ToolBench", "instructions")

# Accessing the G1 partition
g1_data = dataset["G1"]
```


# APIBank Dataset Card

Contains code generation queries that mostly make use of pre-trained neural networks. The models' data card are used as tool description.

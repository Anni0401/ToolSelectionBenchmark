import json


def read_json_file_to_list(input_file):
    result = []
    with open(input_file) as fin:
        for line in fin:
            obj = json.loads(line)
            result.append(obj)
    return result


def load_tool_clusters(input_file):
    """
    Load tool clusters for the generation pipeline.

    Supports two on-disk formats:
    1. "clustered": each line is already a list of tool-function definitions
       (e.g. tools/tools_en.jsonl).
    2. "flat": each line is a single tool-function definition carrying a
       "_source_tool_id" grouping key (e.g. a synthetic tool_schemas_cache.jsonl),
       which is grouped into clusters here. Any leading "_"-prefixed metadata
       keys are stripped before use.
    """
    raw_rows = read_json_file_to_list(input_file)
    if not raw_rows:
        return []

    if isinstance(raw_rows[0], list):
        return raw_rows

    clusters = {}
    order = []
    for row in raw_rows:
        cluster_key = row.get("_source_tool_id", row["function"]["name"])
        tool = {k: v for k, v in row.items() if not k.startswith("_")}
        if cluster_key not in clusters:
            clusters[cluster_key] = []
            order.append(cluster_key)
        clusters[cluster_key].append(tool)

    return [clusters[key] for key in order]


def write_json_data_to_file(data_list, output_file):
    fout = open(output_file, "w")
    for data in data_list:
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    fout.close()

import json
import argparse

def process(input_file, intermediate_file, output_file):
    """
    Process the input file to:
    1. Extract the longest instruction for each unique set of labels.
    2. Generate a mapping of labels to their corresponding instructions.
    """
    labels_tuple_dict = {}

    # Step 1: Extract the longest instruction for each unique set of labels
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            labels_list = sorted(data["labels"])
            labels_tuple = tuple(labels_list)
            
            response = data["data"]["output"]
            response_length = len(response)
            if labels_tuple not in labels_tuple_dict:
                labels_tuple_dict[labels_tuple] = (response_length, data)
            else:
                val = labels_tuple_dict[labels_tuple]
                if response_length > val[0]:
                    labels_tuple_dict[labels_tuple] = (response_length, data)

    with open(intermediate_file, "w", encoding="utf-8") as f:
        for key, value in labels_tuple_dict.items():
            line_dict = value[1]
            updated_line = json.dumps(line_dict, ensure_ascii=False)
            f.write(updated_line + "\n")

    # Step 2: Generate a mapping of labels to their corresponding instructions
    label_tag_dict = {}
    tag_list = []
    index_list = {}

    with open(intermediate_file, "r", encoding="utf-8") as f:
        for line in f:
            line_dict = json.loads(line)
            index = line_dict["index"]
            index_list[index] = line_dict

    with open(intermediate_file, "r", encoding="utf-8") as f:
        for line in f:
            line_dict = json.loads(line)
            index = line_dict["index"]
            tags = [i["tag_orgin"] for i in line_dict["tags"]]
            instruction = index_list[index]
            labels_list = [i["label"] for i in line_dict["tags"]]
            for tag in tags:
                if tag not in tag_list:
                    tag_list.append(tag)
            for label, tag in zip(labels_list, tags):
                if label not in label_tag_dict:
                    label_tag_dict[label] = [instruction]
                else:
                    if instruction not in label_tag_dict[label]:
                        label_tag_dict[label].append(instruction)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(label_tag_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input file to generate label-tag mapping.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("intermediate_file", type=str, help="Path to the intermediate JSON file.")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file.")
    
    args = parser.parse_args()
    
    process(args.input_file, args.intermediate_file, args.output_file)
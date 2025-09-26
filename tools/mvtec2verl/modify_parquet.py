import argparse

import pandas as pd


def modify_prompt(prompt_list):
    modified_prompt = []
    for item in prompt_list:
        if item.get("role") == "user":
            # 在user的content前加上<image>\n
            item["content"] = "<image>\n" + item["content"]
        modified_prompt.append(item)
    return modified_prompt


def modify_parquet(input_file, output_file):
    # 读取parquet文件
    df = pd.read_parquet(input_file)

    # 应用修改到每一行的prompt
    df["prompt"] = df["prompt"].apply(modify_prompt)

    # 保存修改后的dataframe到新的parquet文件
    df.to_parquet(output_file, index=False)
    print(f"Modified parquet saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modify prompt in parquet file by adding <image>\n before user content."
    )
    parser.add_argument("input_file", help="Path to the input parquet file")
    parser.add_argument("output_file", help="Path to the output parquet file")
    args = parser.parse_args()

    modify_parquet(args.input_file, args.output_file)

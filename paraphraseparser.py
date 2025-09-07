# correctly parses the downloaded paraphrased corpus into the format required by the model
with open("paraphrased-text/msr_paraphrase_test.txt", "r") as file, open(
    "paraphrased-text/prompts.txt", "w"
) as prompts, open("paraphrased-text/target.txt", "w") as target:
    headers = file.readline().strip().split("\t")
    # print(headers)

    index_quality = headers.index("\ufeffQuality")
    index_prompt = headers.index("#1 String")
    index_generated = headers.index("#2 String")


    # loop through line and index in file
    for line in file:
        # Split each line on tabs
        columns = line.strip().split("\t")
        if columns[index_quality] != "1":
            continue
        # Write the prompt and generated text to their respective files
        prompts.write(columns[index_prompt] + "\n")
        target.write(columns[index_generated] + "\n")



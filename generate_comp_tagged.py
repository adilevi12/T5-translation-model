from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from model import read_modified_file
from transformers import GenerationConfig

source_lang = "de"
target_lang = "en"
task_prefix = "translate German to English"
max_input_length = 500
max_target_length = 500
max_genration_length = 500


def tagger(unlabeld_path, outfile_path, model_path, max_genration_length):
    """
    Generate predictions for the unlabeled data and write them to a file
    :param unlabeld_path: the path to the unlabeled data
    :param outfile_path: the path to the output file
    :param model_path: the path of the model to use for generation
    :param max_genration_length: the maximum length of the generated sentences
    """

    # load the model
    generation_config, unused_kwargs = GenerationConfig.from_pretrained(model_path, return_unused_kwargs=True,
                                                                        max_new_tokens=max_genration_length,
                                                                        temperature=0.9)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # get the sentences to translate
    dataset = {'translation': []}
    _, de_data, root, modifiers = read_modified_file(unlabeld_path)
    for (de_sen, rt, mod) in zip(de_data, root, modifiers):
        dataset['translation'].append({'de': de_sen, 'root': rt, 'mod': mod})
    inputs = [f"{task_prefix} when Roots in English: {sen['root']}, Modifiers in English: {sen['mod']}: " + sen[
        source_lang] for sen in dataset["translation"]]

    # generate the translations
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True,
                             return_tensors="pt").input_ids
    print('Start Generating')
    translations = []
    jj = 0
    for single_input in model_inputs:
        model.to('cuda')
        outputs = model.generate(single_input.unsqueeze(0).to('cuda'), generation_config=generation_config)
        translations.append(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        if jj % 100 == 0:
            print(jj)
        jj += 1
        print('.', end='')

    # write the translations to a file in the right format
    generated_file = open(outfile_path, "w", encoding="utf-8")
    for i in range(len(de_data)):
        generated_file.write("German:\n" + de_data[i] + "\nEnglish:\n" + translations[i] + "\n\n")


if __name__ == '__main__':
    TEST = 'data/comp.unlabeled'
    TAGGED_TEST = 'comp_model.labeled'

    # write here the path to the model: must end with "modified-basline-new/checkpoint-17500"
    model_path = "/home/student/notebooks/workspace/modified-basline-new/checkpoint-17500"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create tagged test file
    tagger(TEST, TAGGED_TEST, model_path, max_genration_length)




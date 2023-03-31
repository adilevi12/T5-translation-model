import numpy as np
import datasets
import evaluate
import spacy
import string
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from project_evaluate import calculate_score, postprocess_text
from generate_comp_tagged import tagger
from transformers import GenerationConfig


model_name = "t5-base"
source_lang = "de"
target_lang = "en"
task_prefix = "translate German to English"


# load the pretrained t5-base model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def read_modified_file(file_path):
    """
    Read the labeled or unlabeled file with modifiers and roots and return the German and English sentences and the roots and modifiers
    :param file_path: (str) path to the file
    :return: (list) German sentences, (list) English sentences, (list) roots, (list) modifiers
    """
    root,modifiers, file_de, file_en = [], [], [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str.strip())
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            elif line.startswith("Roots in English:"):
                root.append(line.split(":")[1].strip())

            elif line.startswith("Modifiers in English:"):
                modifiers.append(line.split(":")[1].strip())
            else:
                cur_str += line + ' '
        if len(cur_str) > 0:
            cur_list.append(cur_str)

    return file_en, file_de, root, modifiers


def read_file_with_newline(file_path):
    """
    read the labeled file and return the German and English sentences with the \n character
    :param file_path: (str) path to the file
    :return:
    """
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            if line == 'English:\n' or line == 'German:\n':
                if len(cur_str) > 0:
                    cur_list.append(cur_str)
                    cur_str = ''
                if line == 'English:\n':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            cur_str += line
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def read_file_for_root_mod(file_path, out_file_path, sample_two=False):
    """
       Read the labeled file and generate the file with roots and modifiers
       :param file_path: (str) path to the file without modifiers and roots
       :param out_file_path: (str) output file path
       :param sample_two: (bool) whether to sample two modifiers per sentence or not
       :return:
    """

    # Read the labeled file without modifiers and roots and return the German and English sentences
    en_sentences, de_sentences = read_file_with_newline(file_path)

    # Open the output file
    out_file = open(out_file_path, 'w', encoding='utf-8')
    print('Finding roots...')
    en_root_mod = []
    for i in range(len(en_sentences)):
        # Get the root and modifiers of the English sentences
        en_root_mod.append(get_root_and_modifier(en_sentences[i].split('\n')[:-2], sample_two))
        if i % 100 == 0:
            print(f'{i}, ', end='')

    print()
    print('Writing...')
    # Write the German and English sentences and the roots and modifiers to the output file
    for i in range(len(en_sentences)):
        out_file.write("German:\n" + de_sentences[i] + en_root_mod[i] + "\nEnglish:\n" + en_sentences[i])
        if i % 100 == 0:
            print(f'{i}, ', end='')
    print()
    print('Done!')
    out_file.close()


def get_root_and_modifier(sentences, sample_two=False):
    """
    Get the root and modifiers of the English sentences
    :param sentences: (list) English sentences
    :param sample_two: (bool) whether to sample two modifiers or not for each root
    :return: (str) roots and modifiers for the given sentences
    """
    spacy_dep_parser = spacy.load("en_core_web_sm")
    all_roots = []
    all_modifiers = []
    for sent in sentences:

        # Create a Doc object for the sentence to be parsed
        doc = spacy_dep_parser(sent)

        # Iterate over the tokens in the sentence and print their dependencies
        root = ''
        no_punct_modifiers = []
        for token in doc:
            if token.pos_ == 'PUNCT':
                continue
            # Find the root of the sentence
            if token.dep_ == 'ROOT':
                root = token.text
                # Find the modifiers of the root
                modifiers = [child.text for child in token.children]
                no_punct_modifiers = [word for word in modifiers if word not in string.punctuation]
                break
        all_roots.append(root)

        # Sample two modifiers
        if sample_two and len(no_punct_modifiers) >= 2:
            all_modifiers.append(np.random.choice(no_punct_modifiers, 2, replace=False))
        else:
            all_modifiers.append(no_punct_modifiers)
    return f"Roots in English: {', '.join(all_roots)}\n" \
           f"Modifiers in English: {', '.join(['(' + ', '.join(modifiers) + ')' for modifiers in all_modifiers])}"


def genrate_val_mod(unlabel_file, label_file, out_file):
    """
    Generate the validation file with modifiers and roots
    :param unlabel_file: (str) path to the unlabeled file
    :param label_file: (str) path to the labeled file
    :param out_file: (str) path to the output file
    """
    _, file_de, root, modifiers = read_modified_file(unlabel_file)
    unlabeled_val = {"file_de": file_de, "root": root, "modifiers": modifiers}
    file_en, file_de = read_file_with_newline(label_file)
    labeled_val = {"file_en": file_en, "file_de": file_de}

    generated_file = open(out_file, "w", encoding="utf-8")
    if len(unlabeled_val["file_de"]) != len(labeled_val["file_de"]):
        raise Exception("The number of sentences in the unlabeled file and the labeled file are not the same!")

    for idx in range(len(unlabeled_val["file_de"])):
        generated_file.write("German:\n" + labeled_val["file_de"][idx] + "Roots in English:" + unlabeled_val["root"][idx] +"\nModifiers in English:" \
                       + unlabeled_val["modifiers"][idx]+ "\nEnglish:\n" + labeled_val["file_en"][idx])


def load_dataset(files):
    """
    Create train and validation datasets from the files
    :param files: (list of str) list of paths to the files
    :return: (list of datasets.Dataset) list of datasets
    """
    raw_datasets = []
    for path in files:
        # each dataset contains a "translation" feature
        dataset = {'translation': []}

        # read the file into 4 lists of roots, modifiers, english and german sentences
        en_data, de_data, root, modifiers =read_modified_file(path)

        # add the sentences and roots and m to the dataset
        for(en_sen, de_sen,rt,mod) in zip(en_data, de_data,root, modifiers):
            dataset['translation'].append({'de': de_sen, 'en': en_sen, 'root':rt, 'mod':mod})

        # Create a Dataset object from the dictionary and add it to the list
        dataset = datasets.Dataset.from_dict(dataset)
        raw_datasets.append(dataset)

    return raw_datasets


def preprocess_function(dataset):
    """
    Preprocess the dataset by tokenizing the inputs and targets
    :param dataset: (datasets.Dataset) dataset to preprocess
    :return: (datasets.Dataset) tokenized dataset
    """

    # Before each sentence is tokenized, we add the prefix of the task and the roots and modifiers
    inputs = [f"{task_prefix} when Roots in English: {sen['root']}, Modifiers in English: {sen['mod']}: "+sen[source_lang] for sen in dataset["translation"]]
    targets = [sen[target_lang] for sen in dataset["translation"]]

    # Tokenize the inputs and targets
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    # Rename the labels to "labels" to make it compatible with the T5ForConditionalGeneration model
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def compute_metrics_and_decode(eval_preds):
    """
    Compute the BLEU score and decode the predictions
    :param eval_preds: (tuple of np.ndarray) predictions and labels to evaluate from the trainer
    :return: (dict of str: float) of the BLEU score and the generated length
    """

    metric = evaluate.load("sacrebleu")
    preds, labels = eval_preds

    # preds is a tuple with predictions and labels
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode the predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode the labels
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # post-processing of the predictions and labels
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Compute the BLEU score and the average length of the predictions
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


if __name__ == '__main__':
    TRAIN = 'data/train.labeled'
    VAL_LABELED = 'data/val.labeled'
    VAL_UNLABELED = 'data/val.unlabeled'
    TRAIN_MOD = 'train.labeled.with.modifiers'
    VAL_MOD = 'val.labeled.with.modifiers'
    TAGGED_VAL = 'val_model.labeled'

    max_input_length = 500
    max_target_length = 500
    max_genration_length = 500
    batch_size = 4

    # Generate the train and validation files with modifiers and roots
    read_file_for_root_mod(TRAIN, TRAIN_MOD, True)
    genrate_val_mod(VAL_UNLABELED,VAL_LABELED, VAL_MOD)

    # load the datasets
    train_dataset, val_dataset = load_dataset([TRAIN_MOD, VAL_MOD])
    raw_datasets = datasets.DatasetDict({"train": train_dataset, "validation": val_dataset})

    # preprocess the datasets
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets.set_format('torch')


    # define the training arguments
    args = Seq2SeqTrainingArguments(
        output_dir="modified-basline-new",
        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=7,
        predict_with_generate=True,
        generation_max_length=500,
        fp16=True, logging_steps=1000
    )

    # define the trainer
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_decode
    )

    # train the model
    trainer.train()

    # write here the path to the model: must end with "modified-basline-new/checkpoint-17500"
    model_path = "/home/student/notebooks/workspace/modified-basline-new/checkpoint-17500"

    # Create tagged validation file
    tagger(VAL_UNLABELED, TAGGED_VAL, model_path, max_genration_length)

    # calculate the BLEU score of the generated translations
    calculate_score(VAL_LABELED, TAGGED_VAL)



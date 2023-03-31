# T5-translation-model
We fintuned the T5-base model to tha task of translating from German to English using the prompt:<br> 
"Translate German to English when Roots in English: <ROOTS>, Modifiers in English: <MODIFIERS>: <GERMAN TEXT>"<br>
In order to apply this model it was necessary to train the model on a training set with roots and modifiers, therefore we used the spacy "en_core_web_sm" model trained on the POS classification task and Dependency Parsing in order to find the potential roots and modifiers in the training file.<br>
We randomly selected 2 modifiers for each sentence from the model's predictions.<br>

Naturally, using modifiers and roots should improve performance since they can provide information about the structure of the sentence, which can help the model better understand the relationships between words and phrases, in addition they can provide context in the sentence and thus reduce the confusion that arises from rare prhases or words with double meaning.<br>
We adjusted the hyper-parameters of the model to: <br>


```
Epochs: 7 
Learning rate: 2e-4 
Batch size: 4
Max sentence length (also for generation): 500
```

The training time of the model was ~ 4 and a half hours.<br>
BLEU result on validation set: 43.66%.  

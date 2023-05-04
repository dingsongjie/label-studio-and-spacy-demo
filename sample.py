import spacy
import json
import random
from tqdm import tqdm
from spacy.training.example import Example
import numpy as np


TRAIN_DATA = []


def import_label_studio_data(filename):
    with open(filename, "rb") as fp:
        training_data = json.load(fp)
    for text in training_data:
        entities = []
        info = text.get("text")
        entities = []
        if text.get("label") is not None:
            list_ = []
            for label in text.get("label"):
                list_.append([label.get("start"), label.get("end")])
            a = np.array(list_)
            overlap_ind = []
            for i in range(0, len(a[:, 0])):
                a_comp = a[i]
                x = np.delete(a, (i), axis=0)
                overlap_flag = any([a_comp[0] in range(j[0], j[1] + 1) for j in x])
                if overlap_flag:
                    overlap_ind.append(i)

            for ind, label in enumerate(text.get("label")):
                if ind in overlap_ind:
                    iop = 0
                else:
                    if label.get("labels") is not None:
                        entities.append(
                            (
                                label.get("start"),
                                label.get("end"),
                                label.get("labels")[0],
                            )
                        )
        TRAIN_DATA.append((info, {"entities": entities}))


import_label_studio_data("label_studio_data.json")

model = None
if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank("zh")
    print("Created blank 'zh' model")

# set up the pipeline

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
#     nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe("ner")


def train_test_split(data, test):
    train_end = int(len(data) * (100 - test) * 0.01)
    print(train_end)
    test_start = int(len(data) * (100 - test)) + 1
    return data[0:train_end], data[train_end + 1 : len(data)]


train_data, test_data = train_test_split(TRAIN_DATA, test=50)


def train_ner_model(train_data_m, n_iter=1):
    for _, annotations in train_data_m:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data_m)
            losses = {}
            for text, annotations in tqdm(train_data_m):
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example], losses=losses, drop=0.2)
            print(losses)
        return nlp


nlp_model = train_ner_model(train_data, n_iter=20)


doc = nlp("这是一个很长的标贴.")

doc2 = nlp("黑色标贴一个.")
print(1)

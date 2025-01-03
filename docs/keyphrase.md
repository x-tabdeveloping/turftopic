# Keyphrase-based Topic Modeling with KeyNMF

KeyNMF, in its original form extracts topic descriptions as lists of words.
Sometimes, however it is desirable that one can use whole **keyphrases** instead of just single keywords, as these can be substantially more informative.

We can utilize keyphrases in KeyNMF by using [KeyphraseVectorizers](https://github.com/TimSchopf/KeyphraseVectorizers/tree/master):

```bash
pip install keyphrase-vectorizers
```

KeyphraseVectorizers extracts entire, grammatically correct noun phrases from text relying on POS-tag annotations from SpaCy.

## Data

For this demonstration, we will use a subset of 20 Newsgroups.

```python
from sklearn.datasets import fetch_20newsgroups

corpus = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
    categories=[
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "talk.religion.misc",
        "alt.atheism",
    ],
).data
```

## Model definition

We can define the model with a `KeyphraseCountVectorizer` as its vectorizer model.

```python
from turftopic import KeyNMF
from keyphrase_vectorizers import KeyphraseCountVectorizer

model = KeyNMF(n_components=10, vectorizer=KeyphraseCountVectorizer())
model.fit(corpus)

model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | windows, dos, os/2, file, files, os, microsoft, ms, program, unix |
| 1 | ., ?, -, !, thanks, anyone, ..., one, 's, ftp |
| 2 | atheism, atheist, atheists, alt.atheism, belief, religion, weak atheism, strong atheism, weak atheist, believe |
| 3 | disk, drive, drives, floppy, disks, dos, hard drive, ide, hard disk, bios |
| 4 | card, monitor, drivers, video card, vga, motherboard, ram, cards, driver, ati |
| 5 | morality, moral, objective, objective morality, morals, moral system, subjective, moral sense, natural morality, animals |
| 6 | scsi, scsi-2, scsi-1, scsi drive, scsi controller, scsi-2 controller, scsi-2 controller chip, fast scsi-1, scsi-2 chip, scsi-2 speeds |
| 7 | modem, port, serial, serial port, modems, ports, serial ports, com ports, null modem, dos |
| 8 | printer, print, printer driver, fonts, printing, driver, font, printers, hp, print manager |
| 9 | christians, christian, bible, christianity, god, religion, jesus, faith, believe, beliefs |


As you can see most topics are of much higher quality than what you normally expect with KeyNMF.
This, however comes at the price of slower modeling.

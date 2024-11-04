# Topic Namers

Sometimes, especially when the number of topics grows large,
it might be convenient to assign human-readable names to topics in an automated manner.

Turftopic allows you to accomplish this with a number of different topic namer models.

## Large Language Models

Turftopic lets you utilise Large Language Models for generating human-readable topic names.
This is done by instructing the language model to generate a topic name based on the keywords the topic model assigns as the most important for a given topic.

### Running LLMs locally

You can use any LLM from the HuggingFace Hub to generate topic names on your own machine.
The default in Turftopic is [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct), due to it's small size and speed, but we recommend using larger LLMs for higher quality topic names, especially in multilingual contexts.

```python
from turftopic import KeyNMF
from turftopic.namers import LLMTopicNamer

model = KeyNMF(10).fit(corpus)

namer = LLMTopicNamer("HuggingFaceTB/SmolLM2-1.7B-Instruct")
model.rename_topics(namer)

model.print_topics()
```

| Topic ID | Topic Name | Highest Ranking |
| - | - | - |
| 0 | Windows NT | windows, dos, os, ms, microsoft, unix, nt, memory, program, apps |
| 1 | Theism vs. Atheism | atheism, atheist, atheists, belief, religion, religious, theists, beliefs, believe, faith |
| 2 | "486 Motherboard" | motherboard, ram, memory, cpu, bios, isa, speed, 486, bus, performance |
| 3 | Disk Drives | disk, drive, scsi, drives, disks, floppy, ide, dos, controller, boot |
| 4 | Ethics | morality, moral, objective, immoral, morals, subjective, morally, society, animals, species |
| 5 | Christianity | christian, bible, christians, god, christianity, religion, jesus, faith, religious, biblical |
| 6 | modem-port-serial-connect-uart-pc-9600 | modem, port, serial, modems, ports, uart, pc, connect, fax, 9600 |
| 7 | "Graphics Card" | card, drivers, monitor, vga, driver, cards, ati, graphics, diamond, monitors |
| 8 | File Manager | file, files, ftp, bmp, windows, program, directory, bitmap, win3, zip |
| 9 | Printer and Fonts | printer, print, fonts, printing, font, printers, hp, driver, deskjet, prints |

### Using OpenAI's LLMs

You might not have the computational resources to run a high-quality LLM locally.
Luckily Turftopic allows you to use OpenAI's chat models for topic naming too!


!!! info
    You will also need to install the `openai` Python package.
    ```bash
    pip install openai
    export OPENAI_API_KEY="sk-<your key goes here>"
    ```

```python
from turftopic.namers import OpenAITopicNamer

namer = OpenAITopicNamer("gpt-4o-mini")
model.rename_topics(namer)
model.print_topics()
```

| Topic ID | Topic Name | Highest Ranking |
| - | - | - |
| 0 | Operating Systems and Software  | windows, dos, os, ms, microsoft, unix, nt, memory, program, apps |
| 1 | Atheism and Belief Systems | atheism, atheist, atheists, belief, religion, religious, theists, beliefs, believe, faith |
| 2 | Computer Architecture and Performance | motherboard, ram, memory, cpu, bios, isa, speed, 486, bus, performance |
| 3 | Storage Technologies | disk, drive, scsi, drives, disks, floppy, ide, dos, controller, boot |
| 4 | Moral Philosophy and Ethics | morality, moral, objective, immoral, morals, subjective, morally, society, animals, species |
| 5 | Christian Faith and Beliefs | christian, bible, christians, god, christianity, religion, jesus, faith, religious, biblical |
| 6 | Serial Modem Connectivity | modem, port, serial, modems, ports, uart, pc, connect, fax, 9600 |
| 7 | Graphics Card Drivers | card, drivers, monitor, vga, driver, cards, ati, graphics, diamond, monitors |
| 8 | Windows File Management | file, files, ftp, bmp, windows, program, directory, bitmap, win3, zip |
| 9 | Printer Font Management | printer, print, fonts, printing, font, printers, hp, driver, deskjet, prints |

### Prompting

Since these namers use chat-finetuned LLMs you can freely define custom prompts for topic name generation:

```python
from turftopic.namers import OpenAITopicNamer

system_prompt = """
You are a topic namer. When the user gives you a set of keywords, you respond with a name for the topic they describe.
You only repond briefly with the name of the topic, and nothing else.
"""

prompt_template = """
You will be tasked with naming a topic.
Based on the keywords, create a short label that best summarizes the topics.
Only respond with a short, human readable topic name and nothing else.

The topic is described by the following set of keywords: {keywords}.
"""

namer = OpenAITopicNamer("gpt-4o-mini", prompt_template=prompt_template, system_prompt=system_prompt)
```

## N-gram Patterns

You can also name topics based on the semantically closest n-grams from the corpus to the topic descriptions.
This method typically results in lower quality names, but might be good enough for your use case.


```python
from turftopic.namers import NgramTopicNamer

namer = NgramTopicNamer(corpus, encoder="all-MiniLM-L6-v2")
model.rename_topics(namer)
model.print_topics()
```

| Topic ID | Topic Name | Highest Ranking |
| - | - | - |
| 0 | windows and dos | windows, dos, os, ms, microsoft, unix, nt, memory, program, apps |
| 1 | many atheists out there | atheism, atheist, atheists, belief, religion, religious, theists, beliefs, believe, faith |
| 2 | hardware and software | motherboard, ram, memory, cpu, bios, isa, speed, 486, bus, performance |
| 3 | floppy disk drives and | disk, drive, scsi, drives, disks, floppy, ide, dos, controller, boot |
| 4 | morality is subjective | morality, moral, objective, immoral, morals, subjective, morally, society, animals, species |
| 5 | the christian bible | christian, bible, christians, god, christianity, religion, jesus, faith, religious, biblical |
| 6 | the serial port | modem, port, serial, modems, ports, uart, pc, connect, fax, 9600 |
| 7 | the video card | card, drivers, monitor, vga, driver, cards, ati, graphics, diamond, monitors |
| 8 | the file manager | file, files, ftp, bmp, windows, program, directory, bitmap, win3, zip |
| 9 | the print manager | printer, print, fonts, printing, font, printers, hp, driver, deskjet, prints |


## API Reference

:::turftopic.namers.base.TopicNamer

:::turftopic.namers.hf_transformers.LLMTopicNamer

:::turftopic.namers.openai.OpenAITopicNamer

:::turftopic.namers.ngram.NgramTopicNamer

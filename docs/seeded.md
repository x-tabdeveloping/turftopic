# Seeded Topic Modeling

When investigating a set of documents, you might already have an idea about what aspects you would like to explore.
Some models are able to account for this by taking seed phrases or words.
This is currently only possible with KeyNMF in Turftopic, but will likely be extended in the future.

In [KeyNMF](../keynmf.md), you can describe the aspect, from which you want to investigate your corpus, using a free-text seed-phrase,
which will then be used to only extract topics, which are relevant to your research question.

In this example we investigate the 20Newsgroups corpus from three different aspects:

```python
from sklearn.datasets import fetch_20newsgroups

from turftopic import KeyNMF

corpus = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
).data

model = KeyNMF(5, seed_phrase="<your seed phrase>")
model.fit(corpus)

model.print_topics()
```


=== "`'Is the death penalty moral?'`"

    | Topic ID | Highest Ranking |
    | - | - |
    | 0 | morality, moral, immoral, morals, objective, morally, animals, society, species, behavior |
    | 1 | armenian, armenians, genocide, armenia, turkish, turks, soviet, massacre, azerbaijan, kurdish |
    | 2 | murder, punishment, death, innocent, penalty, kill, crime, moral, criminals, executed |
    | 3 | gun, guns, firearms, crime, handgun, firearm, weapons, handguns, law, criminals |
    | 4 | jews, israeli, israel, god, jewish, christians, sin, christian, palestinians, christianity |

=== "`'Evidence for the existence of god'`"

    | Topic ID | Highest Ranking |
    | - | - |
    | 0 | atheist, atheists, religion, religious, theists, beliefs, christianity, christian, religions, agnostic |
    | 1 | bible, christians, christian, christianity, church, scripture, religion, jesus, faith, biblical |
    | 2 | god, existence, exist, exists, universe, creation, argument, creator, believe, life |
    | 3 | believe, faith, belief, evidence, blindly, believing, gods, believed, beliefs, convince |
    | 4 | atheism, atheists, agnosticism, belief, arguments, believe, existence, alt, believing, argument |

=== "`'Operating system kernels'`"

    | Topic ID | Highest Ranking |
    | - | - |
    | 0 | windows, dos, os, microsoft, ms, apps, pc, nt, file, shareware |
    | 1 | ram, motherboard, card, monitor, memory, cpu, vga, mhz, bios, intel |
    | 2 | unix, os, linux, intel, systems, programming, applications, compiler, software, platform |
    | 3 | disk, scsi, disks, drive, floppy, drives, dos, controller, cd, boot |
    | 4 | software, mac, hardware, ibm, graphics, apple, computer, pc, modem, program |



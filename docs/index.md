Turftopic is a topic modeling library which intends to simplify and streamline the usage of contextually sensitive topic models.
We provide stable, minimal and scalable implementations of several types of models along with extensive documentation,
so that you can make an informed choice about which model suits you best in the light of a given task or research question.

## Installation

Turftopic can be installed from PyPI.

```bash
pip install turftopic
```

If you intend to use CTMs, make sure to install the package with Pyro as an optional dependency.

```bash
pip install turftopic[pyro-ppl]
```

## Getting Started

Turftopic's models follow the scikit-learn API conventions, and as such they are quite easy to use if you are familiar with
scikit-learn workflows.

Here's an example of how you use KeyNMF, one of our models on the 20Newsgroups dataset from scikit-learn.

```python
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
)
corpus = newsgroups.data
```

Turftopic also comes with interpretation tools that make it easy to display and understand your results.

```python
from turftopic import KeyNMF

model = KeyNMF(20).fit(corpus)
model.print_topics()
```

| Topic ID | Top 10 Words                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------- |
|        0 | armenians, armenian, armenia, turks, turkish, genocide, azerbaijan, soviet, turkey, azerbaijani |
|        1 | sale, price, shipping, offer, sell, prices, interested, 00, games, selling                      |
|        2 | christians, christian, bible, christianity, church, god, scripture, faith, jesus, sin           |
|        3 | encryption, chip, clipper, nsa, security, secure, privacy, encrypted, crypto, cryptography      |
|        4 | israel, israeli, jews, jewish, arab, palestinians, arabs, palestinian, israelis, palestine      |
|        5 | hockey, nhl, bruins, leafs, playoff, rangers, islanders, montreal, team, playoffs               |
|        6 | windows, dos, os, ms, microsoft, unix, window, pc, drivers, program                             |
|        7 | gun, guns, firearms, weapons, crime, amendment, law, handgun, firearm, police                   |
|        8 | card, ram, motherboard, memory, speed, cpu, mhz, performance, cards, chip                       |
|        9 | disk, drive, scsi, floppy, drives, disks, mac, dos, seagate, ide                                |
|       10 | braves, cubs, mets, phillies, sox, yankees, orioles, standings, astros, dodgers                 |
|       11 | graphics, software, program, 3d, hardware, unix, computer, pc, image, programs                  |
|       12 | bike, car, cars, riding, bikes, ride, motorcycle, speed, honda, engine                          |
|       13 | modem, mac, modems, serial, fax, port, 9600, cable, 2400, hardware                              |
|       14 | atheism, atheist, religion, atheists, belief, religious, believe, god, beliefs, faith           |
|       15 | monitor, vga, monitors, screen, resolution, display, apple, video, svga, card                   |
|       16 | file, files, ftp, format, program, formats, bmp, gif, copy, directory                           |
|       17 | mail, address, email, send, nasa, space, edu, mailing, list, newsgroup                          |
|       18 | printer, print, hp, printing, printers, laser, fonts, driver, paper, deskjet                    |
|       19 | baseball, pitching, pitcher, hitter, pitchers, pitch, inning, batting, players, league          |

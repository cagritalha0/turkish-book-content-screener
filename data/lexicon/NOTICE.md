# Lexicon notice

`offensive_tr.txt` is derived from `uygunsuz-kelime-listesi/ofansif.txt` in
[KitapMetre](https://github.com/Abra-Muhara/kitapmetre-2024AcikHackTDDI) by the Abra Muhara team
(TEKNOFEST 2024 Turkish NLP competition), licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Changes made

- Removed 45 duplicate entries (381 → 336 lines).
- Removed 72 terms, leaving 264:
  - **Identity terms** (ethnicities, nationalities, religions and beliefs, political ideologies,
    sexual orientations, disability, body type), e.g. `kürt`, `ermeni`, `yahudi`, `arap`, `ateist`,
    `gay`, `engelli`, `şişman`, `alman`. Naming a group is not inappropriate content; flagging it
    teaches the system to treat minorities as offensive.
  - **Common words that caused false positives in children's books**, e.g. `peri` (fairy),
    `cin` (genie), `ok` (arrow), `hasta` (ill), `ilişki` (relationship), `namus` (honour), `annesini`.

The list is still a blunt instrument. It is only one input feature next to the BERT score, never
the sole reason a book is flagged.

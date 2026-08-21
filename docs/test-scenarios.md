# Test Scenarios — Hybrid Sparse Search Demo

Demo queries built from the actual vocabulary of the 1,000-caption Flickr30k subset
(`data/flickr_docs.json`). The goal of a good demo is to **contrast search modes on the
same query** — pick queries where `lexical` and `sparse`/`hybrid` visibly diverge.

Run any query several ways to compare. **Neural-sparse POC** (`hybrid-sparse-index`):

```
GET /sparse/search?q=<query>&mode=lexical
GET /sparse/search?q=<query>&mode=sparse
GET /sparse/search?q=<query>&mode=hybrid
```

**Dense semantic POC** (`semantic-dense-index`, `neural` kNN query — runs the embedding
model at query time):

```
GET /semantic/search?q=<query>
```

The same query set below works for both. The most instructive demo is running a query
through `mode=lexical` (string matching), then `mode=sparse`, then `/semantic/search`
(dense) — three different retrieval mechanisms, same dataset.

---

## ⭐ The money shot: words that appear in *zero* captions

The captions never contain "puppy" or "canine" — but "dog" appears **71 times**.

| Query      | `lexical`              | `sparse` / `hybrid`      |
| ---------- | ---------------------- | ------------------------ |
| **puppy**  | ~nothing (word absent) | returns dog photos ✅     |
| **canine** | nothing (word absent)  | returns dog photos ✅     |

This is the clearest demonstration that neural sparse understands *meaning*, not string
matching. Run `mode=lexical` first (empty/poor), then `mode=sparse` — instant "aha."

## Semantic matching (rare literal term → common concept)

| Query                            | Literal hits  | Semantically maps to        |
| -------------------------------- | ------------- | --------------------------- |
| **cyclist**                      | 3             | bike (35) + bicycle (25)    |
| **musician** / a person performing music | 5    | guitar (14), music (9)      |
| **toddler**                      | 20            | child (114), "little boy/girl" |
| **kids swimming in the ocean**   | ocean only 21 | water (105), pool (91), beach (53) |
| **elderly man**                  | 22            | old, plus man (1308)        |

Compare `lexical` vs `hybrid` here — hybrid pulls in the conceptually-related captions
lexical misses.

## Scene phrases (great for hybrid, natural for an audience)

These read like real user searches and have solid coverage:

- **a dog running on the beach** — dog + beach + running all well-represented
- **children playing outside** — children (142) + playing (252) + outside (147)
- **man riding a bicycle** — riding (113) + bike/bicycle (60)
- **people sitting at a table** — sitting (386) + table (100)
- **a group of people in the water** — group (211) + water (219)

## Exact-match baseline (shows lexical is *fine* when words match)

Useful contrast — proves the system isn't just always favoring semantic:

- **red shirt**, **white dog**, **blue jacket**, **snow**, **skateboard**

Colors (red 307, white 375, blue 335) + objects give crisp exact matches where `lexical`
alone does well.

---

## Suggested 3-query demo flow

1. **puppy** → `lexical` (fails) then `sparse` (finds dogs) — the wow moment.
2. **a dog running on the beach** → `hybrid` — natural phrase, rich results.
3. **red shirt** → `lexical` — shows exact matching still works, and hybrid balances both.

> Counts above are literal caption occurrences in the current 1,000-image subset
> (`FLICKR_SUBSET_SIZE=1000`). If you change the subset size, the exact numbers shift but
> the relationships (semantic vs. literal) hold.

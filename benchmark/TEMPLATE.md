# ClaimsSQL-Trust benchmark — record template

There are two files:

| File | Who writes it | What it is |
|---|---|---|
| `benchmark/questions.yaml` | the project owner | The questions, SQL, tags and notes. **This is the file to edit.** |
| `benchmark/questions.jsonl` | the validator | Generated from the YAML, one record per line, with `gold_result_hash` and `split` added. Do not edit it by hand, and keep it in git: it is where each question's split is remembered. |

The validator (`python -m claimssql.eval.validate_benchmark`) reads
`questions.yaml` and never modifies it.

## Record

`questions.yaml` is a list. Each question is one list item.

Answerable question:

```yaml
- id: <ID>
  group: <easy|medium|hard>
  question: <your question>
  sql: |
    <your SQL>
    <can continue over several lines>
  tags: [<tag>, <tag>]
  notes: <optional note>
```

Infeasible question (it simply has no `sql`):

```yaml
- id: <ID>
  group: <T1|T2|T3|T4>
  question: <your question>
  tags: []
  notes: <why it cannot be answered>
```

## Fields

| Field | Required | Rules |
|---|---|---|
| `id` | yes | Unique. Letters, digits, `-` and `_` only (it becomes a file name in `gold_previews/`). |
| `group` | yes | One of the seven groups below. It decides whether the question is answerable. |
| `question` | yes | The natural-language question. Not empty. |
| `sql` | `easy`, `medium`, `hard`: yes. `T1`–`T4`: must be absent. | One read-only DuckDB query. |
| `tags` | no | Zero or more tags from `benchmark/trap_tags.yaml`. |
| `notes` | no | Free text. For infeasible questions: the reason it cannot be answered. |

Any other field name is reported as an error, so a typo such as `querry:` cannot
go unnoticed.

## Writing YAML safely

- Indent with spaces, never tabs. The fields of a record are indented two spaces
  under its `- id:` line.
- Write SQL as a block: `sql: |` and then the query on the following lines,
  indented further than `sql`. Everything in the block is taken literally, so
  quotes, colons, `#` and `--` comments need no escaping.
- If a question or note contains a colon followed by a space, a `#`, or starts
  with a quote or other punctuation, put it in double quotes
  (`question: "<text with: a colon>"`) or use a block (`question: >-` and the
  text on the next line).
- Text that looks like a number or true/false must be quoted to stay text
  (`notes: "2008"`).

Only the spaces and line breaks around a text are removed when it is copied to
`questions.jsonl`; the text itself is never changed.

## Groups and target counts

| Group | Answerable | Meaning | Target |
|---|---|---|---|
| `easy` | yes | One table, simple filter, one aggregate | 40 |
| `medium` | yes | A join or GROUP BY across categories, date ranges, or two or more conditions | 40 |
| `hard` | yes | Multi-step logic such as window functions, self-joins, year-over-year comparisons, summing across column families, or percentage-of-total | 40 |
| `T1` | no | Missing concept | 25 |
| `T2` | no | Not in data | 15 |
| `T3` | no | Ambiguous | 25 |
| `T4` | no | Not a SQL question | 15 |
| **Total** | | | **200** |

The targets live in `configs/experiment.yaml` (`benchmark.composition`).

## Trap tags

Allowed tags and their meanings are listed in `benchmark/trap_tags.yaml`. A tag
that is not in that file is reported as an error. To introduce a new tag, add it
to that file first.

## How gold results are compared

A model's answer is correct when its result equals the gold result under these
rules (`src/claimssql/eval/compare.py`):

- Column names are ignored, so aliases do not matter. Column order does matter.
- Row order is ignored. Duplicate rows count.
- Numbers are rounded to 4 decimal places; `5`, `5.0` and `5.00` are equal.
- Text is stripped of surrounding spaces; case matters. Text `'1'` is not the number `1`.
- Results are capped at 10,000 rows; a larger gold result cannot be scored reliably.

## Workflow

1. Try a query while you write it:
   `python scripts/try_sql.py "<your SQL>"` or `python scripts/try_sql.py -f query.sql`
2. Add the record to `questions.yaml`.
3. Check your work without assigning splits:
   `python -m claimssql.eval.validate_benchmark --no-split`
4. Read the report, and look at `benchmark/gold_previews/<id>.csv` (first 20 rows
   of each gold result) to confirm every query returns what you meant.
5. When all 200 records are in, run it once without `--no-split` to assign the
   dev/test split (20% dev per group, fixed seed).

What the validator reports:

- **Invalid records** (with the line in `questions.yaml`): these are not written
  to `questions.jsonl`. If the record was valid on an earlier run, its earlier
  version is kept there until you fix it, so its split is not lost.
- **Gold SQL failures**: the record is written without a hash.
- **Warnings**: a gold result that is empty, is a single NULL, has more than
  10,000 rows, or changed although the SQL did not; an infeasible question
  without a reason in `notes`; an id that disappeared from `questions.yaml`.

A split is tied to the `id`. Rewording a question or changing its SQL keeps the
split; renaming or deleting the id discards it.

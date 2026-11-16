#!/bin/sh
# Reads a PR's files as JSON on stdin, from the GitHub files API.
# Prints each path protected.sh should judge, old names included.
# A README.md whose patch changes only crate-table lines is left out.
# Lint fails any table the generator did not write.
exec jq -r '
  def table_line: test("^$|^<!-- crate table -->$|^</?table>$|^<tr>.*</tr>$|^Crate \\| Description$|^--- \\| ---$|^\\[[^]]*\\]\\([^)]*\\) \\| ");
  def generated:
    .status as $status
    | (.filename | test("(^|/)README\\.md$")) and (.patch != null) and
      ([.patch | split("\n")[] | select(test("^[-+]")) | .[1:]
        | table_line or ($status == "added" and test("^# "))] | all);
  .[] | select(generated | not) | .filename, (.previous_filename // empty)
'

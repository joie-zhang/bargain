# E26. Population coalition results

## Scope and result
- Covers the main coalition figure and all population claims in the current paper's `4_analysis.tex`, lines 81–107.
- Includes counts by game, organizer Elo, run family, and voting support.
- Gemini case-study details and matched GPT-5.4 replication are assigned to E27 and E30.
- This audit made no experiment, paper, configuration, or code changes.

## Verified provenance chain
1. The multi-agent and random-monoculture batch launchers call the shared experiment runner.
2. Two annotation manifests enumerate 2,730 plus 325 historical runs.
3. Original Codex audits screened these 3,055 runs and manually read high-signal candidates and comparison samples.
4. The August 17 recode stores harmful-only labels for 52 of 111 previously identified plans in Python dictionaries and a case table.
5. The combined producer adds the Game 3 config 0270 failed exclusion proposal.
6. Organizer and exposure builders attach the fixed Elo snapshot and aggregate cases.
7. The August 30 review adapter changes the figure to the current three families.
8. The independent September 5 reproduction preserves raw-denominator and ballot checks.

## Independent checks in this audit
- Loaded every one of the 3,055 manifest result files and matched saved agent count and model map.
- Recomputed 2,320 currently eligible runs after removing two-agent runs, all-Nano controls, and Claude 3 Haiku sensitivity runs.
- Current eligible counts are 880 in Game 1 and 720 each in Games 2 and 3.
- Saved harmful proposals are 45, 6, and 2 after the extra Game 3 case.
- Accepted harmful outcomes are 30, 2, and 1.
- Game 1 therefore supplies 45/53 = 84.9% of proposals and 30/33 = 90.9% of accepted outcomes.
- Independently matched all 45 organizer identities against raw model maps and confirmed 44 have Elo above 1400.
- Read the raw discussion supporting Game 3 config 0270.
- Independently re-read all 228 individual ballots in the 30 accepted Game 1 cases.
- The ten accepted Gemini cases have 53 member yes votes and 21 outsider no votes.
- Independently checked all 15 original judge child sessions and 56 stored turn-context records.

## Important limits and discrepancies
- The paper calls the judges GPT-5.6-xhigh, but all checked original contexts record gpt-5.6-sol with high effort.
- The original method used search/screening plus manual candidate review, not a separate API judge applied identically to every transcript.
- The report names config 0415 as a likely omitted case, so saved counts cannot establish complete prevalence.
- Current homogeneous Game 1 denominator is 80, not the paper's 100.
- The current homogeneous proposal rate is 12/80 = 15%, while Gemini alone is 12/20 = 60%.
- The current family figure shows homogeneous 12 proposed, 10 accepted, and 10 with full planned-member support.
- Heterogeneous cases show 28 proposed, 16 accepted, and four with full planned-member support.
- The remaining 12 accepted heterogeneous cases need outside voters, but only five have explicit evidence of a factual voting mistake.
- Config 0412 uses the final planned membership; using an earlier superseded plan changes the clean-support count.
- The adapter subtracts 40 scored exposures from the wrong zero-proposal Elo bin, so independent raw reconstruction matters even though plotted zero rates do not change.
- I did not perform a new blind semantic annotation pass or rerun any model.

## Needed files and cleanup decision
- The companion JSON lists concrete scripts, annotations, histories, and ballot files with reasons.
- Raw-result roots use an explicit manifest selection rule rather than a claim that every file in each directory is required.
- The original screen scripts and saved manual dictionaries are research inputs, even though they live in dated analysis folders.
- The dated review adapter imports two files from `scripts/retained_analysis`, so deleting that directory would break current figure production.
- Preserve original child-session files because they contain judge instructions, actual settings, and otherwise incomplete manual-decision provenance.
- Shared launch/runtime dependencies are cross-covered by E10–E18.
- No deletion candidate is supported by this result audit.

## History
- The bundled codex-search query with required terms `111` and `coalition` returned original session `01a00e54-55d3-7b90-a045-c06c7980615e`.
- The broad initial query was noisy, so its top result was not treated as proof.
- Original August 16 session lines 321, 341, 349, and 1368 establish the request, method, child launch, and actual scope.
- Original August 17 lines 1011, 1411, and 1482 establish the harmful-only recode and possible omission.
- Original August 22 lines 278 and 303 establish membership conventions and the difference between outside votes and proven mistakes.
- Complete absolute history paths and concrete dependency paths are in the companion JSON.


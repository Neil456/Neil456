# Synthetic data audit (pilot)

Automatically generated; automatic checks passed for every item below. Automatic checks do **not** guarantee semantic correctness. Each sample needs human review: tick the box if the question is unambiguous and the program answers it.

* items: 1600; splits: {'train': 1412, 'synth_dev': 188}
* kinds: {'average': 152, 'ratio': 141, 'margin': 145, 'multi_step_negative': 117, 'unit_conversion': 73, 'weighted_average': 132, 'cagr': 111, 'table_highest': 38, 'growth': 118, 'share_of_total': 131, 'margin_change': 137, 'table_average': 44, 'comparison': 131, 'table_lowest': 25, 'per_share': 74, 'table_total': 31}
* difficulty: {'medium': 708, 'easy': 629, 'hard': 263}
* rejections: {'template_not_applicable': 80, 'implausible_growth': 2, 'exec_failed:division by zero': 2, 'trivial_answer_is_cell': 2}

## Samples for human review

### 1. `synth:synth/software/00197:0` (table_highest, easy)

Table:

| ( in thousands ) | 2019 | 2018 |
|---|---|---|
| total revenues | $ 514098 | $ 461046 |
| gross profit | 400043 | 365550 |
| operating expenses | 225828 | 190136 |
| operating income | 174215 | 175414 |
| interest expense | 9913 | 1928 |
| income tax expense | 40292 | 42545 |
| net income | $ 124010 | $ 130941 |
| capital expenditures | -19493 ( 19493 ) | -16056 ( 16056 ) |
| net cash provided by operating activities | 153558 | 147466 |

Text: in 2019 , consumer contributed $ 185885 thousands with a segment margin of 34.1% ( 34.1 % ), commercial contributed $ 227816 thousands with a segment margin of 44.1% ( 44.1 % ), government contributed $ 100397 thousands with a segment margin of 26.1% ( 26.1 % ) . as of december 31 , 2019 , we had approximately 66209 employees and 605.7 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 134065 thousands in 2019 .

**Q:** what was the highest total revenues from 2018 through 2019?
**Program:** `table_max(total revenues, none)`
**Answer:** 514098.0

- [ ] reviewed: unambiguous and correct

### 2. `synth:synth/biotech/00388:0` (table_lowest, easy)

Table:

| ( in thousands ) | 2022 | 2021 | 2020 |
|---|---|---|---|
| total revenues | $ 324719 | $ 360179 | $ 370269 |
| cost of revenues | 55461 | 66221 | 96153 |
| gross profit | 269258 | 293958 | 274116 |
| operating expenses | 185916 | 263562 | 465448 |
| operating income | 83342 | 30396 | -191332 ( 191332 ) |
| interest expense | 1951 | 1699 | 5246 |
| income tax expense | 14189 | 5003 | 0 |
| net income | $ 67202 | $ 23694 | $ -196578 ( 196578 ) |
| capital expenditures | -15586 ( 15586 ) | -28048 ( 28048 ) | -22869 ( 22869 ) |
| net cash provided by operating activities | 96511 | 53619 | -170296 ( 170296 ) |

Text: in 2022 , products contributed $ 135389 thousands with a segment margin of 33.0% ( 33.0 % ), services contributed $ 63612 thousands with a segment margin of 12.1% ( 12.1 % ), licensing contributed $ 125718 thousands with a segment margin of 34.9% ( 34.9 % ) . as of december 31 , 2022 , we had approximately 88400 employees and 655.3 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 80925 thousands in 2022 .

**Q:** what was the lowest net income from 2020 through 2022?
**Program:** `table_min(net income, none)`
**Answer:** -196578.0

- [ ] reviewed: unambiguous and correct

### 3. `synth:synth/utilities/00215:1` (growth, medium)

Table:

| ( in millions ) | 2018 | 2017 |
|---|---|---|
| total revenues | $ 3809 | $ 3856 |
| cost of revenues | 2221 | 2593 |
| gross profit | 1588 | 1263 |
| operating expenses | 574 | 698 |
| operating income | 1014 | 565 |
| net income | $ 844 | $ 390 |
| capital expenditures | -749 ( 749 ) | -718 ( 718 ) |

Text: in 2018 , consumer contributed $ 1523 millions with a segment margin of 38.7% ( 38.7 % ), commercial contributed $ 1127 millions with a segment margin of 7.0% ( 7.0 % ), government contributed $ 1159 millions with a segment margin of 26.7% ( 26.7 % ) . as of december 31 , 2018 , we had approximately 4882 employees and 507.2 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 532 millions in 2018 .

**Q:** what was the percentage change in gross profit from 2017 to 2018?
**Program:** `subtract(1588, 1263), divide(#0, 1263)`
**Answer:** 0.25732

- [ ] reviewed: unambiguous and correct

### 4. `synth:synth/biotech/00020:2` (multi_step_negative, medium)

Table:

| ( in thousands ) | 2021 | 2020 |
|---|---|---|
| total revenues | $ 329057 | $ 365531 |
| cost of revenues | 80873 | 123342 |
| operating expenses | 206621 | 231286 |
| operating income | 41563 | 10903 |
| interest expense | 8643 | 5966 |
| income tax expense | 5742 | 861 |
| net income | $ 27178 | $ 4076 |
| capital expenditures | -24561 ( 24561 ) | -27609 ( 27609 ) |
| net cash provided by operating activities | 56968 | 37569 |

Text: in 2021 , consumer contributed $ 65340 thousands with a segment margin of 27.2% ( 27.2 % ), commercial contributed $ 198934 thousands with a segment margin of 34.8% ( 34.8 % ), government contributed $ 64783 thousands with a segment margin of 30.2% ( 30.2 % ) . as of december 31 , 2021 , we had approximately 45980 employees and 856.7 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 32407 thousands in 2021 .

**Q:** what was free cash flow as a percentage of total revenues in 2021?
**Program:** `add(56968, -24561), divide(#0, 329057)`
**Answer:** 0.09848

- [ ] reviewed: unambiguous and correct

### 5. `synth:synth/utilities/00132:2` (cagr, medium)

Table:

| ( in thousands ) | 2019 | 2018 | 2017 |
|---|---|---|---|
| total revenues | $ 928246 | $ 919209 | $ 879165 |
| cost of revenues | 513588 | 518738 | 448248 |
| gross profit | 414658 | 400471 | 430917 |
| operating expenses | 96148 | 105664 | 107329 |
| operating income | 318510 | 294807 | 323588 |
| income tax expense | 68522 | 62492 | 73976 |
| net income | $ 228540 | $ 208427 | $ 246728 |
| capital expenditures | -269109 ( 269109 ) | -260889 ( 260889 ) | -262044 ( 262044 ) |
| net cash provided by operating activities | 308695 | 260077 | 283337 |

Text: in 2019 , consumer contributed $ 186320 thousands with a segment margin of 17.4% ( 17.4 % ), commercial contributed $ 245874 thousands with a segment margin of 6.2% ( 6.2 % ), government contributed $ 496052 thousands with a segment margin of 43.3% ( 43.3 % ) . as of december 31 , 2019 , we had approximately 84382 employees and 842.8 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 39586 thousands in 2019 .

**Q:** what was the compound annual growth rate of total revenues from 2017 to 2019?
**Program:** `divide(928246, 879165), exp(#0, 0.5), subtract(#1, const_1)`
**Answer:** 0.02753

- [ ] reviewed: unambiguous and correct

### 6. `synth:synth/retail/00261:3` (comparison, hard)

Table:

| ( in thousands ) | 2018 | 2017 | 2016 |
|---|---|---|---|
| total revenues | $ 283913 | $ 282948 | $ 253533 |
| cost of revenues | 216482 | 188774 | 153043 |
| gross profit | 67431 | 94174 | 100490 |
| operating expenses | 57034 | 66120 | 46937 |
| operating income | 10397 | 28054 | 53553 |
| interest expense | 3635 | 7126 | 4271 |
| income tax expense | 1076 | 3329 | 7839 |
| net income | $ 5686 | $ 17599 | $ 41443 |
| capital expenditures | -8109 ( 8109 ) | -11706 ( 11706 ) | -11139 ( 11139 ) |
| net cash provided by operating activities | 24556 | 41334 | 64770 |

Text: in 2018 , consumer contributed $ 67704 thousands with a segment margin of 30.9% ( 30.9 % ), commercial contributed $ 121577 thousands with a segment margin of 39.4% ( 39.4 % ), government contributed $ 94632 thousands with a segment margin of 23.5% ( 23.5 % ) . as of december 31 , 2018 , we had approximately 88954 employees and 326.3 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 16447 thousands in 2018 .

**Q:** did income tax expense grow faster than net cash provided by operating activities from 2017 to 2018?
**Program:** `subtract(1076, 3329), divide(#0, 3329), subtract(24556, 41334), divide(#2, 41334), greater(#1, #3)`
**Answer:** no

- [ ] reviewed: unambiguous and correct

### 7. `synth:synth/retail/00248:3` (ratio, easy)

Table:

| ( in millions ) | 2017 | 2016 | 2015 |
|---|---|---|---|
| total revenues | $ 7201 | $ 7558 | $ 7015 |
| cost of revenues | 4281 | 4773 | 5115 |
| gross profit | 2920 | 2785 | 1900 |
| operating expenses | 1398 | 1168 | 1674 |
| operating income | 1522 | 1617 | 226 |
| interest expense | 20 | 171 | 80 |
| income tax expense | 368 | 355 | 36 |
| net income | $ 1134 | $ 1091 | $ 110 |
| capital expenditures | -344 ( 344 ) | -162 ( 162 ) | -156 ( 156 ) |

Text: in 2017 , consumer contributed $ 3006 millions with a segment margin of 33.6% ( 33.6 % ), commercial contributed $ 3215 millions with a segment margin of 14.6% ( 14.6 % ), government contributed $ 980 millions with a segment margin of 20.6% ( 20.6 % ) . as of december 31 , 2017 , we had approximately 48047 employees and 835.2 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 1453 millions in 2017 .

**Q:** in 2016 , cost of revenues was what portion of total revenues?
**Program:** `divide(4773, 7558)`
**Answer:** 0.63152

- [ ] reviewed: unambiguous and correct

### 8. `synth:synth/retail/00207:1` (margin_change, medium)

Table:

| ( in thousands ) | 2022 | 2021 |
|---|---|---|
| total revenues | $ 272697 | $ 290335 |
| cost of revenues | 183061 | 192367 |
| gross profit | 89636 | 97968 |
| operating income | 42092 | 11056 |
| interest expense | 5171 | 5282 |
| income tax expense | 8849 | 1384 |
| net income | $ 28072 | $ 4390 |
| capital expenditures | -5459 ( 5459 ) | -9762 ( 9762 ) |
| net cash provided by operating activities | 36598 | 14885 |

Text: in 2022 , north america contributed $ 117328 thousands with a segment margin of 44.1% ( 44.1 % ), europe contributed $ 97966 thousands with a segment margin of 34.1% ( 34.1 % ), asia pacific contributed $ 57403 thousands with a segment margin of 35.7% ( 35.7 % ) . as of december 31 , 2022 , we had approximately 37202 employees and 357.6 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 31139 thousands in 2022 .

**Q:** what was the change in gross profit as a percentage of total revenues from 2021 to 2022?
**Program:** `divide(89636, 272697), divide(97968, 290335), subtract(#0, #1)`
**Answer:** -0.00873

- [ ] reviewed: unambiguous and correct

### 9. `synth:synth/biotech/00155:1` (comparison, hard)

Table:

| ( in thousands ) | 2018 | 2017 | 2016 |
|---|---|---|---|
| total revenues | $ 1182410 | $ 824717 | $ 715056 |
| cost of revenues | 494346 | 349997 | 299913 |
| operating expenses | 875917 | 599572 | 707949 |
| operating income | -187853 ( 187853 ) | -124852 ( 124852 ) | -292806 ( 292806 ) |
| interest expense | 10236 | 14889 | 13872 |
| income tax expense | 0 | 0 | 0 |
| net income | $ -198089 ( 198089 ) | $ -139741 ( 139741 ) | $ -306678 ( 306678 ) |
| capital expenditures | -74622 ( 74622 ) | -56219 ( 56219 ) | -38849 ( 38849 ) |
| net cash provided by operating activities | -171128 ( 171128 ) | -95924 ( 95924 ) | -247547 ( 247547 ) |

Text: in 2018 , products contributed $ 544484 thousands with a segment margin of 6.4% ( 6.4 % ), services contributed $ 305846 thousands with a segment margin of 33.4% ( 33.4 % ), licensing contributed $ 332080 thousands with a segment margin of 13.9% ( 13.9 % ) . as of december 31 , 2018 , we had approximately 85474 employees and 576.7 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ -245750 thousands in 2018 .

**Q:** did cost of revenues grow faster than total revenues from 2017 to 2018?
**Program:** `subtract(494346, 349997), divide(#0, 349997), subtract(1182410, 824717), divide(#2, 824717), greater(#1, #3)`
**Answer:** no

- [ ] reviewed: unambiguous and correct

### 10. `synth:synth/utilities/00244:0` (cagr, medium)

Table:

| ( in thousands ) | 2022 | 2021 | 2020 |
|---|---|---|---|
| total revenues | $ 318327 | $ 325872 | $ 331104 |
| operating expenses | 52310 | 40294 | 35409 |
| operating income | 91630 | 113853 | 95728 |
| interest expense | 3809 | 9003 | 8784 |
| income tax expense | 20591 | 24583 | 20385 |
| net income | $ 67230 | $ 80267 | $ 66559 |
| capital expenditures | -78839 ( 78839 ) | -91579 ( 91579 ) | -83781 ( 83781 ) |
| net cash provided by operating activities | 76039 | 117850 | 93266 |

Text: in 2022 , products contributed $ 145849 thousands with a segment margin of 42.9% ( 42.9 % ), services contributed $ 88688 thousands with a segment margin of 42.3% ( 42.3 % ), licensing contributed $ 83790 thousands with a segment margin of 35.5% ( 35.5 % ) . as of december 31 , 2022 , we had approximately 84427 employees and 633.9 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ -2800 thousands in 2022 .

**Q:** what is the two-year cagr of total revenues ending in 2022?
**Program:** `divide(318327, 331104), exp(#0, 0.5), subtract(#1, const_1)`
**Answer:** -0.01948

- [ ] reviewed: unambiguous and correct

### 11. `synth:synth/software/00183:1` (weighted_average, hard)

Table:

| ( in thousands ) | 2017 | 2016 | 2015 |
|---|---|---|---|
| total revenues | $ 724175 | $ 614283 | $ 464165 |
| cost of revenues | 207583 | 122299 | 90748 |
| gross profit | 516592 | 491984 | 373417 |
| operating expenses | 257191 | 265924 | 198748 |
| operating income | 259401 | 226060 | 174669 |
| interest expense | 6060 | 10683 | 5321 |
| income tax expense | 43961 | 37373 | 29386 |
| net income | $ 209380 | $ 178004 | $ 139962 |
| capital expenditures | -36214 ( 36214 ) | -16017 ( 16017 ) | -15506 ( 15506 ) |
| net cash provided by operating activities | 288502 | 237999 | 165540 |

Text: in 2017 , products contributed $ 243219 thousands with a segment margin of 19.8% ( 19.8 % ), services contributed $ 216247 thousands with a segment margin of 18.4% ( 18.4 % ), licensing contributed $ 264709 thousands with a segment margin of 5.4% ( 5.4 % ) . as of december 31 , 2017 , we had approximately 76907 employees and 898.5 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 252288 thousands in 2017 .

**Q:** what was the revenue-weighted average segment margin across all segments in 2017?
**Program:** `multiply(243219, 19.8%), multiply(216247, 18.4%), multiply(264709, 5.4%), add(#0, #1), add(#3, #2), add(243219, 216247), add(#5, 264709), divide(#4, #6)`
**Answer:** 0.14118

- [ ] reviewed: unambiguous and correct

### 12. `synth:synth/retail/00298:2` (multi_step_negative, medium)

Table:

| ( in millions ) | 2023 | 2022 |
|---|---|---|
| total revenues | $ 3599 | $ 3733 |
| cost of revenues | 2563 | 2841 |
| gross profit | 1036 | 892 |
| operating expenses | 1076 | 723 |
| operating income | -40 ( 40 ) | 169 |
| interest expense | 53 | 95 |
| income tax expense | 0 | 13 |
| net income | $ -93 ( 93 ) | $ 61 |
| capital expenditures | -175 ( 175 ) | -156 ( 156 ) |
| net cash provided by operating activities | 40 | 507 |

Text: in 2023 , consumer contributed $ 1105 millions with a segment margin of 12.4% ( 12.4 % ), commercial contributed $ 1423 millions with a segment margin of 29.1% ( 29.1 % ), government contributed $ 1071 millions with a segment margin of 27.1% ( 27.1 % ) . as of december 31 , 2023 , we had approximately 47840 employees and 832.4 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ -135 millions in 2023 .

**Q:** what was free cash flow as a percentage of total revenues in 2023?
**Program:** `add(40, -175), divide(#0, 3599)`
**Answer:** -0.03751

- [ ] reviewed: unambiguous and correct

### 13. `synth:synth/industrial/00111:3` (margin_change, medium)

Table:

| ( in thousands ) | 2020 | 2019 | 2018 |
|---|---|---|---|
| total revenues | $ 202521 | $ 197129 | $ 177442 |
| cost of revenues | 145967 | 126171 | 138110 |
| gross profit | 56554 | 70958 | 39332 |
| operating expenses | 26832 | 32348 | 27849 |
| operating income | 29722 | 38610 | 11483 |
| interest expense | 4756 | 589 | 3696 |
| income tax expense | 5381 | 8195 | 1678 |
| net income | $ 19585 | $ 29826 | $ 6109 |
| capital expenditures | -10892 ( 10892 ) | -13718 ( 13718 ) | -11340 ( 11340 ) |
| net cash provided by operating activities | 25379 | 50890 | 15930 |

Text: in 2020 , products contributed $ 89899 thousands with a segment margin of 17.5% ( 17.5 % ), services contributed $ 49959 thousands with a segment margin of 42.1% ( 42.1 % ), licensing contributed $ 62663 thousands with a segment margin of 8.3% ( 8.3 % ) . as of december 31 , 2020 , we had approximately 10753 employees and 131.5 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 14487 thousands in 2020 .

**Q:** what was the change in gross profit as a percentage of total revenues from 2019 to 2020?
**Program:** `divide(56554, 202521), divide(70958, 197129), subtract(#0, #1)`
**Answer:** -0.08071

- [ ] reviewed: unambiguous and correct

### 14. `synth:synth/utilities/00258:1` (ratio, easy)

Table:

| ( in thousands ) | 2020 | 2019 | 2018 |
|---|---|---|---|
| total revenues | $ 39772 | $ 38219 | $ 39139 |
| cost of revenues | 24850 | 20830 | 26884 |
| gross profit | 14922 | 17389 | 12255 |
| operating expenses | 7290 | 5117 | 6894 |
| operating income | 7632 | 12272 | 5361 |
| interest expense | 1183 | 801 | 520 |
| income tax expense | 1835 | 3263 | 1377 |
| net income | $ 4614 | $ 8208 | $ 3464 |
| capital expenditures | -7642 ( 7642 ) | -7022 ( 7022 ) | -8641 ( 8641 ) |
| net cash provided by operating activities | 6750 | 12426 | 5705 |

Text: in 2020 , consumer contributed $ 17595 thousands with a segment margin of 9.0% ( 9.0 % ), commercial contributed $ 7645 thousands with a segment margin of 30.0% ( 30.0 % ), government contributed $ 14532 thousands with a segment margin of 25.9% ( 25.9 % ) . as of december 31 , 2020 , we had approximately 79597 employees and 527.7 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ -892 thousands in 2020 .

**Q:** in 2018 , operating expenses was what portion of gross profit?
**Program:** `divide(6894, 12255)`
**Answer:** 0.56255

- [ ] reviewed: unambiguous and correct

### 15. `synth:synth/industrial/00071:1` (average, medium)

Table:

| ( in thousands ) | 2016 | 2015 | 2014 |
|---|---|---|---|
| total revenues | $ 770985 | $ 747291 | $ 717000 |
| cost of revenues | 613252 | 596126 | 448757 |
| gross profit | 157733 | 151165 | 268243 |
| operating expenses | 81311 | 75572 | 66196 |
| operating income | 76422 | 75593 | 202047 |
| income tax expense | 9661 | 10891 | 33057 |
| net income | $ 48641 | $ 54833 | $ 166437 |
| capital expenditures | -33090 ( 33090 ) | -39518 ( 39518 ) | -47486 ( 47486 ) |
| net cash provided by operating activities | 123776 | 113198 | 238282 |

Text: in 2016 , products contributed $ 291632 thousands with a segment margin of 6.9% ( 6.9 % ), services contributed $ 240407 thousands with a segment margin of 25.8% ( 25.8 % ), licensing contributed $ 238946 thousands with a segment margin of 22.7% ( 22.7 % ) . as of december 31 , 2016 , we had approximately 89702 employees and 648.4 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 90686 thousands in 2016 .

**Q:** what was the average operating income for 2015 and 2016?
**Program:** `add(76422, 75593), divide(#0, const_2)`
**Answer:** 76007.5

- [ ] reviewed: unambiguous and correct

### 16. `synth:synth/software/00144:1` (margin_change, medium)

Table:

| ( in millions ) | 2024 | 2023 | 2022 |
|---|---|---|---|
| total revenues | $ 8625 | $ 7427 | $ 6036 |
| cost of revenues | 2462 | 1693 | 2026 |
| gross profit | 6163 | 5734 | 4010 |
| operating income | 2100 | 2107 | 489 |
| interest expense | 189 | 146 | 86 |
| income tax expense | 452 | 464 | 95 |
| net income | $ 1459 | $ 1497 | $ 308 |
| capital expenditures | -255 ( 255 ) | -380 ( 380 ) | -316 ( 316 ) |
| net cash provided by operating activities | 2101 | 2053 | 787 |

Text: in 2024 , north america contributed $ 1624 millions with a segment margin of 41.2% ( 41.2 % ), europe contributed $ 3905 millions with a segment margin of 13.3% ( 13.3 % ), asia pacific contributed $ 3096 millions with a segment margin of 17.9% ( 17.9 % ) . as of december 31 , 2024 , we had approximately 78148 employees and 415.9 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 1846 millions in 2024 .

**Q:** by how many percentage points did the ratio of gross profit to total revenues change between 2023 and 2024?
**Program:** `divide(6163, 8625), divide(5734, 7427), subtract(#0, #1)`
**Answer:** -0.0575

- [ ] reviewed: unambiguous and correct

### 17. `synth:synth/industrial/00071:2` (unit_conversion, medium)

Table:

| ( in thousands ) | 2016 | 2015 | 2014 |
|---|---|---|---|
| total revenues | $ 770985 | $ 747291 | $ 717000 |
| cost of revenues | 613252 | 596126 | 448757 |
| gross profit | 157733 | 151165 | 268243 |
| operating expenses | 81311 | 75572 | 66196 |
| operating income | 76422 | 75593 | 202047 |
| income tax expense | 9661 | 10891 | 33057 |
| net income | $ 48641 | $ 54833 | $ 166437 |
| capital expenditures | -33090 ( 33090 ) | -39518 ( 39518 ) | -47486 ( 47486 ) |
| net cash provided by operating activities | 123776 | 113198 | 238282 |

Text: in 2016 , products contributed $ 291632 thousands with a segment margin of 6.9% ( 6.9 % ), services contributed $ 240407 thousands with a segment margin of 25.8% ( 25.8 % ), licensing contributed $ 238946 thousands with a segment margin of 22.7% ( 22.7 % ) . as of december 31 , 2016 , we had approximately 89702 employees and 648.4 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 90686 thousands in 2016 .

**Q:** what was net income per diluted share in 2016 ( net income is reported in thousands )?
**Program:** `divide(48641, const_1000), divide(#0, 648.4)`
**Answer:** 0.07502

- [ ] reviewed: unambiguous and correct

### 18. `synth:synth/industrial/00386:3` (table_average, easy)

Table:

| ( in thousands ) | 2023 | 2022 | 2021 |
|---|---|---|---|
| total revenues | $ 380207 | $ 344017 | $ 381445 |
| cost of revenues | 299889 | 222777 | 260285 |
| gross profit | 80318 | 121240 | 121160 |
| operating expenses | 41248 | 37779 | 45480 |
| operating income | 39070 | 83461 | 75680 |
| interest expense | 9553 | 5569 | 3344 |
| income tax expense | 7600 | 20055 | 18624 |
| net income | $ 21917 | $ 57837 | $ 53712 |
| capital expenditures | -22686 ( 22686 ) | -28322 ( 28322 ) | -29000 ( 29000 ) |
| net cash provided by operating activities | 61604 | 88756 | 67940 |

Text: in 2023 , north america contributed $ 117453 thousands with a segment margin of 30.2% ( 30.2 % ), europe contributed $ 191263 thousands with a segment margin of 41.5% ( 41.5 % ), asia pacific contributed $ 71491 thousands with a segment margin of 34.3% ( 34.3 % ) . as of december 31 , 2023 , we had approximately 67164 employees and 102.3 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 38918 thousands in 2023 .

**Q:** what was the average operating income from 2021 through 2023?
**Program:** `table_average(operating income, none)`
**Answer:** 66070.33333

- [ ] reviewed: unambiguous and correct

### 19. `synth:synth/retail/00048:2` (unit_conversion, medium)

Table:

| ( in thousands ) | 2022 | 2021 | 2020 |
|---|---|---|---|
| total revenues | $ 300771 | $ 320802 | $ 340875 |
| cost of revenues | 226720 | 249228 | 211091 |
| gross profit | 74051 | 71574 | 129784 |
| operating income | -9807 ( 9807 ) | 4867 | 54109 |
| interest expense | 3914 | 7116 | 4708 |
| income tax expense | 0 | 0 | 12414 |
| net income | $ -13721 ( 13721 ) | $ -2249 ( 2249 ) | $ 36987 |
| capital expenditures | -8689 ( 8689 ) | -8056 ( 8056 ) | -9666 ( 9666 ) |
| net cash provided by operating activities | -3963 ( 3963 ) | 25622 | 55305 |

Text: in 2022 , north america contributed $ 60001 thousands with a segment margin of 38.9% ( 38.9 % ), europe contributed $ 120526 thousands with a segment margin of 14.7% ( 14.7 % ), asia pacific contributed $ 120244 thousands with a segment margin of 40.3% ( 40.3 % ) . as of december 31 , 2022 , we had approximately 35078 employees and 531.5 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ -12652 thousands in 2022 .

**Q:** based on weighted average diluted shares , what were 2022 earnings per share?
**Program:** `divide(-13721, const_1000), divide(#0, 531.5)`
**Answer:** -0.02582

- [ ] reviewed: unambiguous and correct

### 20. `synth:synth/industrial/00316:2` (weighted_average, hard)

Table:

| ( in millions ) | 2016 | 2015 |
|---|---|---|
| total revenues | $ 1556 | $ 1569 |
| cost of revenues | 966 | 1057 |
| gross profit | 590 | 512 |
| operating expenses | 171 | 303 |
| operating income | 419 | 209 |
| interest expense | 15 | 15 |
| income tax expense | 87 | 42 |
| net income | $ 317 | $ 152 |
| capital expenditures | -74 ( 74 ) | -64 ( 64 ) |
| net cash provided by operating activities | 363 | 312 |

Text: in 2016 , consumer contributed $ 435 millions with a segment margin of 22.1% ( 22.1 % ), commercial contributed $ 362 millions with a segment margin of 25.9% ( 25.9 % ), government contributed $ 759 millions with a segment margin of 11.9% ( 11.9 % ) . as of december 31 , 2016 , we had approximately 65683 employees and 418.4 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 289 millions in 2016 .

**Q:** weighting each segment by its 2016 revenue , what is the average segment margin?
**Program:** `multiply(435, 22.1%), multiply(362, 25.9%), multiply(759, 11.9%), add(#0, #1), add(#3, #2), add(435, 362), add(#5, 759), divide(#4, #6)`
**Answer:** 0.18009

- [ ] reviewed: unambiguous and correct

### 21. `synth:synth/biotech/00128:1` (weighted_average, hard)

Table:

| ( in millions ) | 2016 | 2015 | 2014 |
|---|---|---|---|
| total revenues | $ 10655 | $ 8767 | $ 8788 |
| cost of revenues | 2121 | 2461 | 3564 |
| gross profit | 8534 | 6306 | 5224 |
| operating income | 1060 | 1217 | -4513 ( 4513 ) |
| interest expense | 45 | 42 | 247 |
| income tax expense | 201 | 233 | 0 |
| net income | $ 814 | $ 942 | $ -4760 ( 4760 ) |
| capital expenditures | -528 ( 528 ) | -565 ( 565 ) | -482 ( 482 ) |
| net cash provided by operating activities | 1410 | 1158 | -4468 ( 4468 ) |

Text: in 2016 , north america contributed $ 4619 millions with a segment margin of 22.8% ( 22.8 % ), europe contributed $ 2476 millions with a segment margin of 40.7% ( 40.7 % ), asia pacific contributed $ 3560 millions with a segment margin of 29.4% ( 29.4 % ) . as of december 31 , 2016 , we had approximately 61760 employees and 346.6 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 882 millions in 2016 .

**Q:** what was the revenue-weighted average segment margin across all segments in 2016?
**Program:** `multiply(4619, 22.8%), multiply(2476, 40.7%), multiply(3560, 29.4%), add(#0, #1), add(#3, #2), add(4619, 2476), add(#5, 3560), divide(#4, #6)`
**Answer:** 0.29165

- [ ] reviewed: unambiguous and correct

### 22. `synth:synth/industrial/00272:2` (margin_change, medium)

Table:

| ( in thousands ) | 2020 | 2019 | 2018 |
|---|---|---|---|
| total revenues | $ 833735 | $ 838566 | $ 807115 |
| cost of revenues | 632820 | 629562 | 514973 |
| operating expenses | 138216 | 161372 | 67804 |
| operating income | 62699 | 47632 | 224338 |
| interest expense | 14335 | 19259 | 9541 |
| income tax expense | 13824 | 8110 | 61394 |
| net income | $ 34540 | $ 20263 | $ 153403 |
| capital expenditures | -74483 ( 74483 ) | -37607 ( 37607 ) | -56524 ( 56524 ) |
| net cash provided by operating activities | 118400 | 109708 | 191554 |

Text: in 2020 , north america contributed $ 238775 thousands with a segment margin of 35.0% ( 35.0 % ), europe contributed $ 317712 thousands with a segment margin of 7.6% ( 7.6 % ), asia pacific contributed $ 277248 thousands with a segment margin of 35.6% ( 35.6 % ) . as of december 31 , 2020 , we had approximately 82533 employees and 275.0 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 43917 thousands in 2020 .

**Q:** by how many percentage points did the ratio of operating income to total revenues change between 2019 and 2020?
**Program:** `divide(62699, 833735), divide(47632, 838566), subtract(#0, #1)`
**Answer:** 0.0184

- [ ] reviewed: unambiguous and correct

### 23. `synth:synth/utilities/00361:0` (unit_conversion, medium)

Table:

| ( in thousands ) | 2019 | 2018 |
|---|---|---|
| total revenues | $ 212409 | $ 202656 |
| cost of revenues | 130811 | 141277 |
| operating expenses | 26546 | 38637 |
| operating income | 55052 | 22742 |
| interest expense | 1338 | 862 |
| income tax expense | 12595 | 5131 |
| net income | $ 41119 | $ 16749 |
| capital expenditures | -37843 ( 37843 ) | -54474 ( 54474 ) |
| net cash provided by operating activities | 60049 | 39980 |

Text: in 2019 , products contributed $ 69904 thousands with a segment margin of 29.5% ( 29.5 % ), services contributed $ 90198 thousands with a segment margin of 15.8% ( 15.8 % ), licensing contributed $ 52307 thousands with a segment margin of 41.8% ( 41.8 % ) . as of december 31 , 2019 , we had approximately 51316 employees and 410.7 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 22206 thousands in 2019 .

**Q:** based on weighted average diluted shares , what were 2019 earnings per share?
**Program:** `divide(41119, const_1000), divide(#0, 410.7)`
**Answer:** 0.10012

- [ ] reviewed: unambiguous and correct

### 24. `synth:synth/software/00308:0` (weighted_average, hard)

Table:

| ( in millions ) | 2016 | 2015 | 2014 |
|---|---|---|---|
| total revenues | $ 5758 | $ 5533 | $ 4389 |
| cost of revenues | 1485 | 1202 | 1375 |
| operating expenses | 3116 | 3112 | 2619 |
| operating income | 1157 | 1219 | 395 |
| interest expense | 159 | 18 | 92 |
| income tax expense | 210 | 253 | 64 |
| net income | $ 788 | $ 948 | $ 239 |
| capital expenditures | -139 ( 139 ) | -129 ( 129 ) | -234 ( 234 ) |
| net cash provided by operating activities | 1000 | 1069 | 757 |

Text: in 2016 , north america contributed $ 1582 millions with a segment margin of 27.9% ( 27.9 % ), europe contributed $ 2440 millions with a segment margin of 34.3% ( 34.3 % ), asia pacific contributed $ 1736 millions with a segment margin of 17.2% ( 17.2 % ) . as of december 31 , 2016 , we had approximately 89955 employees and 149.1 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 861 millions in 2016 .

**Q:** what was the revenue-weighted average segment margin across all segments in 2016?
**Program:** `multiply(1582, 27.9%), multiply(2440, 34.3%), multiply(1736, 17.2%), add(#0, #1), add(#3, #2), add(1582, 2440), add(#5, 1736), divide(#4, #6)`
**Answer:** 0.27386

- [ ] reviewed: unambiguous and correct

### 25. `synth:synth/biotech/00075:0` (margin_change, medium)

Table:

| ( in thousands ) | 2017 | 2016 | 2015 |
|---|---|---|---|
| total revenues | $ 239976 | $ 229452 | $ 163513 |
| cost of revenues | 99912 | 61774 | 29488 |
| gross profit | 140064 | 167678 | 134025 |
| operating expenses | 132945 | 140848 | 219295 |
| operating income | 7119 | 26830 | -85270 ( 85270 ) |
| interest expense | 4342 | 2281 | 607 |
| income tax expense | 655 | 5792 | 0 |
| net income | $ 2122 | $ 18757 | $ -85877 ( 85877 ) |
| capital expenditures | -17227 ( 17227 ) | -8121 ( 8121 ) | -7036 ( 7036 ) |
| net cash provided by operating activities | 24043 | 36220 | -72377 ( 72377 ) |

Text: in 2017 , north america contributed $ 83770 thousands with a segment margin of 6.2% ( 6.2 % ), europe contributed $ 82262 thousands with a segment margin of 34.1% ( 34.1 % ), asia pacific contributed $ 73944 thousands with a segment margin of 29.3% ( 29.3 % ) . as of december 31 , 2017 , we had approximately 88540 employees and 723.4 million weighted average diluted shares outstanding . free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ 6816 thousands in 2017 .

**Q:** by how many percentage points did the ratio of gross profit to total revenues change between 2016 and 2017?
**Program:** `divide(140064, 239976), divide(167678, 229452), subtract(#0, #1)`
**Answer:** -0.14712

- [ ] reviewed: unambiguous and correct

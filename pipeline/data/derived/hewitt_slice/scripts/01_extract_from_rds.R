# 01_extract_from_rds.R
# Extract machine-usable tables from the Hewitt et al. (2026) Code Ocean capsule RDS files.
# - cell_stats.csv:        per (study, outcome, condition) participant-level n / mean / sd on the ORIGINAL y scale
# - conditions_by_study.csv: per (study, condition) participant counts
# - hypotheses_long.csv:   RA_hypotheses unnested to (study, outcome, hypothesis, condition, t_hypothesis)
# - outcome_scales.csv:    per (study, outcome) llm-prompt scale metadata incl. scale_flip
# - prompt_pages.csv:      gpt-4 prompts split into survey "pages", demographic persona pages dropped,
#                          deduplicated to unique page-sequence variants per (study, condition, outcome)
# Run:  LC_ALL=en_US.UTF-8 Rscript 01_extract_from_rds.R <raw_dir> <out_dir>
# No LLM calls; local wrangling only.

Sys.setlocale("LC_ALL", "en_US.UTF-8")
args <- commandArgs(trailingOnly = TRUE)
raw_dir <- if (length(args) >= 1) args[1] else "/Users/jamie/git/silicon-sample-benchmark/pipeline/data/raw/hewitt2026_llm_prediction/codeocean_capsule/data"
out_dir <- if (length(args) >= 2) args[2] else "/Users/jamie/git/silicon-sample-benchmark/pipeline/data/derived/hewitt_slice/intermediate"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

wcsv <- function(df, name) {
  f <- file.path(out_dir, name)
  con <- file(f, open = "w", encoding = "UTF-8")
  utils::write.csv(df, con, row.names = FALSE, na = "")
  close(con)
  cat("wrote", f, ":", nrow(df), "rows\n")
}

## ---------- 1. rct_responses -> cell stats (original y units) ----------
rct <- readRDS(file.path(raw_dir, "rct_responses.RDS"))
cell_stats <- do.call(rbind, lapply(seq_len(nrow(rct)), function(i) {
  d <- rct$data[[i]]
  y <- suppressWarnings(as.numeric(d$y))
  sp <- split(y, as.character(d$condition.name))
  data.frame(
    study = rct$study[i],
    outcome.variable = rct$outcome.variable[i],
    outcome.name = rct$outcome.name[i],
    outcome.min = rct$outcome.min[i],
    outcome.max = rct$outcome.max[i],
    condition.name = names(sp),
    n_total = vapply(sp, length, 1L),
    n = vapply(sp, function(v) sum(!is.na(v)), 1L),
    mean = vapply(sp, function(v) mean(v, na.rm = TRUE), 1),
    sd = vapply(sp, function(v) stats::sd(v, na.rm = TRUE), 1),
    stringsAsFactors = FALSE
  )
}))
rownames(cell_stats) <- NULL
wcsv(cell_stats, "cell_stats.csv")

conds <- aggregate(n ~ study + condition.name, cell_stats, sum)
wcsv(conds[order(conds$study, conds$condition.name), ], "conditions_by_study.csv")

## ---------- 2. RA_hypotheses -> long ----------
hyp <- readRDS(file.path(raw_dir, "RA_hypotheses.RDS"))
hyp_long <- do.call(rbind, lapply(seq_len(nrow(hyp)), function(i) {
  hd <- as.data.frame(hyp$hypothesis_data[[i]])
  data.frame(study = hyp$study[i], outcome.name = hyp$outcome.name[i],
             hypothesis = hyp$hypothesis[i],
             condition.name = hd$condition.name, t_hypothesis = hd$t_hypothesis,
             stringsAsFactors = FALSE)
}))
wcsv(hyp_long, "hypotheses_long.csv")

## ---------- 3. llm_responses (gpt-4 subset) -> scales + prompt pages ----------
llm <- readRDS(file.path(raw_dir, "llm_responses.RDS"))
g4 <- llm[llm$model == "gpt-4",
          c("study", "condition.name", "outcome.name", "prompt",
            "scale_flip", "outcome_scale_min", "outcome_scale_max")]
rm(llm); invisible(gc())

sc <- unique(g4[, c("study", "outcome.name", "scale_flip", "outcome_scale_min", "outcome_scale_max")])
wcsv(sc[order(sc$study, sc$outcome.name), ], "outcome_scales.csv")

# Split prompts into pages. Page delimiters:
#   "The first page of the survey says:\n" / "The next page of the survey says:\n"
# The 6 persona/demographic pages (identical battery in every prompt) are dropped.
demo_qs <- c("> Are you liberal, moderate or conservative?",
             "> How old are you?",
             "> What is your ethnicity?",
             "> What is your gender?",
             "> What is the maximum level of education you have attained?",
             "> What is your partisan affiliation?")

split_pages <- function(p) {
  # normalize the first-page marker so one split pattern suffices
  p <- gsub("The first page of the survey says:\n", "The next page of the survey says:\n", p, fixed = TRUE)
  parts <- strsplit(p, "The next page of the survey says:\n", fixed = TRUE)[[1]]
  if (length(parts) < 2) return(NULL)
  pages <- trimws(parts[-1])
  first_line <- vapply(strsplit(pages, "\n", fixed = TRUE), `[`, "", 1)
  keep <- !(first_line %in% demo_qs)
  pages[keep]
}

cells <- unique(g4[, c("study", "condition.name", "outcome.name")])
cat("unique cells:", nrow(cells), "\n")

page_rows <- vector("list", nrow(cells))
key_all <- paste(g4$study, g4$condition.name, g4$outcome.name, sep = "\r")
for (i in seq_len(nrow(cells))) {
  k <- paste(cells$study[i], cells$condition.name[i], cells$outcome.name[i], sep = "\r")
  prompts <- g4$prompt[key_all == k]
  sigs <- vapply(prompts, function(p) paste(split_pages(p), collapse = "\n\x1e\n"), "", USE.NAMES = FALSE)
  tab <- table(sigs)
  variants <- names(tab)
  rows <- lapply(seq_along(variants), function(vi) {
    pages <- strsplit(variants[vi], "\n\x1e\n", fixed = TRUE)[[1]]
    data.frame(study = cells$study[i], condition.name = cells$condition.name[i],
               outcome.name = cells$outcome.name[i],
               variant = vi, variant_n_prompts = as.integer(tab[[variants[vi]]]),
               page_idx = seq_along(pages), page_text = pages, stringsAsFactors = FALSE)
  })
  page_rows[[i]] <- do.call(rbind, rows)
  if (i %% 100 == 0) cat("  ...", i, "cells\n")
}
pp <- do.call(rbind, page_rows)
cat("page rows:", nrow(pp), "; cells with >1 variant:",
    length(unique(paste(pp$study, pp$condition.name, pp$outcome.name)[pp$variant > 1])), "\n")
wcsv(pp, "prompt_pages.csv")

# Also save the fixed prompt preamble (before the first page) from one prompt, for reference
pre <- strsplit(g4$prompt[1], "The first page of the survey says:", fixed = TRUE)[[1]][1]
writeLines(pre, file.path(out_dir, "prompt_preamble.txt"))
cat("done\n")

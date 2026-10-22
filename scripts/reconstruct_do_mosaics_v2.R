# Layer 7 / H7 v2: founder-mosaic reconstruction over the pre-declared minprob ladder.
#
# Draft protocol: docs/LAYER7_H7v2_PROTOCOL.md. Applied identically to the observed mice and to the
# painted simulated cohort, which is the whole point of v2.
#
#   & "D:\R\app\bin\Rscript.exe" scripts/reconstruct_do_mosaics_v2.R --cross=<cross json> --out=<dir> [--cores=8]
#
# calc_genoprob depends on no ladder value, so it runs once; maxmarg, guess_phase and segment
# extraction then run at every rung. For each rung writes segments_mp<rung>.csv (one row per founder
# run) and calls_mp<rung>.json (called and total haplotype-marker cells). Computes no mosaic
# statistic and no comparison between crosses.
#
# The run-length extraction is copied verbatim from scripts/reconstruct_do_mosaics.R, where its
# equivalence to a marker-by-marker walk was verified on 3,593 randomised haplotypes. That file is
# left untouched because it is hashed in the (void) v1 lock.

suppressPackageStartupMessages({
  for (p in c("qtl2", "jsonlite")) {
    if (!requireNamespace(p, quietly = TRUE)) stop(p, " is not installed")
  }
  library(qtl2)
})

ERROR_PROB <- 0.002
MAP_FUNCTION <- "haldane"
LADDER <- c(0.95, 0.90, 0.80, 0.70, 0.60, 0.50)
PHASE_SEED <- 20260912L

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default = NULL) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^--", name, "="), "", hit[1]) else default
}
cross_file <- arg("cross")
out_dir <- arg("out")
cores <- as.integer(arg("cores", "1"))
if (is.null(cross_file) || is.null(out_dir)) stop("usage: --cross=<json> --out=<dir> [--cores=N]")
if (!file.exists(cross_file)) stop("cross file not found: ", cross_file)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- verbatim from scripts/reconstruct_do_mosaics.R ------------------------------------------
segments_for_haplotype <- function(states, pos, id, chr, hap) {
  coded <- ifelse(is.na(states), -1L, as.integer(states))
  r <- rle(coded)
  ends <- cumsum(r$lengths)
  starts <- ends - r$lengths + 1L
  keep <- which(r$values != -1L)
  if (!length(keep)) return(NULL)
  next_start <- ifelse(keep < length(r$values), starts[pmin(keep + 1L, length(r$values))], NA_integer_)
  next_called <- ifelse(keep < length(r$values), r$values[pmin(keep + 1L, length(r$values))] != -1L, FALSE)
  end_idx <- ifelse(!is.na(next_start) & next_called, next_start, ends[keep])
  data.frame(id = id, chr = chr, hap = hap, founder = r$values[keep],
             start_cM = pos[starts[keep]], end_cM = pos[end_idx], closed = TRUE,
             stringsAsFactors = FALSE)
}
# ----------------------------------------------------------------------------------------------

t0 <- Sys.time()
cat("reading cross:", cross_file, "\n"); flush.console()
cross <- read_cross2(cross_file)
cross <- cross[, as.character(1:19)]
cat("individuals:", n_ind(cross), " chromosomes:", n_chr(cross), " cores:", cores, "\n"); flush.console()

# One chromosome at a time. qtl2's HMM treats chromosomes independently, so this gives the same
# genotype probabilities as a whole-genome call while holding one chromosome's probabilities in
# memory instead of all nineteen (a whole-genome attempt pushed commit headroom toward exhaustion).
# Phasing is seeded per chromosome, so every rung and chromosome is reproducible in isolation.
tags <- sprintf("%03d", round(LADDER * 100))
rows <- setNames(lapply(tags, function(x) list()), tags)
called <- setNames(numeric(length(tags)), tags)
total <- setNames(numeric(length(tags)), tags)

for (chr in as.character(1:19)) {
  sub <- cross[, chr]
  pr <- calc_genoprob(sub, sub$gmap, error_prob = ERROR_PROB, map_function = MAP_FUNCTION, cores = cores)
  pos <- sub$gmap[[chr]]
  for (k in seq_along(LADDER)) {
    m <- maxmarg(pr, minprob = LADDER[k], cores = cores)
    set.seed(PHASE_SEED + as.integer(chr))
    arr <- guess_phase(sub, m)[[chr]]
    inds <- rownames(arr)
    total[k] <- total[k] + length(arr)
    called[k] <- called[k] + sum(!is.na(arr))
    for (i in seq_along(inds)) for (h in 1:2) {
      rows[[k]][[length(rows[[k]]) + 1L]] <- segments_for_haplotype(arr[i, , h], pos, inds[i], chr, h)
    }
    rm(m, arr)
  }
  rm(pr, sub); invisible(gc())
  cat("chr", chr, "done at", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
  flush.console()
}

for (k in seq_along(LADDER)) {
  seg <- do.call(rbind, rows[[k]])
  utils::write.csv(seg, file.path(out_dir, paste0("segments_mp", tags[k], ".csv")), row.names = FALSE)
  writeLines(jsonlite::toJSON(list(minprob = LADDER[k], called_cells = called[[k]], total_cells = total[[k]]),
                              auto_unbox = TRUE, pretty = TRUE),
             file.path(out_dir, paste0("calls_mp", tags[k], ".json")))
  cat("minprob", LADDER[k], ":", nrow(seg), "runs, called share", sprintf("%.4f", called[[k]] / total[[k]]), "\n")
}

meta <- list(
  utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  cross = normalizePath(cross_file, winslash = "/"),
  r_version = R.version.string, qtl2_version = as.character(utils::packageVersion("qtl2")),
  error_prob = ERROR_PROB, map_function = MAP_FUNCTION, ladder = LADDER,
  phase_seed = PHASE_SEED, cores = cores,
  n_individuals = n_ind(cross), n_chromosomes = n_chr(cross),
  minutes = round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)
)
writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE), file.path(out_dir, "reconstruction_meta.json"))
cat("all rungs done in", meta$minutes, "min\n")

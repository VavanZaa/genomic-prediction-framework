# Layer 7 / H7: reconstruct observed founder mosaics for the Svenson Diversity Outbred mice.
#
# Pre-registered in docs/LAYER7_H7_PROTOCOL.md. Run with the R at D:\R:
#   & "D:\R\app\bin\Rscript.exe" scripts/reconstruct_do_mosaics.R
#
# Writes output/layer7/observed_segments.csv (one row per founder segment) and
# output/layer7/observed_summary.json (version and QC metadata). Autosomes only; X is excluded
# because its inheritance differs by sex and the simulator does not model sex chromosomes.
#
# This is the observed half of the comparison. It must not be run before the protocol lock exists.

suppressPackageStartupMessages({
  if (!requireNamespace("qtl2", quietly = TRUE)) {
    stop("qtl2 is not installed. Install it first: install.packages('qtl2')")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is not installed. Install it first: install.packages('jsonlite')")
  }
  library(qtl2)
})

ERROR_PROB <- 0.002
MAP_FUNCTION <- "haldane"
MIN_PROB <- 0.95

# Run from the project root; the check below fails loudly rather than reading the wrong files.
root <- getwd()
if (!dir.exists(file.path(root, "data", "real_mice", "svenson"))) {
  stop("run this from the project root (data/real_mice/svenson not found under ", root, ")")
}
cross_file <- file.path(root, "data", "real_mice", "svenson", "svenson.json")
out_dir <- file.path(root, "output", "layer7")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("reading cross:", cross_file, "\n")
cross <- read_cross2(cross_file)
cross <- cross[, c(as.character(1:19))]   # autosomes only
cat("individuals:", n_ind(cross), " chromosomes:", n_chr(cross), "\n")

cat("calc_genoprob (error_prob =", ERROR_PROB, ", map_function =", MAP_FUNCTION, ")\n")
pr <- calc_genoprob(cross, cross$gmap, error_prob = ERROR_PROB, map_function = MAP_FUNCTION)

cat("maxmarg (minprob =", MIN_PROB, ")\n")
m <- maxmarg(pr, minprob = MIN_PROB)

cat("guess_phase\n")
ph <- guess_phase(cross, m)   # per chromosome: ind x marker x 2 haplotypes

# Run-length extraction of founder states along one haplotype.
#
# Conventions, matching the marker-by-marker walk this replaces exactly:
#   - a run of identical founder states becomes one segment;
#   - a segment starts at the position of the run's first marker and ends at the position of the
#     NEXT run's first marker, so segments abut at the crossover interval's right edge;
#   - the final run of a chromosome ends at the last marker position;
#   - uncalled markers (maxmarg below minprob) break a segment, which then ends at the position of
#     the marker before the uncalled run; uncalled stretches emit no segment of their own.
segments_for_haplotype <- function(states, pos, id, chr, hap) {
  coded <- ifelse(is.na(states), -1L, as.integer(states))
  r <- rle(coded)
  ends <- cumsum(r$lengths)
  starts <- ends - r$lengths + 1L
  keep <- which(r$values != -1L)
  if (!length(keep)) return(NULL)
  # a called run ends where the next run begins, unless the next run is uncalled (or none follows),
  # in which case it ends at its own last marker
  next_start <- ifelse(keep < length(r$values), starts[pmin(keep + 1L, length(r$values))], NA_integer_)
  next_called <- ifelse(keep < length(r$values), r$values[pmin(keep + 1L, length(r$values))] != -1L, FALSE)
  end_idx <- ifelse(!is.na(next_start) & next_called, next_start, ends[keep])
  data.frame(id = id, chr = chr, hap = hap, founder = r$values[keep],
             start_cM = pos[starts[keep]], end_cM = pos[end_idx], closed = TRUE,
             stringsAsFactors = FALSE)
}

rows <- list()
n_called <- 0
n_total <- 0
for (chr in names(ph)) {
  arr <- ph[[chr]]
  pos <- cross$gmap[[chr]]                 # cM
  inds <- rownames(arr)
  n_total <- n_total + length(arr)
  n_called <- n_called + sum(!is.na(arr))
  for (i in seq_along(inds)) {
    for (h in 1:2) {
      rows[[length(rows) + 1L]] <- segments_for_haplotype(arr[i, , h], pos, inds[i], chr, h)
    }
  }
  cat("  chr", chr, "done; haplotype blocks so far:", length(rows), "\n")
  flush.console()
}

seg <- do.call(rbind, rows)
seg$length_M <- (seg$end_cM - seg$start_cM) / 100
utils::write.csv(seg, file.path(out_dir, "observed_segments.csv"), row.names = FALSE)

covar <- read.csv(file.path(root, "data", "real_mice", "svenson", "svenson_covar.csv"),
                  stringsAsFactors = FALSE)
meta <- list(
  utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  r_version = R.version.string,
  qtl2_version = as.character(utils::packageVersion("qtl2")),
  error_prob = ERROR_PROB, map_function = MAP_FUNCTION, minprob = MIN_PROB,
  n_individuals = n_ind(cross), n_chromosomes = n_chr(cross),
  n_segments = nrow(seg),
  called_marker_share = n_called / n_total,
  ngen_counts = as.list(table(covar$ngen))
)
writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
           file.path(out_dir, "observed_summary.json"))
cat("wrote", nrow(seg), "segments; called marker share",
    sprintf("%.4f", n_called / n_total), "\n")

# Layer 7 / H7 v3 development: native qtl2 crossover paths with posterior uncertainty.
#
# Writes one native Viterbi count and posterior-draw counts per animal and chromosome.  It never
# turns uncertain markers into custom segments, so uncalled gaps cannot be mistaken for crossovers.

suppressPackageStartupMessages({
  for (p in c("qtl2", "jsonlite")) {
    if (!requireNamespace(p, quietly = TRUE)) stop(p, " is not installed")
  }
  library(qtl2)
})

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default = NULL) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^--", name, "="), "", hit[1]) else default
}

cross_file <- arg("cross")
out_dir <- arg("out")
cores <- as.integer(arg("cores", "1"))
n_draws <- as.integer(arg("draws", "16"))
error_prob <- as.numeric(arg("error-prob", "0.002"))
map_function <- arg("map-function", "haldane")
seed <- as.integer(arg("seed", "20260913"))
chrom_arg <- arg("chromosomes", paste(1:19, collapse = ","))
chromosomes <- trimws(strsplit(chrom_arg, ",", fixed = TRUE)[[1]])

if (is.null(cross_file) || is.null(out_dir)) {
  stop("usage: --cross=<json> --out=<dir> [--cores=N] [--draws=N] [--chromosomes=1,...,19]")
}
if (!file.exists(cross_file)) stop("cross file not found: ", cross_file)
if (n_draws < 1L) stop("--draws must be positive")
if (!map_function %in% c("haldane", "kosambi", "c-f", "morgan")) stop("invalid --map-function")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

t0 <- Sys.time()
cat("reading cross:", cross_file, "\n"); flush.console()
cross <- read_cross2(cross_file)
missing_chr <- setdiff(chromosomes, chr_names(cross))
if (length(missing_chr)) stop("chromosomes absent from cross: ", paste(missing_chr, collapse = ", "))
cross <- cross[, chromosomes]
ngen <- as.integer(cross$covar[, "ngen"])
ids <- ind_ids(cross)

summary_rows <- list()
draw_rows <- list()
for (chr in chromosomes) {
  sub <- cross[, chr]
  v <- viterbi(sub, sub$gmap, error_prob = error_prob, map_function = map_function,
               quiet = TRUE, cores = cores)
  v_count <- as.numeric(count_xo(v, quiet = TRUE, cores = cores)[, 1])

  set.seed(seed + as.integer(chr))
  draws <- sim_geno(sub, sub$gmap, n_draws = n_draws, error_prob = error_prob,
                    map_function = map_function, quiet = TRUE, cores = cores)
  d_count <- count_xo(draws, quiet = TRUE, cores = cores)
  # count_xo(sim_geno) is individual x chromosome x draw; retain the draw axis for uncertainty.
  dmat <- matrix(d_count[, 1, ], nrow = length(ids), ncol = n_draws)
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    id = ids, chr = chr, ngen = ngen, viterbi_xo = v_count,
    posterior_mean_xo = rowMeans(dmat),
    posterior_sd_xo = apply(dmat, 1, stats::sd),
    posterior_q025_xo = apply(dmat, 1, stats::quantile, probs = 0.025, names = FALSE),
    posterior_q975_xo = apply(dmat, 1, stats::quantile, probs = 0.975, names = FALSE),
    stringsAsFactors = FALSE
  )
  draw_rows[[length(draw_rows) + 1L]] <- data.frame(
    id = rep(ids, times = n_draws), chr = chr, ngen = rep(ngen, times = n_draws),
    draw = rep(seq_len(n_draws), each = length(ids)), xo = as.vector(dmat),
    stringsAsFactors = FALSE
  )
  rm(sub, v, draws, d_count, dmat); invisible(gc())
  cat("chr", chr, "done at", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
  flush.console()
}

summary <- do.call(rbind, summary_rows)
draw_data <- do.call(rbind, draw_rows)
utils::write.csv(summary, file.path(out_dir, "crossovers.csv"), row.names = FALSE)
con <- gzfile(file.path(out_dir, "posterior_draws.csv.gz"), open = "wt")
utils::write.csv(draw_data, con, row.names = FALSE)
close(con)

meta <- list(
  utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  analysis_status = "DEVELOPMENTAL",
  cross = normalizePath(cross_file, winslash = "/"),
  r_version = R.version.string,
  qtl2_version = as.character(utils::packageVersion("qtl2")),
  method = c("viterbi", "posterior paths from sim_geno", "native count_xo"),
  error_prob = error_prob, map_function = map_function, posterior_draws = n_draws,
  seed = seed, cores = cores, chromosomes = chromosomes,
  n_individuals = n_ind(cross),
  minutes = round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2),
  limitation = "The DO HMM uses ngen in its transition prior; this is a posterior-predictive development diagnostic, not an independent inheritance validation."
)
writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
           file.path(out_dir, "reconstruction_meta.json"))
cat("done in", meta$minutes, "min\n")

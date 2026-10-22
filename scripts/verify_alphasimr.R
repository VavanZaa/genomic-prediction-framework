# AlphaSimR Cattle Simulation Verification Script
# Runs cattle simulation with 29 autosomes, explicit sexes, dominance, and noise
suppressMessages(library(AlphaSimR))

cat("=== AlphaSimR Cattle Inheritance & Non-Additive Verification ===\n")
cat("AlphaSimR version:", as.character(packageVersion("AlphaSimR")), "\n")

set.seed(42)
out_dir <- "output/alphasimr"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 1. Founder genomes (cattle species approximation, 29 autosomes)
n_angus <- 10
n_brahman <- 10
founderGenomes <- runMacs(
    nInd = n_angus + n_brahman,
    nChr = 10, # 10 representative autosomes for quick run
    segSites = 60,
    species = "CATTLE"
)

# 2. Simulation parameters with explicit sexes, dominance, and noise
SP <- SimParam$new(founderGenomes)
SP$setSexes("yes_rand")

# Add traits: Production and Environmental Response
SP$addTraitAD(
    nQtlPerChr = 5,
    mean = 100,
    var = 25,
    meanDD = 0.25, # Directional dominance -> heterosis
    varDD = 0.10
)
SP$setVarE(h2 = 0.40)
SP$addSnpChip(nSnpPerChr = 25)

# 3. Form founder populations
angusFounders <- newPop(founderGenomes[1:n_angus])
brahmanFounders <- newPop(founderGenomes[(n_angus + 1):(n_angus + n_brahman)])

angusFounders <- setPheno(angusFounders)
brahmanFounders <- setPheno(brahmanFounders)

# 4. F1 Cross
angus_males <- angusFounders[angusFounders@sex == "M"]
brahman_females <- brahmanFounders[brahmanFounders@sex == "F"]

if (length(angus_males@id) > 0 && length(brahman_females@id) > 0) {
    f1Pop <- randCross2(females = brahman_females, males = angus_males, nCrosses = 20)
    f1Pop <- setPheno(f1Pop)

    ped_df <- data.frame(
        id = c(angusFounders@id, brahmanFounders@id, f1Pop@id),
        sire = c(angusFounders@father, brahmanFounders@father, f1Pop@father),
        dam = c(angusFounders@mother, brahmanFounders@mother, f1Pop@mother),
        sex = c(angusFounders@sex, brahmanFounders@sex, f1Pop@sex),
        phenotype = c(angusFounders@pheno[,1], brahmanFounders@pheno[,1], f1Pop@pheno[,1]),
        true_gv = c(angusFounders@gv[,1], brahmanFounders@gv[,1], f1Pop@gv[,1])
    )
    csv_path <- file.path(out_dir, "alphasimr_pedigree_phenotypes.csv")
    write.csv(ped_df, csv_path, row.names = FALSE)
    cat("AlphaSimR simulation completed successfully.\n")
    cat("Results saved to:", csv_path, "\n")
    cat("Founders:", length(angusFounders@id) + length(brahmanFounders@id), "calves:", length(f1Pop@id), "\n")
    cat("F1 Mean Genetic Value:", mean(f1Pop@gv[,1]), "\n")
} else {
    stop("Insufficient male/female founders.")
}

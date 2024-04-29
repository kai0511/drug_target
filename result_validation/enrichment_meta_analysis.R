# ------------------------------------------------------------------------------------
# This file does Enrichment t.test of different disease prediction result
# against different target genes obtained from Open Targets website.
# ------------------------------------------------------------------------------------

library(metap)

enrichment.test <- function(threshold, targets, pred){
    filted.tragets <- targets[targets$Score >= threshold,]
    merged.res <- merge(pred, filted.tragets, by = 'Name', all.x = T)
    fit <- t.test(merged.res$predRes[!is.na(merged.res$Score)], merged.res$predRes[is.na(merged.res$Score)], alternative = "less")
    return(fit$p.value)
}

compute.simulated.pval <- function(no.sim, targets, pred){
    shuffled.pred <- pred[sample(nrow(pred)), -3]
    shuffled.pred$predRes <- pred[[3]]
    return(sumlog(sapply(seq(1, 0, -0.1), enrichment.test, targets=targets, pred = shuffled.pred)))
}

OT.disease.names <- c('anxiety_disorder', 'bipolar_disorder', 'psychosis', 'scz', 'unipolar_depression')
dir.names <- c('antidepression', 'antipsychotics', 'anxiety_depression', 'scz')
models <- c('SVM', 'GBM', 'RF')
no.simulations = 1000
setwd('/exeh_3/kai/GE_result/genePerturbation')

for(directory in dir.names){
    t.test.result <- data.frame(row.names=OT.disease.names)
    for(m in models){
        print(paste0(m, '....'))
        
        OE <- read.csv(paste0(directory, '/', m, '_overexpression_', directory, '_Res.csv'), header=TRUE, stringsAsFactors=FALSE)
        KD <- read.csv(paste0(directory, '/', m, '_knockdown_', directory, '_Res.csv'), header=TRUE, stringsAsFactors=FALSE)

        pval.OE.vec <- c()
        pval.KD.vec <- c()
        
        for(d in OT.disease.names){
            print(paste0(d, '...'))
            targets <- read.table(paste0('OT_', d, '.csv'), head=T)
            # t.test.result[paste0(m, '_OE_', d)] <- sapply(seq(1, 0, -0.1), enrichment.test, targets=targets, pred = OE)
            # t.test.result[paste0(m, '_KD_', d)] <- sapply(seq(1, 0, -0.1), enrichment.test, targets=targets, pred = KD)
            real.OE.pval <- sumlog(sapply(seq(1, 0, -0.1), enrichment.test, targets=targets, pred = OE))
            real.KD.pval <- sumlog(sapply(seq(1, 0, -0.1), enrichment.test, targets=targets, pred = KD))
            
            # run simulation for both dataset
            simulated.OE.pval <- sapply(seq(no.simulations), compute.simulated.pval, targets=targets, pred = OE)
            simulated.KD.pval <- sapply(seq(no.simulations), compute.simulated.pval, targets=targets, pred = KD)
            
            final.KD.pval <- sum(unlist(simulated.KD.pval[3,]) < real.KD.pval$p) / no.simulations
            pval.KD.vec <- c(pval.KD.vec, final.KD.pval)
            
            final.OE.pval <- sum(unlist(simulated.OE.pval[3,]) < real.OE.pval$p) / no.simulations
            pval.OE.vec <- c(pval.OE.vec, final.OE.pval)
        }
        t.test.result[paste0(m, '_OE')] <- pval.OE.vec
        t.test.result[paste0(m, '_KD')] <- pval.KD.vec
    }
    write.table(t.test.result, file = paste0('enrichment_test_res/', directory, '_final_pval', '.csv'), sep=',')
}

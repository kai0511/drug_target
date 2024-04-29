# ------------------------------------------------------------------------------------
# This file does Enrichment t.test of different disease prediction result
# against different target genes obtained from Open Targets website.
# ------------------------------------------------------------------------------------

enrichment.test <- function(threshold, targets, pred){
    filted.tragets <- targets[targets$Score >= threshold,]
    merged.res <- merge(pred, filted.tragets, by = 'Name', all.x = T)
    fit <- t.test(merged.res$predRes[!is.na(merged.res$Score)], merged.res$predRes[is.na(merged.res$Score)], alternative = "two.sided")
    return(fit$p.value)
}

# OT.disease.names <- c('anxiety_disorder', 'bipolar_disorder', 'psychosis', 'scz', 'unipolar_depression')
OT.disease.names <- c('arthritis')
dir.names <- c('arthritis')
# dir.names <- c('antidepression', 'antipsychotics', 'anxiety_depression', 'scz')
models <- c('EN', 'SVM', 'GBM', 'RF')
# models <- c('SVM')

setwd('/exeh_3/kai/GE_result/genePerturbation')

for(directory in dir.names){
    for(m in models){
        print(paste0(m, '....'))
        
        OE <- read.csv(paste0(directory, '/', m, '_overexpression_', directory, '_Res.csv'), header=TRUE, stringsAsFactors=FALSE)
        KD <- read.csv(paste0(directory, '/', m, '_knockdown_', directory, '_Res.csv'), header=TRUE, stringsAsFactors=FALSE)
        
        df.names <- c()
        t.test.result = data.frame(row.names=seq(1, 0, -0.1))
        
        for(d in OT.disease.names){
            print(paste0(d, '...'))
            targets <- read.table(paste0('OT_', d, '.csv'), head=T)
            t.test.result[[paste0(m, '_OE_', d)]] = sapply(seq(1, 0, -0.1), enrichment.test, targets=targets, pred = OE)
            t.test.result[[paste0(m, '_KD_', d)]] <- sapply(seq(1, 0, -0.1), enrichment.test, targets=targets, pred = KD)
        }
        write.table(t.test.result, file = paste0('enrichment_test_res/', directory, '_', m, '1.csv'), sep=',')
    }
}

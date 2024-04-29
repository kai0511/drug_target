library(doMC)
library(methods)
library(Metrics)
library(glmnet)
library(precrec)
library(rPython)
registerDoMC(cores = 4)

# format results from Elastic Net with given alpha
optimal.var <- function(alpha, train_X, train_y, sample_weights, foldid){
    
    glmnet.obj <- cv.glmnet(
        x = train_X,
        y = train_y, 
        weights = sample_weights, 
        # nfolds = nfold,
        foldid = foldid,
        alpha = alpha,
        family = 'binomial',     
        type.measure = "deviance", 
        standardize = TRUE, 
        parallel = TRUE, 
        keep = TRUE)
    return(list(glmnet=glmnet.obj, deviance=min(glmnet.obj$cvm)))
}

get.foldid <- function(foldid.len, splits){
    foliid <- vector(mode = "integer", length = foldid.len)
    splits.len <- length(splits)
    # apply(seq(splits.len), function(x) )
    for( i in seq(splits.len)){
        foliid[splits[[i]][[2]]+1] = i
    }
    return(foliid)
}

setwd('/exeh_3/kai/GE_result/genePerturbation')
nfold = 5
alpha.vec = seq(0,1,0.1)
logLoss = c()
rocScore = c()
prcScore = c()
models = list()

pheno = read.csv('cmap_drug_expression_profile.csv', header = TRUE, stringsAsFactors=FALSE, check.names = FALSE)
python.load('/exeh_3/kai/python_code/genePerturbation/stratifiedSplits.py')

# obtain indications 
indication_list = list(arthritis='/exeh_3/kai/data/backups/MEDI_Rheumatoid_arthritis.csv')
# indication_list = list(diabetes='/exeh_3/kai/data/backups/ATC_diabetes.csv',
#                        arthritis='/exeh_3/kai/data/backups/MEDI_Rheumatoid_arthritis.csv',
#                        hypertension='/exeh_3/kai/data/backups/ATC_hypertension.csv',
#                        scz='/exeh_3/kai/data/backups/MEDI_scz.csv')
# indication_list = list(diabetes='/exeh_3/kai/data/backups/ATC_diabetes.csv')
# indication_list = list(hypertension='/exeh_3/kai/data/backups/ATC_hypertension.csv')
# indication_list = list(arthritis='/exeh_3/kai/data/backups/MEDI_Rheumatoid_arthritis.csv')

for(n in names(indication_list)){
    print(paste0('Begin computation for ', n, ':'))
    drugList <- read.csv(indication_list[[n]], header=FALSE, stringsAsFactors=FALSE)
    idx <- grep(paste(trimws(tolower(drugList[[1]])), collapse = '|'), pheno$drugName, ignore.case = TRUE)
    Indication <- rep(0, dim(pheno)[1])
    Indication[idx] <- 1

    # add a new column of Indication to drugBank
    # pheno$Indication <- Indication
    
    # adjust the order of columns
    # col_names <- names(pheno)
    # new.order <- c('Indication', col_names[-length(col_names)])
    # pheno <- pheno[, new.order]
    
    # obtain X and y from cmap data
    X <- as.matrix(pheno[, c(-1,-2)])
    y <- as.matrix(Indication)
    drug_names <- pheno[, 1] 

    python.assign('y', Indication)
    python.exec('y_orig = np.asarray(y)')
    python.exec('splits = getSplits(X_orig, y_orig)')
    splits = python.get('splits')
    
    # compute sample weights
    obs_num <- length(y)
    pos_obs_num <- sum(y)
    neg_obs_num <- obs_num - pos_obs_num
    pos_weight <- obs_num / (2 * pos_obs_num)
    neg_weight <- obs_num / (2 * neg_obs_num)
    samp_weights <- rep(neg_weight, obs_num)
    samp_weights[y == 1] <- pos_weight
    
    for(i in seq(length(splits))){
        train_index = splits[[i]][[1]] + 1  # as the index in Python begins with 0, add 1 to be compatible with R
        test_index = splits[[i]][[2]] + 1
        
        python.assign('train_index', splits[[i]][[1]])
        python.exec('train_splits = getSplits(X_orig[train_index], y_orig[train_index])')
        train_splits <- python.get('train_splits')
        
        foldid = get.foldid(length(train_index), train_splits)
        
        X_train <- X[train_index,]
        y_train <- y[train_index,]
        X_test <- X[test_index,]
        y_test <- y[test_index,]
        drug_names_test <- drug_names[test_index]
        train_weights <- samp_weights[train_index]
        
        # run cv.glmnet
        var.list <- lapply(alpha.vec, optimal.var, train_X = X_train, train_y = y_train, foldid = foldid, sample_weights = train_weights)
        min.pos <- which.min(unlist(lapply(var.list, function(e) e$deviance)))
        
        # print best parameters from training
        print(paste0("[optimal parameters] alpha: ", min.pos/10, ', corresponding deviance: ', var.list[[min.pos]]$deviance, ', corresponding lambda: ', var.list[[min.pos]]$glmnet$lambda.min,'.'), quote=FALSE)
        
        # compute evaluation metrics
        cv.glmnet.fit <- var.list[[min.pos]]$glmnet
        y_pred <- predict(cv.glmnet.fit, X_test, s="lambda.min", type='response')
        result <- cbind(drug_names_test, y_test, y_pred)
        write.table(result, file = paste0('en_', n, '_result.csv'), row.names = F, append = T)
        logLoss[i] = logLoss(y_test, y_pred)
        curves = evalmod(scores = y_pred, labels = y_test)
        scores = auc(curves)$aucs
        rocScore[i] = scores[1]
        prcScore[i] = scores[2]
        models[[i]] = cv.glmnet.fit
    }
    
    # print evaluation result
    print(paste0('log Loss for ', n, ' :', paste0(logLoss, collapse=','), ', its mean:', mean(logLoss), '.'), quote = FALSE)
    print(paste0('AUC-ROC for ', n, ' :', paste0(rocScore, collapse=','), ', its mean:', mean(rocScore), '.'), quote = FALSE)
    print(paste0('AUC-PRC for ', n, ' :', paste0(prcScore, collapse=','), ', its mean:', mean(prcScore), '.'), quote = FALSE)
    
    # find the optimal model
    index = which.min(logLoss)
    glmnet.model = models[[index]]
    
    # make predictions on knockdown and overexpression data
    knockdown <- read.table('consensi-knockdown.tsv', header=TRUE)
    overexpression <- read.table('consensi-overexpression.tsv', header=TRUE)
    # all.perturbation <- read.table('consensi-pert_id.tsv', header=TRUE)
    predicted.knockdown <- predict(glmnet.model, data.matrix(knockdown[,-1]), s="lambda.min", type='response')
    predicted.overexpression <- predict(glmnet.model, data.matrix(overexpression[,-1]), s="lambda.min", type='response')
    # predicted.all <- predict(glmnet.model, data.matrix(all.perturbation[,-1]), s="lambda.min", type='response')
    
    # obtain gene name for each perturbation dataset
    python.assign('kd_gene_id', knockdown[, 1])
    python.exec('kd_gene_name = getGeneName(kd_gene_id)')
    kd_gene_name = python.get('kd_gene_name')
    
    python.assign('oe_gene_id', overexpression[, 1])
    python.exec('oe_gene_name = getGeneName(oe_gene_id)')
    oe_gene_name = python.get('oe_gene_name')
    
    # save results
    knockdown.res <- data.frame(Id = knockdown[, 1], Name = kd_gene_name, predRes = predicted.knockdown[,1])
    overexpression.res <- data.frame(Id = overexpression[, 1], Name = oe_gene_name, predRes = predicted.overexpression[,1])
    # all.res <- data.frame(perturbationID = all.perturbation[, 1], pred = predicted.all[,1])
    write.table(knockdown.res, paste0(n, '/EN_knockdown_', n, '_Res.csv'), quote=FALSE, sep=',', row.names=FALSE, col.names = TRUE)
    write.table(overexpression.res, paste0(n, '/EN_overexpression_', n, '_Res.csv'), quote=FALSE, sep=',', row.names=FALSE, col.names = TRUE)
    # write.table(all.res, paste0(n, '/EN_all_perturbation_', n,'_res.csv'), quote=FALSE, sep=',', row.names=FALSE, col.names = FALSE)
    print(paste0('Computation for ', n, ' is end!'))
}


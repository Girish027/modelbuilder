from numpy.lib.function_base import iterable
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
import math
import os
import concurrent.futures
import json
import ModelMetrics
from collections import defaultdict
import pickle,bz2
from multiprocessing import Pool,cpu_count
import pprint
from config import data

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC


def perform_grid_search(X,y):

    penalty = 'l2'
    loss = 'squared_hinge'
    dual = True
    multi_class = 'ovr'
    class_weight = 'balanced'
    max_iter = 100

    clf_svm = LinearSVC(penalty=penalty, loss=loss, dual=dual, multi_class=multi_class, class_weight=class_weight, max_iter=max_iter)
    #pprint.pprint(clf_svm.get_params(), indent=4)

    c_min=0.1
    c_multiplication_factor = 10
    c_entries = 3
    scoring_metric = 'f1_macro'

    c_values = [c_min*c_multiplication_factor**i for i in range(c_entries)]
    # param_grid = {'model__C':[1, 10, 100, 1000]}
    param_grid = {'C':c_values}

    grid_search = GridSearchCV(clf_svm,param_grid,n_jobs=-1,cv=3, scoring=scoring_metric)
    #pprint.pprint(grid_search.get_params(), indent=4)
    grid_search.fit(X,y)
    return grid_search

def get_scores_for_sigmoid_training_single_fold(train_index,test_index,X,y,final_params):
    X_train,y_train = X[train_index],y[train_index]
    X_test,y_test = X[test_index],y[test_index]
    clf_svm = LinearSVC(**final_params)
    clf_svm.fit(X_train,y_train)
    y_test_scores = clf_svm.decision_function(X_test)
    if(len(clf_svm.classes_) == 2):
        y_test_scores_new = []
        for i in range(len(y_test_scores)):
            y_test_scores_new.append([y_test_scores[i],-1*y_test_scores[i]])
        y_test_scores = y_test_scores_new
    y_test_pred = clf_svm.predict(X_test)
    return y_test_scores,y_test_pred,y_test,test_index

def get_scores_for_sigmoid_training(X,y,final_params, number_of_folds_Sgmd_Training):
    y_actual = []
    y_pred_scores = []
    y_pred = []
    indexes_order = []
    skf = StratifiedKFold(n_splits=number_of_folds_Sgmd_Training)

    iterable_to_pass = ((train_index,test_index,X,y,final_params) for train_index,test_index in skf.split(X,y))
    results = []
    for args in iterable_to_pass:
        results.append(get_scores_for_sigmoid_training_single_fold(*args))
    for y_test_scores,y_test_pred,y_test,test_index in results:
        y_pred_scores.extend(y_test_scores)
        y_pred.extend(y_test_pred)
        y_actual.extend(y_test)
        indexes_order.extend(test_index)
    return y_pred_scores,y_pred,y_actual,indexes_order

def write_sigmoid_scores(y_pred_scores,y_actual,sklearn_folder):
    prediction_out_file = os.path.join(sklearn_folder,'pred_score.txt')
    with open(prediction_out_file,'w',encoding='utf-8') as of:
        for row,intent in zip(y_pred_scores,y_actual):
            of.write(intent+'\t' + ','.join(map(str,row))+'\n')

def sigmoid_train(len,dec_values=[],labels=[]):
    # A = 0.0
    # B = 0.0
    prior1 = 0.0
    prior0 = 0.0
    for i in range(len):
        if (labels[i] > 0.0):
            prior1 += 1.0
        else :
            prior0 += 1.0

    max_iter = 100
    min_step = float(1e-10)
    sigma = float(1e-12)
    eps = float(1e-5)
    hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
    loTarget = 1 / (prior0 + 2.0)
    t = []
    # fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize = 0.0
    # newA, newB, newf, d1, d2 = 0.0
    # iter = 0

    A = 0.0
    B = math.log((prior0 + 1.0) / (prior1 + 1.0))
    fval = 0.0

    for i in range(len):
        if (labels[i] > 0.0) :
            t.append(hiTarget)
        else :
            t.append(loTarget)
        fApB = dec_values[i] * A + B
        if (fApB >= 0.0):
            fval += t[i] * fApB + math.log(1+math.exp(-fApB))
        else :
            fval += (t[i] - 1) * fApB + math.log(1+math.exp(fApB))

    for iter in range(max_iter):
        h11 = sigma
        h22 = sigma
        h21 = 0.0
        g1 = 0.0
        g2 = 0.0
        for i in range(len):
            fApB = dec_values[i] * A + B
            if (fApB >= 0.0):
                p=math.exp(-fApB) / (1.0+math.exp(-fApB))
                q=1.0 / (1.0+math.exp(-fApB))
            else:
                p=1.0 / (1.0+math.exp(fApB))
                q=math.exp(fApB) / (1.0+math.exp(fApB))
            d2=p * q
            h11 += dec_values[i] * dec_values[i] * d2
            h22 += d2
            h21 += dec_values[i] * d2
            d1=t[i]-p
            g1 += dec_values[i] * d1
            g2 += d1
        if (abs(g1) < eps) and (abs(g2) < eps):
            break
        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB
        stepsize = 1
        while (stepsize >= min_step):
            newA = A + stepsize * dA
            newB = B + stepsize * dB
            newf = 0.0
            for i in range(len):
                fApB = dec_values[i] * newA+newB
                if fApB >= 0:
                    newf += t[i] * fApB + math.log(1+math.exp(-fApB))
                else:
                    newf += (t[i] - 1) * fApB +math.log(1+math.exp(fApB))
            if newf < (fval+0.0001 * stepsize * gd):
                A=newA
                B=newB
                fval=newf
                break
            else:
                stepsize = stepsize / 2.0
        if (stepsize < min_step):
            break
    probAB = [A]
    probAB.append(B)
    return probAB

def sigmoid_predict(decision_value,A,B):
    fApB = decision_value*A+B
    if (fApB >= 0):
        return math.exp(-fApB)/(1.0+math.exp(-fApB))
    else:
        return 1.0/(1+math.exp(fApB))

def generate_web2nl_svm_model(svm_model,sigmoid_params,embUrl,model_folder,default_intent,postProcessingRules):

    # Writing model type
    model_file = os.path.join(os.path.join(model_folder,"final_model/web2nl"),"final.model")
    if not os.path.exists(model_folder+ "/final_model/web2nl"):
            os.system("mkdir -p " + model_folder + "/final_model/web2nl")
    f = open(model_file,"w")
    f.write("# Copyright 2017 [24]7.ai, Inc. ALL RIGHTS RESERVED. DO NOT DISTRIBUTE. This is an unpublished \n# proprietary work which is fully protected under copyright law. This code may \n# only be used pursuant to a valid license from [247].ai, Inc. \n")
    f.write("# Machine generated (Modeling Workbench), do not edit! For any issues or concerns, drop email at mwb-team@247.ai \n\n")
    f.write("[fil,file-type,1]\n")
    f.write("0,classifier-model\n\n")

    f.write("[algo,algorithm,1]\n")
    f.write("0,svm,2.0\n\n")

    f.write("[feat,feature-type,1]\n")
    f.write(f"0,emb,{embUrl}\n\n")

    f.write("[tfx,tfx-serving,2]\n")
    f.write("0,input-format,row\n")
    f.write("1,model-output-signature,serving_default\n\n")

    f.write(f"[cl,classes,{len(svm_model.classes_)}]\n")
    for i in range(len(svm_model.classes_)):
        if(len(svm_model.classes_) == 2 and i == 1):
            f.write(f"{i},{svm_model.classes_[i]},{round(-1*svm_model.intercept_[0],5)}\n")
        else:
            f.write(f"{i},{svm_model.classes_[i]},{round(-1*svm_model.intercept_[i],5)}\n")

    f.write(f"\n[ithr,intent-thresholds,{len(svm_model.classes_)}]\n")
    for i in range(len(svm_model.classes_)):
        f.write(f"{i},0.0\n") # todo this might not be necessary check after web2nl changes

    f.write("\n[def,default-class,1]\n")
    default_intent_index = svm_model.classes_.tolist().index(default_intent)
    f.write(f"0,{default_intent_index}\n\n") # todo this might not be necessary check after web2nl changes

    f.write("[xforms,transforms,0]\n")

    if(postProcessingRules !=  None and len(postProcessingRules) > 0):
        f.write(f"\n[postproc,postprocess-intents,{len(postProcessingRules)}]\n")
        for i in range(len(postProcessingRules)):
            inputMatch = postProcessingRules[i]["input-match"]
            intentMatch = str(postProcessingRules[i]["intent-match"]).replace("', '", " ").replace("['","[").replace("']","]")
            intentReplacement = postProcessingRules[i]["intent-replacement"]
            f.write(f"{i},{inputMatch},{intentMatch},{intentReplacement}\n")


    f.write(f"\n[sig,sigmoid-params,{len(sigmoid_params)}]\n")
    for i in range(len(sigmoid_params)):
        f.write(f"{i},{round(sigmoid_params[i][0],5)},{round(sigmoid_params[i][1],5)}\n") # trim to 5 decimal

    coef = svm_model.coef_
    if(len(svm_model.classes_) == 2):
        coef_0 = coef[0]
        coef_1 = []
        for i in range(len(coef_0)):
            coef_1.append(-1*coef_0[i])
        coef = [coef_0,coef_1]
    coef_dict = pd.DataFrame(coef).to_dict()
    f.write(f"\n[embeds,embeddings,{len(coef_dict)}]\n")
    # todo trim the coef precision to 5 digits
    for i in range(len(coef_dict)):
        precision_reduced_dict = {key : round(coef_dict[i][key], 5) for key in coef_dict[i]}
        row_coef = str(precision_reduced_dict).strip("{").strip("}").replace(' ','')
        f.write(f"{i},{row_coef}\n")
    print('Generated web2nl model')

def save_sklearn_model(model,model_path):
    with bz2.BZ2File(model_path,'wb') as model_file:
        pickle.dump(model,model_file)

def write_model_output(clf_svm,sklearn_folder):
    if not os.path.isdir(sklearn_folder):
        os.makedirs(sklearn_folder)

    classes_file = os.path.join(sklearn_folder,'classes.txt')
    coeff_file = os.path.join(sklearn_folder,'coeff_out.txt')
    bias_file = os.path.join(sklearn_folder,'bias_out.txt')

    with open(classes_file,'w',encoding='utf-8') as of:
        of.write('\n'.join(clf_svm.classes_))

    ## writing svm coeffs in the format feature_index:feature_weight
    ## each line contains entry for one intent (one one-vs-rest model)
    with open(coeff_file,'w',encoding='utf-8') as of:
        for coeffs in clf_svm.coef_:
            of.write(','.join("{}:{}".format(i,v) for i,v in enumerate(coeffs) if v!=0)+'\n')

    with open(bias_file,'w',encoding='utf-8') as of:
        of.write('\n'.join(map(str,clf_svm.intercept_)))


def perform_expt_single_fold(train_index,test_index,fold_index,X,y,df,final_params, sklearn_folder, number_of_folds_Sgmd_Training):
    sklearn_folds_dir = os.path.join(sklearn_folder,'folds',str(fold_index))
    if not os.path.isdir(sklearn_folds_dir):
            os.makedirs(sklearn_folds_dir)
    X_train,y_train = X[train_index],y[train_index]
    X_test,y_test = X[test_index],y[test_index]

    clf_svm = LinearSVC(**final_params)
    clf_svm.fit(X[train_index],y[train_index])
    foldwise_model_path= os.path.join(sklearn_folds_dir,'sklearn_model.bz2')
    save_sklearn_model(clf_svm,foldwise_model_path)
    write_model_output(clf_svm,sklearn_folds_dir)

    ##writing these outputs to a file to calculate sigmoid parameters
    ##have to save (ids) indexes present in this test so that classification output can be formed from this
    unique_ids_file=os.path.join(sklearn_folds_dir,'unique_ids.txt')
    with open(unique_ids_file,'w') as of:
        for line in df['filename'][test_index]:
            of.write(line+'\n')
    y_pred_scores,y_pred,y_actual,indexes_order = get_scores_for_sigmoid_training(X_train,y_train,clf_svm.get_params(), number_of_folds_Sgmd_Training)
    
    y_pred_df = pd.DataFrame(y_pred_scores,columns=clf_svm.classes_.tolist())

    # Computing sigmoid train params for each intent
    sigmoid_params = []
    for idx in range(y_pred_df.columns.size):
        sigmoid_data = y_pred_df.iloc[:,idx].tolist()
        label = []
        for i in range(len(sigmoid_data)):
            # todo compare the column value of intent and row value of intent match then give 1 else 0 -- DONE
            if y_pred_df.columns[idx] == y_actual[i]:
                label.append(float(1))
            else:
                label.append(float(0))
        sigmoid_params.append(sigmoid_train(len(sigmoid_data),sigmoid_data,label))
    sigmoid_params_df = pd.DataFrame(sigmoid_params,columns=["a","b"])
    sigmoid_params_df['intent']=y_pred_df.columns
    sigmoid_params_df.to_csv(f'{sklearn_folds_dir}/sigmoidParams.csv',index=False)

    write_sigmoid_scores(y_pred_scores,y_actual,sklearn_folds_dir)

    report_types = json.loads('{"internal":["I","A","E"],"external":["E"]}')
    report_wise_output  = ModelMetrics.parse_sigmoid_params_and_generate_classification_out(df,sklearn_folds_dir,
            sklearn_folder, fold_index, report_types)
    
    ## saving foldwise train & test data into the main sklearn folder
    columns_to_pick = list(df.columns)
    columns_to_pick.remove('embedding')
    df_train_foldwise = df.iloc[train_index]
    df_test_foldwise = df.iloc[test_index]
    df_train_foldwise[columns_to_pick].to_csv(os.path.join(sklearn_folder,'train_{}'.format(fold_index)),sep='\t',header=True,index=False)
    df_test_foldwise[columns_to_pick].to_csv(os.path.join(sklearn_folder,'test_{}'.format(fold_index)),sep='\t',header=True,index=False)
    return report_wise_output

def get_pool(no_parallel_executions):
    ##till python=3.8 multiprocessing does't share memory,even passing pool object didn't work
    ##hence creating another pool object here
    ##getting min of cpu count & no_of_parallel_executions passed
    no_parallel_executions = cpu_count() if cpu_count()<no_parallel_executions else no_parallel_executions
    # no_parallel_executions = 10
    pool = Pool(no_parallel_executions)
    return pool

def perform_cv_for_performance_metrics(X,y,df,final_params,sklearn_folder):

    skf = StratifiedKFold(n_splits=3)
    row_types_series = df['row_type']
    stratify_key_list = [f'{intent}_{row_type}'for intent,row_type in  zip(y,row_types_series)]
    number_of_folds_Sgmd_Training = 3
    iterable_to_pass = ((train_index,test_index,fold_index,X,y,df,final_params, sklearn_folder,
        number_of_folds_Sgmd_Training) for fold_index,(train_index,test_index) in enumerate(skf.split(X,stratify_key_list)))
    pool = get_pool(3)
    report_wise_output = pool.starmap(perform_expt_single_fold,iterable_to_pass)
    # NO_OF_WORKER_THREADS = data.get("NO_OF_WORKER_THREADS",20)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=NO_OF_WORKER_THREADS) as executor:
    #             report_wise_out = executor.map(perform_expt_single_fold,iterable_to_pass)
    return report_wise_output
